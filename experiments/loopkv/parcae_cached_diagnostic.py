"""Observe pinned Parcae cache arithmetic with writable legacy cache properties."""

from unittest.mock import patch

import torch

from vllm_rlt.core.kv_cache_manager import KVCacheManager


def compatible_cache(config):
    from parcae_lm.utils.cache import ParcaeDynamicCache

    class WritableCache(ParcaeDynamicCache):
        @property
        def key_cache(self) -> dict[int, dict[int, torch.Tensor]]:
            return self._keys

        @key_cache.setter
        def key_cache(self, values: dict[int, dict[int, torch.Tensor]]) -> None:
            self._keys = values

        @property
        def value_cache(self) -> dict[int, dict[int, torch.Tensor]]:
            return self._values

        @value_cache.setter
        def value_cache(self, values: dict[int, dict[int, torch.Tensor]]) -> None:
            self._values = values

    assert WritableCache.__init__ is ParcaeDynamicCache.__init__
    assert WritableCache.update is ParcaeDynamicCache.update
    assert WritableCache.reset is ParcaeDynamicCache.reset
    assert WritableCache.get_seq_length is ParcaeDynamicCache.get_seq_length
    prelude, core = config.n_layers_in_prelude, config.n_layers_in_recurrent_block
    return WritableCache(
        core_step_range=(prelude, prelude + config.mean_recurrence * core), n_core=core
    )


def comparison(expected, actual, atol, rtol):
    assert expected.shape == actual.shape and expected.dtype == actual.dtype
    delta = (expected.float() - actual.float()).abs()
    failed = (delta > atol + rtol * expected.float().abs()) | ~torch.isfinite(delta)
    return {
        "max_abs": float(delta.max()),
        "failed_elements": int(failed.sum()),
        "elements": expected.numel(),
        "bitwise_equal": torch.equal(
            expected.contiguous().view(torch.uint8), actual.contiguous().view(torch.uint8)
        ),
    }


def aggregate(checks):
    return {
        "max_abs": max(row["max_abs"] for row in checks),
        "failed_elements": sum(row["failed_elements"] for row in checks),
        "elements": sum(row["elements"] for row in checks),
        "bitwise_equal": all(row["bitwise_equal"] for row in checks),
        "comparisons": len(checks),
        "first_failure": next((row for row in checks if row["failed_elements"]), None),
    }


def cached_comparison(native, original, tokens, prefill, atol, rtol):
    config = native.config
    dtype = native.transformer.wte.weight.dtype
    author_cache = compatible_cache(config)
    blocks = max(128, config.mean_recurrence * ((len(tokens) + 1) // 2))
    cache = KVCacheManager(
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        blocks,
        2,
        max_loops=config.mean_recurrence,
        dtype=dtype,
        recurrent_layers=native.recurrent_kv_layers,
    )
    assert cache.allocate("cached", len(tokens))
    prelude, core = config.n_layers_in_prelude, config.n_layers_in_recurrent_block
    locations = [(layer, layer, 0) for layer in range(prelude)]
    locations += [
        (prelude + depth * core + layer, prelude + layer, depth)
        for depth in range(config.mean_recurrence)
        for layer in range(core)
    ]
    locations += [
        (prelude + config.mean_recurrence * core + layer, prelude + core + layer, 0)
        for layer in range(config.n_layers_in_coda)
    ]
    chunks = []
    for index, positions in enumerate([list(range(prefill)), [prefill], [prefill + 1]]):
        initial, states = [], []

        def initialize(value):
            state = initialize_original(value)
            initial.append(state[0].clone())
            return state

        initialize_original = original.initialize_state
        hook = original.transformer.core_block[-1].register_forward_hook(
            lambda _module, _inputs, output: states.append(output[0].clone())
        )
        torch.manual_seed(109 + index)
        with patch.object(original, "initialize_state", initialize):
            expected = original.forward_for_generation(
                tokens[positions][None], past_key_values=author_cache
            )["logits"][0]
        hook.remove()
        assert len(initial) == 1 and len(states) == config.mean_recurrence
        boundary = cache._prepare_batch(
            ["cached"] * len(positions), [0] * len(positions), positions
        )
        torch.manual_seed(109 + index)
        hidden = native.prelude_prepared(tokens[positions], boundary, cache)
        initial_exact = torch.equal(initial[0], hidden[:, : config.n_embd])
        state_checks = []
        for depth, wanted in enumerate(states):
            recurrent = cache._prepare_batch(
                ["cached"] * len(positions), [depth] * len(positions), positions
            )
            hidden, _ = native.recurrent_prepared(hidden, recurrent, cache)
            state_checks.append(
                dict(comparison(wanted, hidden[:, : config.n_embd], atol, rtol), depth=depth)
            )
        for position in positions:
            cache.finalize_token("cached", position, config.mean_recurrence - 1)
        actual = native.coda_prepared(hidden, boundary, cache)
        length = positions[-1] + 1
        assert author_cache.get_seq_length() == length
        assert (
            set(author_cache.key_cache)
            == set(author_cache.value_cache)
            == {step for step, _, _ in locations}
        )
        kv_checks = []
        for step, layer, depth in locations:
            for kind, source, observed in zip(
                ("key", "value"),
                (author_cache.key_cache, author_cache.value_cache),
                cache.read(layer, "cached", depth, length),
                strict=True,
            ):
                assert list(source[step]) == list(range(length))
                wanted = torch.stack(list(source[step].values()), dim=1)[0]
                kv_checks.append(
                    dict(
                        comparison(wanted, observed, atol, rtol),
                        step=step,
                        layer=layer,
                        depth=depth,
                        kind=kind,
                    )
                )
        chunks.append(
            {
                "positions": positions,
                "initial_state_exact": initial_exact,
                "logits": comparison(expected, actual, atol, rtol),
                "states": aggregate(state_checks),
                "kv": aggregate(kv_checks),
                "greedy_differences": (expected.argmax(-1) != actual.argmax(-1))
                .nonzero()
                .flatten()
                .tolist(),
                "author_cache_length": length,
            }
        )
    cache.free("cached")
    author_cache.reset()
    assert cache.num_free_blocks == blocks
    assert (
        author_cache.get_seq_length() == 0
        and not author_cache.key_cache
        and not author_cache.value_cache
    )
    return {
        "dtype": str(dtype),
        "prefill": prefill,
        "forced_decode": 2,
        "atol": atol,
        "rtol": rtol,
        "chunks": chunks,
        "caches_drained": True,
        "native_close": all(
            row["initial_state_exact"]
            and not any(row[field]["failed_elements"] for field in ("logits", "states", "kv"))
            for row in chunks
        ),
    }
