"""CPU checks for the accuracy protocol and comparison failure modes."""

import json
from types import SimpleNamespace

import pytest

from benchmarks.gsm8k import build_records, compare, file_digest, make_task, score, select_doc_ids


@pytest.fixture
def task(monkeypatch):
    pytest.importorskip("lm_eval")
    from datasets import Dataset, DatasetDict
    from lm_eval.api.task import ConfigurableTask

    def download(self, *args, **kwargs):
        self.dataset = DatasetDict(
            {
                split: Dataset.from_dict(
                    {
                        "question": [f"What is 6 times 3? Question #{i}" for i in range(150)],
                        "answer": ["#### 18"] * 150,
                    }
                )
                for split in ("train", "test")
            }
        )

    monkeypatch.setattr(ConfigurableTask, "download", download)
    return make_task()


def test_paper_prompt_and_strict_scoring(task):
    task.build_all_requests(limit=1, rank=0, world_size=1)
    instance = task.instances[0]
    assert instance.args[0].count("Q:") == 4  # three demonstrations plus the question
    row = {"doc": instance.doc}
    assert score(task, row, "6 * 3 = 18. The answer is 18.")["correct"]
    # A bare number must not silently pass the paper's strict extraction rule.
    assert score(task, row, "18")["unparseable"]
    assert not score(task, row, "The answer is 19.")["correct"]


def test_sample_is_fixed_and_does_not_use_the_first_hundred():
    selected = select_doc_ids(1319, limit=100, seed=0, split="test")
    assert len(selected) == len(set(selected)) == 100
    assert selected == sorted(selected)
    assert selected == select_doc_ids(1319, limit=100, seed=0, split="test")
    assert selected != list(range(100))
    assert selected != select_doc_ids(1319, limit=100, seed=1, split="test")
    assert select_doc_ids(1319, limit=None, seed=0, split="test") == list(range(1319))


@pytest.mark.parametrize("population,limit", [(10, 100), (10, 0), (0, None)])
def test_selection_rejects_missing_coverage(population, limit):
    with pytest.raises(ValueError):
        select_doc_ids(population, limit=limit, seed=0, split="test")


def test_sampled_requests_keep_original_dataset_ids(task):
    tokenizer = SimpleNamespace(encode=lambda prompt, **kwargs: [1, 2, 3])
    records, selection = build_records(
        task, tokenizer, split="test", limit=100, seed=0, max_new_tokens=10, max_length=20
    )
    assert len(records) == 100
    assert [row["id"] for row in records] == selection["ids"]
    for row in records:
        assert row["doc"]["question"] == f"What is 6 times 3? Question #{row['id']}"
        assert row["prompt"].count("Q:") == 4
    with pytest.raises(ValueError, match="truncated"):
        build_records(
            task, tokenizer, split="test", limit=100, seed=0, max_new_tokens=20, max_length=20
        )


def comparison_fixture(tmp_path, native_correct=True):
    for backend, correct in (("transformers", True), ("native", native_correct)):
        path = tmp_path / backend
        path.mkdir()
        row = {
            "id": 0,
            "prompt_sha256": "same",
            "correct": correct,
            "answer": "18" if correct else "19",
        }
        (path / "samples.jsonl").write_text(json.dumps(row) + "\n")
        summary = {
            "backend": backend,
            "protocol_sha256": "same",
            "expected_examples": 1,
            "split": "test",
            "max_regression_pp": 1.0,
            "min_reference_accuracy_pct": 75.92,
            "complete": True,
            "examples": 1,
            "correct": int(correct),
            "accuracy": float(correct),
            "samples_sha256": file_digest(path / "samples.jsonl"),
        }
        (path / "summary.json").write_text(json.dumps(summary))
    return SimpleNamespace(
        transformers=tmp_path / "transformers",
        native=tmp_path / "native",
        output=tmp_path / "comparison.json",
    )


def test_accuracy_regression_fails(tmp_path):
    result = compare(comparison_fixture(tmp_path, native_correct=False))
    assert result["delta_pp"] == -100
    assert result["reference_correct_native_wrong"] == 1
    assert not result["passes_observed_accuracy_gate"]


def test_measured_transformers_score_is_the_default_baseline(tmp_path):
    args = comparison_fixture(tmp_path, native_correct=False)
    for backend in ("transformers", "native"):
        folder = tmp_path / backend
        samples = folder / "samples.jsonl"
        row = json.loads(samples.read_text())
        row.update(correct=False, answer="19")
        samples.write_text(json.dumps(row) + "\n")
        path = folder / "summary.json"
        summary = json.loads(path.read_text())
        summary.update(
            correct=0,
            accuracy=0.0,
            min_reference_accuracy_pct=None,
            samples_sha256=file_digest(samples),
        )
        path.write_text(json.dumps(summary))
    result = compare(args)
    assert result["transformers_correct"] == result["native_correct"] == 0
    assert result["delta_pp"] == 0
    assert result["passes_observed_accuracy_gate"]


@pytest.mark.parametrize("change", ["different_protocol", "partial", "wrong_score", "duplicate"])
def test_invalid_comparisons_rejected(tmp_path, change):
    args = comparison_fixture(tmp_path)
    path = args.native / "summary.json"
    summary = json.loads(path.read_text())
    if change == "different_protocol":
        summary["protocol_sha256"] = "different"
    elif change == "partial":
        summary["complete"] = False
    elif change == "wrong_score":
        summary["accuracy"] = 0.0
    else:
        rows = args.native / "samples.jsonl"
        rows.write_text(rows.read_text() * 2)
        summary.update(examples=2, samples_sha256=file_digest(rows))
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        compare(args)
