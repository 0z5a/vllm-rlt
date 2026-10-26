# Re:Loop measured baseline scope

P4D3 means four recurrent passes during prefill and three during decoding.
P4D4 performs four in both phases. Guided D3 uses reference loop 1,
two heads and fixed strength 0.3; its matched D3-off arm disables guidance.
The common R4 KV pool remains allocated for every arm.

| Existing measurement | D4 off | D3 off | D3 guided | Comparison scope |
|---|---:|---:|---:|---|
| Fixed-output matched speed improvement | baseline | +15.9342% | +14.2596% | Same vllm-rlt, P512/output128, B16/initial total C32, observed A=S16 |
| Development correct /128 | 80 | 79 | 80 | Development screen; final quality remains unconfirmed |
| Development natural elapsed seconds | 113.162 | 89.428 | 166.367 | Variable output lengths and task stops |
| Development output tokens | 13234 | 13079 | 15222 | Guided D3 includes two 1024-token limit hits |

The fixed-output speed ratios use ten matched measured repetitions in two
fresh rounds on the earlier physical H20, FP32/Triton eager and a common
64 GiB R4 KV pool. Guidance adds 1.475311% elapsed time versus D3 off,
128 KiB peak allocation and 2 MiB reserved memory. It provides no measured
VRAM saving in this case. See the [complete cost report](../../../docs/reloop-quality-cost.md)
and [raw latency provenance](cost-c32-latency/RESULTS.md).

The slower guided natural-stop development run and its extra output tokens
are retained. Equal 80/128 point scores do not establish noninferiority or
that guidance improves complete-task latency.

No complete matched FlashLoop or other external-system E2E result is available
in this evidence set. These internal D4 comparisons do not establish a win
over FlashLoop or a current state-of-the-art system. Any external comparison
requires its own model/checkpoint, precision, hardware, output and arrival
protocol, numerical verification, actual residency and complete raw results.

The original 505 mathematics and 148 code problems, three arms, 1959 planned
completions and 1800-second admission/600-second natural-drain budget remain
frozen. All seven original W5 cases also remain pending real new-device runs.
An external-baseline comparison does not replace or alter these experiments.
