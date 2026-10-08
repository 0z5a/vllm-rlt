# H100 48 GiB confirmation input preparation

The complete original 653 confirmation inputs pass canonical tokenization and static capacity checks under a separately declared common 48 GiB FP32/R4 pool. All original fields and ID order remain unchanged. The byte-identical generator uses the pinned Transformers 4.54.1/tokenizers 0.21.4 loader, without truncation, chat templates or added special tokens. CUDA remains uninitialized. This result prepares the three-arm/1,959-output experiment; actual final model outputs remain zero.

| Task | Original IDs | Planned three-arm outputs | Actual B16 cohorts | Maximum prompt tokens | Maximum R4 groups / available | Maximum target KV GiB | Static result |
|---|---:|---:|---:|---:|---:|---:|---|
| gsm8k | 505 | 1515 | 32 | 420 | 1363 / 2048 | 31.94531250 | PASS |
| humanevalplus | 148 | 444 | 10 | 395 | 1199 / 2048 | 28.10156250 | PASS |
| Total | 653 | 1959 | 42 | — | <=2048 | <=48 | PASS |

Each R4 block group covers 16 positions and occupies 24 MiB. Worst spans include the original 1,024-token output maximum. These are static KV reservations, excluding weights, activation, scratch, context and guidance reference memory; they are not observed GPU peaks or memory savings. Both the original 41 global cohorts and the 42 actual per-task cohorts were independently recomputed from all archived token IDs.

All 48 archived payload files, the complete archive and 18 control files pass independent hash review. Both CPU children, preparation parent, resource wrapper, scheduler step, archival and transfer returned zero naturally; the completed resource handoff was verified. Independent review checks recorded inputs, order and capacity arithmetic. It does not claim independent retokenization or another device weight hash; those checks rely on the original byte-pinned executed driver and complete recorded model-file checks. Existing tokenizer/W2/public qualification was reused without replay. Operational records remain private.

Source `e8aa1ee57734bc5c1118c143ebdb68d0b53210a1`, Ouro-1.4B revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, the selected candidate, scoring/reference cache, B16, max 1,024 tokens, M2, 1pp noninferiority margin and original 1,800-second admission/600-second natural-drain policy remain frozen. Every arm uses the same declared 48 GiB pool. The original 64/102 GiB profiles retain their capacity stop on this H100; this preparation does not complete or replace them.

The complete C32 six-fresh/60-trial case, seven original W5 cases/42 jobs/420 trials, Graph/HTTP high-concurrency E2E and final quality remain unrun on this new profile. Any speed ratio requires a complete newly measured baseline on the same device and pool. No new speedup, VRAM saving or external-system/SOTA result is reported.

[Machine-readable scientific summary](H100_48GiB_CPU_PREPARATION-v1.json).
