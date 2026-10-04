# LAST_EXITED alias integration map

Reference: upstream `ecb1f8b505b7e831815b40aec3b4598619cca23a`.

| Entry | Current behavior | Alias status |
|---|---|---|
| `KVCacheManager.finalize_token` | Copies current token across skipped depth planes | Override publishes source-depth metadata only |
| `KVCacheManager._prepare_batches` | Builds depth-selected page tables | Retains write addresses and adds all-depth tables |
| `KVCacheManager._write_prepared` | Tracks physical writes | Physical tracking unchanged; separate logical readability |
| `KVCacheManager._attend_prepared` | Reads one page table per row | Triton resolves each historical token before loading K/V |
| `KVCacheManager.read` | Materializes dense diagnostic view | Alias-aware diagnostic export; materialization bytes counted |
| `ModelRunner.finalize_many` | Resident async batched payload kernel | Async rejected before cache allocation in this revision |
| `Workspace`, `AsyncState` | Construct descriptors directly | Static buffers and async rejected |
| `RecurrentGraphs` | Writes/reads direct page tables | CUDA Graph execution rejected pending stable alias descriptors |
| `PrefillMetadataBank` | FA4 packed metadata | FA4/UVA rejected; ordinary full-depth prefill retained |
| `PreemptionManager` | Copies physical pages and runtime state | Preemption rejected pending metadata snapshot/restore |
| P/D engine and transport | Exports physical block tables | P/D rejected before worker creation |
| `SpeculativeRunner` | Direct temporary pages and rollback | Speculation rejected |
| Prefix cache / incremental allocation | Shared prompt pages and growth | Rejected for this rectangular prototype |

The CUDA reader does not construct dense K/V. For prompt and in-progress
positions, source metadata is -1 and the query's own depth is used. Publication
occurs in the same stream after model writes. The current implementation only
supports synchronous eager execution; it does not claim cross-stream lifetime
validation or capacity savings.
