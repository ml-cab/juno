# Prefill microbatch — 20260916T003101Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · raw prompt target: 512 tokens · backend: GPU

| prefill-batch | pp t/s (JFR) | prefill ms | prefill count | wall ms |
|--------------:|-------------:|-----------:|--------------:|--------:|
| null | - | - | - | null |
| 1 | 21.0687 | 25155.7467 | 529.0 | 26166 |
| null | - | - | - | null |
| null | - | - | - | null |
| 32 | 31.7203 | 16708.533631 | 17.0 | 17733 |
| null | - | - | - | null |

Speedup prefill-batch=32 over 1: 1.5055651274164994
