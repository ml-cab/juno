# Prefill microbatch — 20260901T234024Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · raw prompt target: 256 tokens · backend: CPU

| prefill-batch | pp t/s (JFR) | prefill ms | prefill count | wall ms |
|--------------:|-------------:|-----------:|--------------:|--------:|
| 1 | 2.30 | 118848 | 246 | 118026 |
| 32 | 5.39 | 50628 | 9 | 54100 |

Speedup prefill-batch=32 over 1: **2.35×** (JFR `ForwardPass.prefill.total_ms` / `prefill.count`).

Prompt tokens (API usage): 273 (chat template overhead over 256 raw words).
