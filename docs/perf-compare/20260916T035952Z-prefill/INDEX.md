# Prefill microbatch — 20260916T035952Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · raw prompt target: 512 tokens · backend: GPU · gpu-attention: off (default)

| prefill-batch | pp t/s (JFR) | prefill ms | prefill count | wall ms | attention share of prefill |
|--------------:|-------------:|-----------:|--------------:|--------:|---------------------------:|
| 1 | 20.8539 | 25414.892104 | 529.0 | 26268 | 0.0% |
| 32 | 31.0363 | 17076.764353 | 17.0 | 18088 | 78.7% |

Speedup prefill-batch=32 over 1: 1.4882731767199422
