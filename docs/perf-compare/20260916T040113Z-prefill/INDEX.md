# Prefill microbatch — 20260916T040113Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · raw prompt target: 512 tokens · backend: GPU · gpu-attention: on

| prefill-batch | pp t/s (JFR) | prefill ms | prefill count | wall ms | attention share of prefill |
|--------------:|-------------:|-----------:|--------------:|--------:|---------------------------:|
| 1 | 40.7439 | 13008.093387 | 529.0 | 13597 | 0.2% |
| 32 | 119.5622 | 4432.838426 | 17.0 | 5075 | 11.0% |

Speedup prefill-batch=32 over 1: 2.934480989792337
