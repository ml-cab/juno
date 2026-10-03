# Prefill microbatch — 20261003T024818Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · raw prompt target: 512 tokens · backend: GPU · gpu-attention: engine default (auto)

| prefill-batch | pp t/s (JFR) | prefill ms | prefill count | wall ms | attention share of prefill |
|--------------:|-------------:|-----------:|--------------:|--------:|---------------------------:|
| 1 | 60.4161 | 8772.501668 | 528.0 | 9202 | 0.1% |
| 32 | 338.3541 | 1566.406535 | 17.0 | 2037 | 0.0% |
| 512 | 787.9721 | 672.612623 | 2.0 | 1174 | 0.0% |

Speedup prefill-batch=32 over 1: 5.600396251992433
