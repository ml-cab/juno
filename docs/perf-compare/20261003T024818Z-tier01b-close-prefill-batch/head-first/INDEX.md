# Prefill microbatch — 20261003T015257Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · raw prompt target: 512 tokens · backend: GPU · gpu-attention: engine default (auto)

| prefill-batch | pp t/s (JFR) | prefill ms | prefill count | wall ms | attention share of prefill |
|--------------:|-------------:|-----------:|--------------:|--------:|---------------------------:|
| 1 | 59.1581 | 8959.047472 | 528.0 | 9386 | 0.1% |
| 32 | 236.2830 | 2243.072801 | 17.0 | 2103 | 0.0% |

Speedup prefill-batch=32 over 1: 3.9940937927350606
