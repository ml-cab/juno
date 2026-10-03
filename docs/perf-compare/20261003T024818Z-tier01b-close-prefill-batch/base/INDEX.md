# Prefill microbatch — 20261003T024859Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · raw prompt target: 512 tokens · backend: GPU · gpu-attention: off (default)

| prefill-batch | pp t/s (JFR) | prefill ms | prefill count | wall ms | attention share of prefill |
|--------------:|-------------:|-----------:|--------------:|--------:|---------------------------:|
| 1 | 60.8212 | 8714.062023 | 529.0 | 9106 | 0.1% |
| 32 | 152.6761 | 3471.402185 | 17.0 | 3906 | 12.3% |
| 512 | 196.7793 | 2693.372235 | 2.0 | 3117 | 14.0% |

Speedup prefill-batch=32 over 1: 2.5102447830690613
