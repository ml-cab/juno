# Multi-session static batch — 20260901T173121Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · sessions=8 · max_tokens=64 · backend=gpu

| parallel | aggregate tg t/s | wall ms | ok/fail |
|---------:|-----------------:|--------:|--------:|
| 1 | 28.8110 | 17771 | 8/8 |
| 8 | 32.0561 | 15972 | 8/8 |

Speedup parallel=8 over parallel=1: 1.1126340633785707
