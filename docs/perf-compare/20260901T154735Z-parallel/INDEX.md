# Multi-session static batch — 20260901T154735Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · sessions=8 · max_tokens=64 · backend=gpu

| parallel | aggregate tg t/s | wall ms | ok/fail |
|---------:|-----------------:|--------:|--------:|
| 1 | 28.8029 | 17776 | 8/8 |
| 8 | 24.9161 | 20549 | 8/8 |

Speedup parallel=8 over parallel=1: 0.86505525485281
