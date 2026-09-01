# Multi-session static batch — 20260901T172932Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · sessions=8 · max_tokens=64 · backend=gpu

| parallel | aggregate tg t/s | wall ms | ok/fail |
|---------:|-----------------:|--------:|--------:|
| 1 | 27.8488 | 18385 | 8/8 |
| 8 | - | 2790 | 0/8 |

Speedup parallel=8 over parallel=1: 
