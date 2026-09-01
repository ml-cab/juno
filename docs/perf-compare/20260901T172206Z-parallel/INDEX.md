# Multi-session static batch — 20260901T172206Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · sessions=8 · max_tokens=64 · backend=gpu

| parallel | aggregate tg t/s | wall ms | ok/fail |
|---------:|-----------------:|--------:|--------:|
| 1 | 21.3511 | 23980 | 8/8 |
| 8 | 19.9727 | 25635 | 8/8 |

Speedup parallel=8 over parallel=1: 0.9354412653212247
