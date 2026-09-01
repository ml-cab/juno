# Multi-session static batch — 20260901T155136Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · sessions=8 · max_tokens=64 · backend=cpu

| parallel | aggregate tg t/s | wall ms | ok/fail |
|---------:|-----------------:|--------:|--------:|
| 1 | 1.4616 | 350294 | 8/8 |
| 8 | 2.2170 | 230945 | 8/8 |

Speedup parallel=8 over parallel=1: 1.5168308702791462
