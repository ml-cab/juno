# llama.cpp vs Juno - 20260831T214609Z

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno tg t/s | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 57.072691 | 6.832592 | 2.2648 | 0.33147010680573347 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| Qwen3.5-0.8B.Q4_K_M | 57.575503 | 6.492781 | - | - | Qwen3.5-0.8B.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 20.621456 | 2.372160 | 0.7968 | 0.33589639821934436 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 15.983839 | 1.991430 | 0.6087 | 0.3056597520374806 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 9.805143 | 1.660195 | 0.3827 | 0.23051509009483823 | mistral-7b-instruct-v0.1-q4_k_m-*.json |

Host meta: see `host.json`.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno metrics from POST /v1/chat/completions (x_juno_latency_ms / completion_tokens).
- Both runs use CPU (-ngl 0 / --cpu), temperature 0, max_tokens=64, n_prompt=128, reps=1.
- Juno ran without jdk.incubator.vector (scalar kernels, `--vector 0`).
- Qwen3.5-0.8B: Juno failed to load (unsupported arch / missing blk.0.ffn_norm.weight).
