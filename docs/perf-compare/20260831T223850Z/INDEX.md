# llama.cpp vs Juno - 20260831T223850Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno tg t/s | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3784.262618 | 170.844711 | 18.1457 | 0.10621165790727934 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1403.612858 | 63.471359 | 9.3595 | 0.14746021114814953 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1134.164310 | 55.262626 | 9.1064 | 0.16478406219784056 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 660.776089 | 33.341548 | 0.3832 | 0.011493167623770796 | mistral-7b-instruct-v0.1-q4_k_m-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno metrics from POST /v1/chat/completions (x_juno_latency_ms / completion_tokens).
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran without jdk.incubator.vector (scalar kernels).
