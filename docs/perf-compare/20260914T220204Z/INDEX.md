# llama.cpp vs Juno - 20260914T220204Z (cpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q2_K | 45.631471 | 15.894777 | - | 1.4591 | - | 0.09179745019385928 | tinyllama-1.1b-chat-v1.0.Q2_K-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 73.727233 | 25.279336 | - | 2.4927 | - | 0.09860622921424834 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno tg from POST /v1/chat/completions (x_juno_latency_ms / completion_tokens); JFR disabled (--no-jfr).
- Backend: cpu (llama -ngl 0 / juno --cpu), temperature 0, max_tokens=64.
- Juno ran without jdk.incubator.vector (scalar kernels).
