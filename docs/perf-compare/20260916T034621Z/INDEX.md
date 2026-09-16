# llama.cpp vs Juno - 20260916T034621Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3694.332001 | 203.544088 | 59.27554507622755 | 28.851032661540913 | 0.01604499678431244 | 0.14174340775518332 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1468.292238 | 72.963435 | 33.109383943543 | 13.391330269656981 | 0.022549587259714846 | 0.1835348112332839 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1174.553392 | 63.949360 | 36.9319640360966 | 12.86796605139981 | 0.031443410140095704 | 0.20122118581639925 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 643.653944 | 38.917691 | 0.8963330542562125 | 0.5365321068501244 | 0.0013925698158329196 | 0.013786329380387045 | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 643.653944 | 38.917691 | 21.13771546724977 | 16.297571901590157 | 0.032840186352139825 | 0.41877026829752456 | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran without jdk.incubator.vector (scalar kernels).
