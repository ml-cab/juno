# llama.cpp vs Juno - 20260918T030920Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3367.909574 | 181.107778 | 62.069649139515 | 27.787796930883943 | 0.018429725554001766 | 0.15343237732663223 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1508.909385 | 71.976921 | 32.73970170022919 | 12.961313978543945 | 0.021697592993782853 | 0.18007597155404778 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1231.539440 | 60.584900 | 33.58425220980182 | 12.566059739234527 | 0.027270139403575915 | 0.20741240373813488 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 668.842110 | 37.274698 | 0.8752636059902932 | 0.5158640529254107 | 0.001308625149200449 | 0.013839523339006279 | mistral-7b-instruct-v0.1-q4_k_m-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran without jdk.incubator.vector (scalar kernels).
