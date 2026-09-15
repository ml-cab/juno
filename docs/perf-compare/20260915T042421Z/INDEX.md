# llama.cpp vs Juno - 20260915T042421Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3598.584002 | 194.391195 | 64.20139766671385 | 23.287115780336954 | 0.017840738921484776 | 0.11979511613340796 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1477.056687 | 69.356628 | 35.6358403741666 | 11.586841899171525 | 0.024126251001608714 | 0.16706178246110126 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1201.237020 | 59.895362 | 32.32912135192838 | 9.160950678727371 | 0.026913190997001055 | 0.15294924970530058 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 668.959015 | 37.293004 | 0.8970693505855691 | 0.5332377258087373 | 0.0013409929913054076 | 0.014298599431913216 | mistral-7b-instruct-v0.1-q4_k_m-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran without jdk.incubator.vector (scalar kernels).
