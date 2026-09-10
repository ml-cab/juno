# llama.cpp vs Juno - 20260910T222026Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3255.830459 | 174.845754 | 28.910688953204932 | 28.671240673614438 | 0.008879666591142034 | 0.16398019407216738 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1351.273953 | 64.169834 | 14.303620835385024 | 12.537466367002517 | 0.010585285688093941 | 0.19537944210674626 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1096.610221 | 53.907540 | 14.559316840358495 | 12.149292020528417 | 0.013276656155075629 | 0.22537277754704477 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 601.562019 | 32.747343 | 0.8394365692194611 | 0.48657549503083336 | 0.0013954281399196202 | 0.014858472488312513 | mistral-7b-instruct-v0.1-q4_k_m-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran without jdk.incubator.vector (scalar kernels).
