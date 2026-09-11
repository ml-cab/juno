# llama.cpp vs Juno - 20260911T204900Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3528.767408 | 164.731219 | 25.508821225875174 | 26.192839359503648 | 0.007228819096448414 | 0.15900349380346446 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1412.249728 | 61.254503 | 12.981186870188147 | 11.765983105773447 | 0.009191849439101571 | 0.19208356168971688 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1166.408993 | 54.231388 | 12.24304122880863 | 11.250944658539618 | 0.010496353596622716 | 0.20746186062100452 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 648.357264 | 32.931325 | 0.7246405260036071 | 0.42484537182055554 | 0.0011176562155453958 | 0.012900949834862566 | mistral-7b-instruct-v0.1-q4_k_m-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran without jdk.incubator.vector (scalar kernels).
