# llama.cpp vs Juno - 20260910T025804Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3554.265459 | 195.369965 | 26.242844632543203 | 27.32025935100004 | 0.007383479071911157 | 0.13983858445693043 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1485.548618 | 71.809387 | 11.360110740358877 | 10.676538156615269 | 0.007647081086886971 | 0.14867886501545083 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1204.813146 | 60.634090 | 8.322580189835612 | 7.3893809091938865 | 0.006907776709995836 | 0.121868422684234 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 666.821555 | 37.064539 | 5.556515412723574 | 5.344964492303072 | 0.008332837130202808 | 0.14420695998142785 | mistral-7b-instruct-v0.1-q4_k_m-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran without jdk.incubator.vector (scalar kernels).
- Juno fused Q4_K MMQ enabled via `-DJUNO_MMQ=on` (JAVA_TOOL_OPTIONS). JFR `cuda_resident_q4k` counts: tinyllama 12462, qwen2.5-3b 12744, Phi-3.5 6720, mistral 15936.
