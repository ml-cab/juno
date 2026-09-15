# llama.cpp vs Juno - 20260915T042207Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3563.213049 | 194.514456 | 59.38030135578153 | 39.20151286184425 | 0.016664819234551903 | 0.2015352157777119 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1466.976803 | 70.665575 | 30.99014184993668 | 18.745390088184376 | 0.02112517511289964 | 0.2652690519844263 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1194.233357 | 58.761856 | 35.03573674344278 | 19.9995512913172 | 0.029337429354221917 | 0.34034921040133925 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 672.288316 | 37.377748 | 21.56797002542191 | 16.05986921707735 | 0.03208142922034943 | 0.42966390637224455 | mistral-7b-instruct-v0.1-q4_k_m-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran without jdk.incubator.vector (scalar kernels).
