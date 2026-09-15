# llama.cpp vs Juno - 20260915T041705Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3527.209375 | 192.460809 | 64.96536663821833 | 28.664104252801874 | 0.018418347121289994 | 0.14893475924650132 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1472.751313 | 68.269949 | 34.53817989421681 | 13.215000628858517 | 0.023451467732092807 | 0.1935698037339755 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1207.845203 | 61.130617 | 35.480621743965514 | 12.737484928375935 | 0.029375139840635285 | 0.20836506407870764 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 671.640577 | 37.381621 | 0.9298593133031633 | 0.5398515117374566 | 0.0013844597023255241 | 0.014441629263146629 | mistral-7b-instruct-v0.1-q4_k_m-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran without jdk.incubator.vector (scalar kernels).
