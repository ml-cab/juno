# llama.cpp vs Juno - 20260912T193402Z (cpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 81.301944 | 46.004749 | 6.003474854470786 | 3.3327564035175223 | 0.07384171348314605 | 0.07244374713396486 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 29.197576 | 15.724738 | 2.1715665067768377 | 1.1304940904016818 | 0.07437489012022222 | 0.07189271391368694 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 19.317319 | 13.773947 | 0.9450626934602765 | 0.9058613628743513 | 0.04892307744466385 | 0.06576628782398765 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 11.989937 | 5.340794 | 0.9024216839547986 | 0.4827049917614293 | 0.07526492290616695 | 0.0903807545772088 | mistral-7b-instruct-v0.1-q4_k_m-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: cpu (llama -ngl 0 / juno --cpu), temperature 0, max_tokens=64.
- Juno ran without jdk.incubator.vector (scalar kernels).
