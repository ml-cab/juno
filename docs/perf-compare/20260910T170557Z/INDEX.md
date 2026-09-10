# llama.cpp vs Juno - 20260910T170557Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3436.941576 | 173.354175 | 16.653945522622635 | 33.41978532240072 | 0.004845571318091744 | 0.19278327344813426 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1343.934930 | 63.253677 | 14.073608182391292 | 12.39781942665463 | 0.01047194166044281 | 0.1960015609314638 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1078.503108 | 53.219196 | 9.903402737876235 | 12.81579570401584 | 0.0091825444585332 | 0.24081152417289134 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 597.630831 | 32.373144 | 0.8066278118367852 | 0.4711459196507434 | 0.0013497091682620793 | 0.01455360405065209 | mistral-7b-instruct-v0.1-q4_k_m-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran without jdk.incubator.vector (scalar kernels).
