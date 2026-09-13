# llama.cpp vs Juno - 20260913T031043Z (cpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 50.986786 | 3.241321 | 5.446608383488324 | 2.918132438568112 | 0.10682392068188655 | 0.9002910969225547 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 19.560367 | 0.979319 | 1.764672371743623 | 0.9751766484844588 | 0.0902167311964864 | 0.9957701713991649 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 15.308978 | 1.173556 | 0.8675538967822228 | 0.8110575232203306 | 0.05666961548852071 | 0.6911110532606289 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 9.326266 | 1.422740 | 0.804667130544044 | 0.46849593664375083 | 0.08627966761231601 | 0.3292913228304194 | mistral-7b-instruct-v0.1-q4_k_m-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: cpu (llama -ngl 0 / juno --cpu), temperature 0, max_tokens=64.
- Juno ran without jdk.incubator.vector (scalar kernels).
