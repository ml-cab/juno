# llama.cpp vs Juno - 20260913T032734Z (cpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 53.450789 | 5.044845 | 5.589584271781835 | 2.939035022003164 | 0.10457440154497691 | 0.5825818279854315 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 21.893461 | 2.270970 | 1.909892648923015 | 0.9971428343490623 | 0.08723575723925125 | 0.4390823455831923 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 17.210732 | 2.858020 | 0.871170596136538 | 0.8283252121658615 | 0.050617870067149844 | 0.28982484802970643 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 10.247523 | 1.972754 | 0.8185661561483443 | 0.47300451095173335 | 0.07987941633781591 | 0.23976862343289299 | mistral-7b-instruct-v0.1-q4_k_m-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: cpu (llama -ngl 0 / juno --cpu), temperature 0, max_tokens=64.
- Juno ran without jdk.incubator.vector (scalar kernels).
