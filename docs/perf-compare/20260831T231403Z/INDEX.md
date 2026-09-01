# llama.cpp vs Juno - 20260831T231403Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3583.402274 | 185.732095 | 19.52876830232714 | 31.366294334507387 | 0.005449783978768313 | 0.16887923616275038 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1356.304009 | 68.001800 | 8.100798307872198 | 13.254269522561193 | 0.0059727009977983475 | 0.1949105688755473 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1095.720199 | 57.820043 | 11.706915125417794 | 12.618713275911556 | 0.010684219507956514 | 0.21824116035181013 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 609.917436 | 35.161896 | 0.8163662953813189 | 0.4752929618890457 | 0.001338486567518491 | 0.01351727341122463 | mistral-7b-instruct-v0.1-q4_k_m-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran without jdk.incubator.vector (scalar kernels).
