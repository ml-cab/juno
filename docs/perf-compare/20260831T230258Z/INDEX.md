# llama.cpp vs Juno - 20260831T230258Z (cpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 32.592528 | 0.608315 | 5.233203711972242 | 3.124386169942561 | 0.16056452300883936 | 5.136132053200333 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| Qwen3.5-0.8B.Q4_K_M | 68.105452 | 8.007915 | - | - |  |  | Qwen3.5-0.8B.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 23.610719 | 3.798940 | 1.7840368630534575 | 1.006948112369853 | 0.07556046315461455 | 0.2650602832289673 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 17.495180 | 3.538043 | 0.8571021535714866 | 0.8404457170986165 | 0.048990759373238026 | 0.23754536536119444 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 10.713910 | 2.179294 | 0.8097569727877868 | 0.48073106076273014 | 0.07557996779773088 | 0.22059027408084 | mistral-7b-instruct-v0.1-q4_k_m-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: cpu (llama -ngl 0 / juno --cpu), temperature 0, max_tokens=64.
- Juno ran without jdk.incubator.vector (scalar kernels).
