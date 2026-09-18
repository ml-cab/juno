# llama.cpp vs Juno - 20260918T032455Z (cpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 71.067089 | 22.446026 | 5.798846234144042 | 3.299862941760149 | 0.08159678855206863 | 0.14701323707636038 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 26.788949 | 9.882594 | 2.148515909584562 | 1.0991653535203678 | 0.08020157526839004 | 0.11122235250384341 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 19.155678 | 8.080427 | 0.953189843374552 | 0.9186025617007919 | 0.04976017259083975 | 0.11368242813168065 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 11.647341 | 4.888192 | 0.900433513647613 | 0.520457059897228 | 0.07730807517764036 | 0.10647230303090141 | mistral-7b-instruct-v0.1-q4_k_m-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: cpu (llama -ngl 0 / juno --cpu), temperature 0, max_tokens=64.
- Juno ran with jdk.incubator.vector. On CPUs without HW FMA this can be pathologically slow; re-run with --vector 0.
