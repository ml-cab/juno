# llama.cpp vs Juno - 20260918T063656Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q2_K | 3058.093299 | 135.298679 | 67.04196070954048 | 31.123385115140888 | 0.021922797689482944 | 0.23003465625219363 | tinyllama-1.1b-chat-v1.0.Q2_K-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3561.508775 | 190.738474 | 58.66166955800585 | 46.41265151096437 | 0.016471016432636994 | 0.24333135595372526 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran with jdk.incubator.vector. On CPUs without HW FMA this can be pathologically slow; re-run with --vector 0.
