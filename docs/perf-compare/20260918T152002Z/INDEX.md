# llama.cpp vs Juno - 20260918T152002Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| mistral-7b-instruct-v0.1-q4_k_m | 674.673978 | 36.860587 | 21.39409459299014 | 18.756143245022077 | 0.03171027087247485 | 0.5088400584890869 | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 674.673978 | 36.860587 | 21.173698956023998 | 18.731875348391295 | 0.03138360103763183 | 0.5081816887070001 | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran with jdk.incubator.vector. On CPUs without HW FMA this can be pathologically slow; re-run with --vector 0.
