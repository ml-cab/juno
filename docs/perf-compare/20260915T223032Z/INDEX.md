# llama.cpp vs Juno - 20260915T223032Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3577.643819 | 206.018501 | 66.50224371144199 | 28.218632600319754 | 0.018588279626458253 | 0.13697135190940815 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1487.833001 | 74.410803 | 32.45921036861955 | 13.425804679080516 | 0.021816433932304983 | 0.1804281655054914 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1185.933785 | 63.554789 | 31.18282685460832 | 11.819042214739563 | 0.026293902112425542 | 0.18596619390459407 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 613.424858 | 35.241571 | 0.7465480161818927 | 0.4787311288422976 | 0.0012170162432215826 | 0.013584273210813945 | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 613.424858 | 35.241571 | 20.697750540839724 | 15.649659373829952 | 0.033741297358444715 | 0.44406815388082305 | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran with jdk.incubator.vector. On CPUs without HW FMA this can be pathologically slow; re-run with --vector 0.
