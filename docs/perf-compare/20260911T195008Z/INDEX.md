# llama.cpp vs Juno - 20260911T195008Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3271.66161 | 168.146487 | 24.509427990195192 | 25.84459125781284 | 0.007491431239490319 | 0.1537028320895746 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1412.996595 | 62.718867 | 13.659094719962813 | 12.27014761920117 | 0.00966675699594507 | 0.19563726524589753 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1174.467356 | 55.726778 | 13.962107269043106 | 12.532832371114456 | 0.011888033496814452 | 0.2248978466171946 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 649.837059 | 34.178493 | 0.7746600945159094 | 0.4471355879997687 | 0.0011920835904741921 | 0.013082366972697382 | mistral-7b-instruct-v0.1-q4_k_m-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass.
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64, --vector 0.
- Qwen2.5 re-measured alone after the first matrix pass fell to CPU MatVec (VRAM contention); ratios use the re-run.
