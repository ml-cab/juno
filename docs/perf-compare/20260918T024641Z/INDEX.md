# llama.cpp vs Juno - 20260918T024641Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3640.799772 | 179.120660 | 66.22384386363714 | 27.173505740871406 | 0.018189367175020014 | 0.151705033583906 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned | 3640.799772 | 179.120660 | 61.223748238886124 | 44.1636735524732 | 0.016816016280196067 | 0.24655823372062832 | tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1340.287774 | 69.213592 | 33.322225202950804 | 13.731940356034558 | 0.024861992961036146 | 0.1983994755832721 | qwen2.5-3b-instruct-q4_k_m-*.json |
| qwen2.5-3b-instruct-q4_k_m-tuned | 1340.287774 | 69.213592 | 29.729447018758556 | 20.26254718073165 | 0.022181390888937974 | 0.29275387384506285 | qwen2.5-3b-instruct-q4_k_m-tuned-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1213.149221 | 61.183345 | 38.14211730020787 | 12.449850925698996 | 0.031440581784957414 | 0.2034843130217708 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| Phi-3.5-mini-instruct-Q4_K_M-tuned | 1213.149221 | 61.183345 | 34.59469666364249 | 19.950312282160223 | 0.02851643974607349 | 0.3260742328187552 | Phi-3.5-mini-instruct-Q4_K_M-tuned-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 672.164646 | 37.357369 | 0.9155348134836544 | 0.5292661333276292 | 0.0013620692771212106 | 0.01416765011817693 | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 672.164646 | 37.357369 | 20.64839279843155 | 18.864725628101137 | 0.03071924850750266 | 0.5049800382918063 | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran with jdk.incubator.vector. On CPUs without HW FMA this can be pathologically slow; re-run with --vector 0.
