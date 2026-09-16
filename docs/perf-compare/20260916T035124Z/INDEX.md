# llama.cpp vs Juno - 20260916T035124Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3506.959781 | 202.830170 | 49.27415612082426 | 29.098180078014757 | 0.014050390993299007 | 0.14346080801497507 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1487.845476 | 75.733273 | 31.882391774117192 | 14.553061725187048 | 0.021428563845105372 | 0.19216205966942757 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1209.875551 | 64.848029 | 39.33279760439738 | 13.227424886117772 | 0.03250978794627893 | 0.2039757428266289 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 672.406580 | 39.362454 | 0.918131064176203 | 0.5371003895238122 | 0.0013654403325086483 | 0.013644992497769886 | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 672.406580 | 39.362454 | 20.09929604979239 | 19.519016131402132 | 0.02989158144435825 | 0.4958790458390153 | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran without jdk.incubator.vector (scalar kernels).
