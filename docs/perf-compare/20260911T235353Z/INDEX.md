# llama.cpp vs Juno - 20260911T235353Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| Phi-3.5-mini-instruct-Q4_K_M | 1111.539623 | 58.662920 | 14.908814696901656 | 12.828591654375295 | 0.0134127604526264 | 0.21868314182750015 | Phi-3.5-mini-instruct-Q4_K_M-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno `--gpu --mmq off --vector 0`), temperature 0, max_tokens=64.
- Juno ran without jdk.incubator.vector (scalar kernels).
- Paired FP16-resident control for [`20260911T235203Z`](../20260911T235203Z/) (`--mmq on`). Phi-3.5 JFR tg **12.83** vs MMQ **19.34** (**1.51×**).
