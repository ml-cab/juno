# llama.cpp vs Juno - 20260911T235203Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3384.546806 | 185.081756 | 35.705968646496096 | 39.336263757550164 | 0.0105497044931386 | 0.21253452856558244 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1367.437557 | 68.477167 | 20.08361711559631 | 18.1776553334954 | 0.014687045132545172 | 0.2654557150925272 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1112.295997 | 58.712636 | 25.704351244818888 | 19.33550274109105 | 0.023109272454586466 | 0.3293243849772143 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 612.091958 | 35.856099 | 15.317228533754317 | 15.303371627670328 | 0.025024391079737595 | 0.4267996813504539 | mistral-7b-instruct-v0.1-q4_k_m-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno `--gpu --mmq on --gpu-layers auto --vector 0`), temperature 0, max_tokens=64.
- Juno ran without jdk.incubator.vector (scalar kernels).
- JFR `cuda_resident_q4k` on all four models (cpu.count=0). Phi-3.5 q4k p95 ≈ **0.32 ms** (prior PTX ≈ 2.3 ms). Mistral packed-Q4 tg **0.427×** llama (P0 0.15× fit **met**). Phi-3.5 tg **0.329×** (P0 0.5× **unmet**).
- Paired `--mmq off` Phi-3.5: [`20260911T235353Z`](../20260911T235353Z/) JFR tg **12.83**; this run **19.34** → MMQ/FP16 **1.51×** (tile-kernel 1.3× **met**).
