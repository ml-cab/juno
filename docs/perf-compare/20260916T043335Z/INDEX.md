# llama.cpp vs Juno - 20260916T043335Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3525.095882 | 199.453257 | 63.99777158052749 | 28.52103699210584 | 0.01815490236941234 | 0.142996095531826 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1512.280932 | 76.067445 | 34.039594556840086 | 13.617512135756066 | 0.022508777196458156 | 0.17901892374268735 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1216.978006 | 65.147160 | 35.64919465006221 | 13.288902091293147 | 0.02929321193505794 | 0.203982830430262 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 674.973009 | 39.653287 | 0.9220909993634796 | 0.5327964199022551 | 0.001366115366197524 | 0.013436374641584066 | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 674.973009 | 39.653287 | 20.824010835824016 | 16.319881197384838 | 0.030851620076891125 | 0.4115643981137008 | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran without jdk.incubator.vector (scalar kernels).
