# llama.cpp vs Juno - 20260918T031702Z (cpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 61.171032 | 25.980170 | 6.0661457040785125 | 3.2714450386363874 | 0.09916696687540784 | 0.12592084804050116 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 26.604239 | 9.681474 | 2.123528802429983 | 1.0960732668725683 | 0.07981918980768378 | 0.1132134700638114 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 18.993640 | 8.006464 | 0.9514197692326593 | 0.9191373157156246 | 0.05009149216435919 | 0.11479940654396556 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 11.399561 | 4.794358 | 0.904221005577829 | 0.5213794193865913 | 0.07932068661046061 | 0.10874853721532503 | mistral-7b-instruct-v0.1-q4_k_m-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: cpu (llama -ngl 0 / juno --cpu), temperature 0, max_tokens=64.
- Juno ran without jdk.incubator.vector (scalar kernels).
