# llama.cpp vs Juno - 20260911T221215Z (cpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q2_K | 42.120831 | 9.101056 | 2.1975861411145132 | 2.06373090984445 | 0.052173380461428054 | 0.2267573026519615 | tinyllama-1.1b-chat-v1.0.Q2_K-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 69.705640 | 11.359771 | 5.789806906795793 | 3.092085049513268 | 0.08306080981102523 | 0.2721960723955851 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 23.750684 | 4.598331 | 2.0071177372615283 | 1.0460776649549122 | 0.08450778669201815 | 0.2274907276041921 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 17.977643 | 4.155350 | 0.8951054911694423 | 0.8637586769776407 | 0.04978992469532531 | 0.20786664829139317 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 10.707242 | 2.406415 | 0.844889909092966 | 0.4974690177297303 | 0.07890826686208885 | 0.20672619549401508 | mistral-7b-instruct-v0.1-q4_k_m-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: cpu (llama -ngl 0 / juno --cpu), temperature 0, max_tokens=64.
- Juno ran without jdk.incubator.vector (scalar kernels).
