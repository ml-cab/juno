# llama.cpp vs Juno - 20260904T195731Z (cpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 55.704403 | 10.854346 | 5.4595585115015925 | 2.977274378790931 | 0.0980094609666958 | 0.2742932995494092 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 21.963042 | 2.404461 | 1.880545088138357 | 0.9650654489573732 | 0.0856231613151929 | 0.4013645673426906 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 13.675294 | 1.444766 | 0.7665878439702718 | 0.8123907934195408 | 0.05605640682900652 | 0.5622992189873937 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 9.983364 | 1.576380 | 0.7993269603687 | 0.4630179259473105 | 0.08006589365755871 | 0.29372227885872093 | mistral-7b-instruct-v0.1-q4_k_m-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: cpu (llama -ngl 0 / juno --cpu), temperature 0, max_tokens=64.
- Juno `--vector 1`: `jdk.incubator.vector` enabled (Q8_0 dequant Vector when probe passes; Q4/Q5 weight-stationary accumulate remains scalar).
- Vector SIMD track bake-off pair with `20260904T194612Z` (`--vector 0`).
