# llama.cpp vs Juno - 20260915T034519Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| Devstral-Small-2-24B-Instruct-2512-UD-IQ1_S | 190.439322 | 16.060482 | - | - |  |  | Devstral-Small-2-24B-Instruct-2512-UD-IQ1_S-*.json |
| gemma-4-E4B-it-qat-UD-Q4_K_XL | 1159.243208 | 56.200855 | - | - |  |  | gemma-4-E4B-it-qat-UD-Q4_K_XL-*.json |
| minimax-m2.5-tiny-24e-iq4_nl-imat | 2531.156616 | 131.874436 | - | - |  |  | minimax-m2.5-tiny-24e-iq4_nl-imat-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 670.262609 | 37.306422 | 16.970813441457516 | 16.48245666211913 | 0.025319648170106287 | 0.4418128509380806 | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1201.249922 | 60.829209 | 27.707923939136325 | 20.324965339947756 | 0.0230659111244765 | 0.3341316724987786 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1470.544223 | 69.982692 | 21.86668470114199 | 21.06770371241933 | 0.014869790625223507 | 0.30104163058516425 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Qwen3.5-0.8B.Q4_K_M | 3823.196098 | 138.885304 | - | - |  |  | Qwen3.5-0.8B.Q4_K_M-*.json |
| tinyllama-1.1b-chat-v1.0.Q2_K | 3436.159186 | 136.609392 | 64.67369766483391 | 28.242131197042987 | 0.018821508016373348 | 0.20673638015344498 | tinyllama-1.1b-chat-v1.0.Q2_K-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3827.208141 | 194.888969 | 40.54919275194209 | 45.33565973837977 | 0.010594979749741834 | 0.23262301592030982 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran with jdk.incubator.vector. On CPUs without HW FMA this can be pathologically slow; re-run with --vector 0.
