# llama.cpp vs Juno - 20260904T194612Z (cpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno/llama pp | Juno/llama tg | Results |
|-------|------------------|------------------|-------------|-------------|---------------|---------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 51.639674 | 4.200437 | 4.799880570523616 | 2.888973842424852 | 0.09294947467181175 | 0.6877793530589441 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 20.650738 | 2.032461 | 1.8707089465391145 | 0.9571908252816935 | 0.09058799479898076 | 0.4709516321748331 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 16.418609 | 2.536594 | 0.8337089058296002 | 0.8176346671842517 | 0.0507782910129354 | 0.3223356466128406 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 8.179455 | 1.162556 | 0.8125844753247287 | 0.4526401810366134 | 0.09934457434202261 | 0.38934914192229314 | mistral-7b-instruct-v0.1-q4_k_m-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: cpu (llama -ngl 0 / juno --cpu), temperature 0, max_tokens=64.
- Juno `--vector 0`: no `jdk.incubator.vector` module (scalar kernels).
- Vector SIMD track bake-off pair with `20260904T195731Z` (`--vector 1`).
