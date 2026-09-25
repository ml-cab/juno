# llama.cpp vs Juno - 20260925T055156Z (cpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 76.505346 | 36.408754 | 6.328101377453247 | 3.278444277132456 | 3.22 / 3.28 | 0.08271449916000963 | 0.09004549502387409 | 128/128 | 64/64 | 631 | 152627855 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 28.614773 | 13.121851 | 2.1558147592945236 | 1.1197259367139258 | 1.11 / 1.12 | 0.07533922283061703 | 0.08533292572167797 | 128/128 | 64/64 | 4 | 412624245 | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 19.687895 | 10.963165 | 0.9491867287979242 | 0.9071184365265359 | 0.9 / 0.91 | 0.04821169194563076 | 0.08274238657600573 | 128/128 | 64/64 | 11 | 241041601 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 12.010921 | 6.218830 | 0.8872563746153709 | 0.5247326526303271 | 0.52 / 0.53 | 0.07387080263165255 | 0.08437803455478396 | 128/128 | 64/64 | 5 | 580740265 | mistral-7b-instruct-v0.1-q4_k_m-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Prompt tokens column is Juno actual / requested. A prefill ratio is published
  only when the two are within 10%; otherwise it reads `withheld`, because the two
  engines then prefilled measurably different amounts of work.
- Generated-token column is Juno actual / requested. The reference tool generates the
  requested count whatever the model would rather do, while Juno stops at a stop token.
  Where Juno generated nothing there is no reading and the ratio reads `-`; where it
  generated fewer tokens, the ratio is published and the shortfall noted, since it is then
  an average over a shorter and slightly cheaper span of context.
- Juno readings are the median of 3 measured cycle(s), each preceded by
  2 discarded request(s) so the measured request runs on compiled code. The
  min/max column is that median's own spread; a difference smaller than the spread
  is not a result. Each cycle records only its measured request: the recording is
  started after the warmup requests return and stopped before the engine exits.
- Clock state this run: CPU governor schedutil, turbo enabled,
  GPU clocks {"graphics_mhz": 139, "sm_mhz": 139, "memory_mhz": 405, "active_throttle_reasons": "Idle"}. A throttled run and a regression look the same
  without this.
- Thread counts are not matched: the reference tool ran with -t 12, while Juno
  dispatches its kernels on the common pool at an effective parallelism of
  11. Juno has no thread-count control reaching the hot path yet, so this
  mismatch is recorded rather than removed.
- GC max ms and Alloc B/tok come from the recording taken alongside each run. A
  result whose GC max is a large fraction of its measurement window should be
  re-run rather than scored: one long pause looks exactly like a regression.
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: cpu (llama -ngl 0 / juno --cpu), temperature 0, max_tokens=64.
- Juno ran with jdk.incubator.vector. On CPUs without HW FMA this can be pathologically slow; re-run with --vector 0.
