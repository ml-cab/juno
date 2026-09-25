# llama.cpp vs Juno - 20260925T174146Z (cpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 73.570879 | 26.793220 | 6.229390376450727 | 3.417368788583141 | 3.13 / 3.42 | 0.08467195799646116 | 0.12754602800944198 | 128/128 | 64/64 | 4 | 64223482 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 28.771879 | 14.528403 | 2.1631856705755115 | 1.147018180505739 | 1.12 / 1.15 | 0.07518402501885649 | 0.07895005256295126 | 128/128 | 64/64 | 5 | 180264823 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 19.194463 | 7.623840 | 0.9478234763345764 | 0.9394645200929218 | 0.94 / 0.94 | 0.049380046544390245 | 0.123227208348145 | 128/128 | 64/64 | 7 | 199077342 | yes | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 11.983404 | 6.045361 | 0.8881317437487736 | 0.5132197777723091 | 0.51 / 0.54 | 0.0741134775852315 | 0.08489481071061085 | 128/128 | 64/64 | 8 | 272674769 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Prompt tokens column is Juno actual / requested. A prefill ratio is published
  only when the two are within 10%; otherwise it reads `withheld`, because the two
  engines then prefilled measurably different amounts of work.
- Prefill and generation are measured in two separate Juno runs per model, mirroring the
  reference tool, which benchmarks prompt processing and generation separately. The prefill
  run is given the requested prompt length and asked for one token; the generation run is
  given the shortest prompt the chat template allows and asked for the full count, so it
  decodes at a shallow context like the reference does. Measuring both in one request would
  time Juno generation at the prompt length while the reference times it near zero, and
  decode slows as context grows.
  Residual: Juno cannot reach an empty context, because the chat template wraps every
  request; the generation run prefilled 19 tokens against the reference 0.
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
  GPU clocks {"graphics_mhz": 1607, "sm_mhz": 1607, "memory_mhz": 4513, "active_throttle_reasons": "none"}. A throttled run and a regression look the same
  without this.
- Thread counts are not matched: the reference tool ran with -t 12, while Juno
  dispatches its kernels on the common pool at an effective parallelism of
  11. Juno has no thread-count control reaching the hot path yet, so this
  mismatch is recorded rather than removed.
- The Scorable column applies the collection-pause rule: a single pause above 200 ms,
  or above 5% of the measurement window when that is smaller, means re-run rather than
  score. Lock and park totals are recorded in each result JSON but are not gated on:
  the park figure sums every thread, so an idle worker pool exceeds wall time on a
  perfectly healthy run and the figure cannot discriminate.
- GC max ms and Alloc B/tok come from the recording taken alongside each run. A
  result whose GC max is a large fraction of its measurement window should be
  re-run rather than scored: one long pause looks exactly like a regression.
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: cpu (llama -ngl 0 / juno --cpu), temperature 0, max_tokens=64.
- Juno ran with jdk.incubator.vector. On CPUs without HW FMA this can be pathologically slow; re-run with --vector 0.
