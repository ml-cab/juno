# llama.cpp vs Juno - 20260927T214616Z (gpu)

> **Superseded, not a reference.** Taken on an intermediate build of the per-request device-memory fix whose device scratch had moved from per thread to per instance but whose batched FP16 pack loop was still inline in its large caller: an A/B against the pre-change build showed prefill 7% to 16% slower, traced to that loop running interpreted after deoptimization. The final build extracts the loop; its sweep is [`20260927T232837Z`](../20260927T232837Z/INDEX.md). Generation figures here were within the 0.95x gate.

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3697.277427 | 187.336620 | 155.62497593967962 | 71.20349455981771 | 70.34 / 72.35 | 0.0420917767228398 | 0.3800831602482083 | 128/128 | 64/64 | 5 | 49324769 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned | 3697.277427 | 187.336620 | 145.54452389087038 | 72.06345196854555 | 71.66 / 73.25 | 0.03936532401599259 | 0.3846735996867326 | 128/128 | 64/64 | 6 | 49322079 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1446.551573 | 70.455138 | 73.76648069892914 | 31.696094878717986 | 31.61 / 31.7 | 0.05099471188983951 | 0.4498762727385189 | 128/128 | 64/64 | 7 | 143475210 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |
| qwen2.5-3b-instruct-q4_k_m-tuned | 1446.551573 | 70.455138 | 73.87669857918607 | 31.717893506296342 | 31.68 / 31.73 | 0.05107090542646423 | 0.4501856700116937 | 128/128 | 64/64 | 7 | 143471186 | yes | qwen2.5-3b-instruct-q4_k_m-tuned-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1180.183602 | 59.732827 | 45.83200889569502 | 24.98171033363135 | 20.01 / 33.36 | 0.038834643031834815 | 0.41822414220628384 | 128/128 | 64/64 | 9 | 212295737 | NOISY | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| Phi-3.5-mini-instruct-Q4_K_M-tuned | 1180.183602 | 59.732827 | 44.979118440749886 | 25.143977785701072 | 24.89 / 25.15 | 0.03811196695541774 | 0.42094069623895536 | 128/128 | 64/64 | 10 | 212287764 | yes | Phi-3.5-mini-instruct-Q4_K_M-tuned-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 648.104558 | 36.328041 | 55.3737460267504 | 23.605689096920187 | 23.51 / 23.69 | 0.08543952568028446 | 0.6497925141881498 | 128/128 | 64/64 | 8 | 224427789 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 648.104558 | 36.328041 | 56.651577420175606 | 23.277644560868683 | 22.43 / 23.39 | 0.08741116957269664 | 0.6407624501653884 | 128/128 | 64/64 | 6 | 225972082 | yes | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

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
  GPU clocks {"graphics_mhz": 1809, "sm_mhz": 1809, "memory_mhz": 5005, "active_throttle_reasons": "none"}. A throttled run and a regression look the same
  without this.
- Thread counts are not matched: the reference tool ran with -t 12, while Juno
  dispatches its kernels on the common pool at an effective parallelism of
  11. Juno has no thread-count control reaching the hot path yet, so this
  mismatch is recorded rather than removed.
- Rows marked NOISY are not scorable and should be re-run:
  - Phi-3.5-mini-instruct-Q4_K_M: the 3 generation readings span 53% of their median, over the 15% this host can resolve: re-run rather than score this row
- The Scorable column asks whether a row's own repetitions agree: a generation reading
  whose cycles span more than 15% of their median is not stable at the resolution a
  gate would read it at, and should be re-run rather than scored. Collection pauses and
  lock/park totals are recorded in every result JSON but are not gated on. Neither
  measures lost time reliably here: the park figure sums every thread, so an idle worker
  pool exceeds wall time on a healthy run, and the pause counter has reported ~635 ms on
  rows that produced their tokens in the same span as pause-free repetitions of
  themselves. A pause that does cost time appears in the dispersion anyway.
- GC max ms and Alloc B/tok come from the recording taken alongside each run. A
  result whose GC max is a large fraction of its measurement window should be
  re-run rather than scored: one long pause looks exactly like a regression.
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran with jdk.incubator.vector. On CPUs without HW FMA this can be pathologically slow; re-run with --vector 0.
