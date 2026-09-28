# llama.cpp vs Juno - 20260927T093054Z (gpu)

> **Part of the GPU reference from 2026-09-27**: re-run of the three models whose rows in [`20260927T091155Z`](../20260927T091155Z/INDEX.md) spread over 15%; these rows are the reference for those models. Three rows still spread over 15% here: the `qwen2.5-3b` default generation lane (23%; median 31.08 t/s against 30.83 in the first run), and the `tinyllama` default and `qwen2.5-3b` tuned prefill lanes, each with one repetition four to five times faster than the other two (medians 168.16 against 170.50 t/s, and 79.60 against 86.77 t/s, in the first run). The one-fast-prefill-repetition pattern recurred in both runs, on the same `tinyllama` lane, so it is recorded as a harness question for the owner rather than re-run a third time; the median of three is robust to a single such reading.

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3588.622439 | 190.862269 | 168.15871640479017 | 71.54156310340733 | 69.97 / 72.74 | 0.04685884883773089 | 0.37483345177777033 | 128/128 | 64/64 | 635 | 47476395 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned | 3588.622439 | 190.862269 | 188.7409071376954 | 72.04707165707298 | 71.1 / 73.98 | 0.05259425039717737 | 0.37748200330298376 | 128/128 | 64/64 | 6 | 47179047 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1446.924421 | 69.908374 | 80.35787474944641 | 31.078751994389677 | 24.18 / 31.44 | 0.05553702293165347 | 0.4445640803259089 | 128/128 | 64/64 | 636 | 141986315 | NOISY | qwen2.5-3b-instruct-q4_k_m-*.json |
| qwen2.5-3b-instruct-q4_k_m-tuned | 1446.924421 | 69.908374 | 79.59621333476842 | 31.494650940610583 | 31.16 / 31.58 | 0.05501062265557575 | 0.4505132810070877 | 128/128 | 64/64 | 636 | 141988741 | yes | qwen2.5-3b-instruct-q4_k_m-tuned-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1168.385823 | 59.128552 | 45.514777538844164 | 24.566012071087332 | 24.43 / 24.69 | 0.038955263443695654 | 0.4154678448930617 | 128/128 | 64/64 | 9 | 212090286 | yes | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| Phi-3.5-mini-instruct-Q4_K_M-tuned | 1168.385823 | 59.128552 | 45.00960731298827 | 24.490694933043056 | 24.29 / 24.64 | 0.03852289751121729 | 0.4141940586172828 | 128/128 | 64/64 | 9 | 212099596 | yes | Phi-3.5-mini-instruct-Q4_K_M-tuned-*.json |

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
  GPU clocks {"graphics_mhz": 1822, "sm_mhz": 1822, "memory_mhz": 5005, "active_throttle_reasons": "none"}. A throttled run and a regression look the same
  without this.
- Thread counts are not matched: the reference tool ran with -t 12, while Juno
  dispatches its kernels on the common pool at an effective parallelism of
  11. Juno has no thread-count control reaching the hot path yet, so this
  mismatch is recorded rather than removed.
- Rows marked NOISY are not scorable and should be re-run:
  - qwen2.5-3b-instruct-q4_k_m: the 3 generation readings span 23% of their median, over the 15% this host can resolve: re-run rather than score this row
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
