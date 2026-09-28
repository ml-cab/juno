# llama.cpp vs Juno - 20260927T234659Z (gpu)

> **Part of the GPU reference from 2026-09-27, late**: re-run of the rows of [`20260927T232837Z`](../20260927T232837Z/INDEX.md) that spread over 15%. The `qwen2.5-3b` tuned and both `Phi-3.5-mini` rows here are the reference; the `qwen2.5-3b` default row is **not** (it was clean in the first run and is read there; here it spread 23.84 to 31.88 t/s in generation and 31% in prefill, recorded rather than re-run a third time).

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| qwen2.5-3b-instruct-q4_k_m | 1407.005669 | 67.712041 | 90.99505570900367 | 23.92957977539399 | 23.84 / 31.88 | 0.06467284227339078 | 0.3534021338301409 | 128/128 | 64/64 | 8 | 143473386 | NOISY | qwen2.5-3b-instruct-q4_k_m-*.json |
| qwen2.5-3b-instruct-q4_k_m-tuned | 1407.005669 | 67.712041 | 88.93108316766325 | 31.29797655823417 | 31.13 / 31.41 | 0.0632059167401004 | 0.46222172742118595 | 128/128 | 64/64 | 638 | 143473870 | yes | qwen2.5-3b-instruct-q4_k_m-tuned-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1175.595130 | 59.432811 | 46.5507439118185 | 24.70803466121087 | 24.7 / 25.13 | 0.03959759846216657 | 0.4157305408490753 | 128/128 | 64/64 | 9 | 212998643 | yes | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| Phi-3.5-mini-instruct-Q4_K_M-tuned | 1175.595130 | 59.432811 | 46.680991463150235 | 24.791133086512065 | 24.73 / 25.03 | 0.03970839132614495 | 0.4171287319139602 | 128/128 | 64/64 | 8 | 212993318 | yes | Phi-3.5-mini-instruct-Q4_K_M-tuned-*.json |

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
  request; the generation run prefilled 9 tokens against the reference 0.
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
  GPU clocks {"graphics_mhz": 1607, "sm_mhz": 1607, "memory_mhz": 5005, "active_throttle_reasons": "Idle"}. A throttled run and a regression look the same
  without this.
- Thread counts are not matched: the reference tool ran with -t 12, while Juno
  dispatches its kernels on the common pool at an effective parallelism of
  11. Juno has no thread-count control reaching the hot path yet, so this
  mismatch is recorded rather than removed.
- Rows marked NOISY are not scorable and should be re-run:
  - qwen2.5-3b-instruct-q4_k_m: the 3 generation readings span 33% of their median, over the 15% this host can resolve: re-run rather than score this row
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
