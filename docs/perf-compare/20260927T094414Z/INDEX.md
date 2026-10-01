# llama.cpp vs Juno - 20260927T094414Z (cpu)

> **Superseded as reference on 2026-10-01** by [`20261001T180241Z`](../20261001T180241Z/INDEX.md), the first pinned-clock CPU sweep with the thread count recorded as matched. Kept as the unpinned reading; compare ratios, not absolute t/s, across the two.

> **CPU reference sweep from 2026-09-27** (supersedes [`20260925T174146Z`](../20260925T174146Z/INDEX.md) as reference). Taken after the scalar CPU RoPE began reading a per-position table, a measurement boundary; every row within 1% spread.

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 70.793465 | 25.178570 | 6.358101758070104 | 3.3932841808923144 | 3.38 / 3.41 | 0.08981198699724761 | 0.13476874107196374 | 128/128 | 64/64 | 5 | 64170360 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 27.719123 | 12.123212 | 2.1425086850174484 | 1.137271245818261 | 1.13 / 1.14 | 0.07729352350063343 | 0.09380940016707297 | 128/128 | 64/64 | 635 | 180324860 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 19.476154 | 9.795329 | 0.9435042415107611 | 0.9347461654814612 | 0.93 / 0.94 | 0.048444073789453555 | 0.09542774576346146 | 128/128 | 64/64 | 7 | 199023982 | yes | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 11.818178 | 5.900274 | 0.8865822156117108 | 0.5315321992467844 | 0.53 / 0.53 | 0.07501851940389719 | 0.09008601960634106 | 128/128 | 64/64 | 7 | 272658218 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |

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
  GPU clocks {"graphics_mhz": 139, "sm_mhz": 139, "memory_mhz": 405, "active_throttle_reasons": "Idle"}. A throttled run and a regression look the same
  without this.
- Thread counts are not matched: the reference tool ran with -t 12, while Juno
  dispatches its kernels on the common pool at an effective parallelism of
  11. Juno has no thread-count control reaching the hot path yet, so this
  mismatch is recorded rather than removed.
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
- Backend: cpu (llama -ngl 0 / juno --cpu), temperature 0, max_tokens=64.
- Juno ran with jdk.incubator.vector. On CPUs without HW FMA this can be pathologically slow; re-run with --vector 0.
