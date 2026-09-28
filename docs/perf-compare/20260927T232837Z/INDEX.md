# llama.cpp vs Juno - 20260927T232837Z (gpu)

> **GPU reference from 2026-09-27, late** (supersedes [`20260927T091155Z`](../20260927T091155Z/INDEX.md) + [`20260927T093054Z`](../20260927T093054Z/INDEX.md) as reference). Taken on the final build of the per-request device-memory fix: `CudaMatVec` scratch owned by the instance, and the batched FP16 pack loop moved into its own method, which a same-hour A/B showed is what the old code was paying for in prefill (tinyllama 181 to 229 t/s, qwen2.5-3b 83 to 90, mistral-7b 61 to 65). **A measurement boundary for prefill**: pp rose 10% to 34% on the three models that take the batched K-quant path, 2% to 4% on Phi-3.5-mini; generation within the 0.95x gate (0.962x to 1.021x). The `qwen2.5-3b` tuned and both `Phi-3.5-mini` rows spread over 15% and were re-run in [`20260927T234659Z`](../20260927T234659Z/INDEX.md); read those three rows there.

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3599.245120 | 186.169193 | 224.89357921627712 | 68.80957724036533 | 68.65 / 70.68 | 0.06248354077542713 | 0.36960775374025134 | 128/128 | 64/64 | 5 | 48206404 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned | 3599.245120 | 186.169193 | 219.54290046396397 | 70.23101820757911 | 69.11 / 71 | 0.06099692939611848 | 0.3772429641867713 | 128/128 | 64/64 | 5 | 48212662 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1443.944325 | 70.232876 | 93.70758281423372 | 31.232290254416107 | 30.55 / 31.47 | 0.06489695010521526 | 0.4446961598784037 | 128/128 | 64/64 | 6 | 143469800 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |
| qwen2.5-3b-instruct-q4_k_m-tuned | 1443.944325 | 70.232876 | 92.88440935022155 | 30.98030937280627 | 25.73 / 45.25 | 0.06432686339912833 | 0.4411083688614185 | 128/128 | 64/64 | 7 | 143471277 | NOISY | qwen2.5-3b-instruct-q4_k_m-tuned-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1180.328335 | 59.880509 | 46.18773057217306 | 24.916677947627722 | 19.98 / 25.1 | 0.039131256280628954 | 0.4161066491206299 | 128/128 | 64/64 | 8 | 212996594 | NOISY | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| Phi-3.5-mini-instruct-Q4_K_M-tuned | 1180.328335 | 59.880509 | 46.70212100506563 | 25.001532681849408 | 24.91 / 25.23 | 0.03956705911416219 | 0.41752371680490236 | 128/128 | 64/64 | 9 | 213000855 | yes | Phi-3.5-mini-instruct-Q4_K_M-tuned-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 643.663465 | 36.421047 | 65.12984357614347 | 23.289963581938828 | 23.26 / 23.6 | 0.10118617432503098 | 0.639464416877934 | 128/128 | 64/64 | 8 | 224376012 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 643.663465 | 36.421047 | 65.85213461680213 | 23.435052945070304 | 23.42 / 23.62 | 0.10230833066904323 | 0.6434480849787295 | 128/128 | 64/64 | 8 | 224395334 | yes | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

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
  GPU clocks {"graphics_mhz": 1126, "sm_mhz": 1126, "memory_mhz": 5005, "active_throttle_reasons": "Idle"}. A throttled run and a regression look the same
  without this.
- Thread counts are not matched: the reference tool ran with -t 12, while Juno
  dispatches its kernels on the common pool at an effective parallelism of
  11. Juno has no thread-count control reaching the hot path yet, so this
  mismatch is recorded rather than removed.
- Rows marked NOISY are not scorable and should be re-run:
  - qwen2.5-3b-instruct-q4_k_m-tuned: the 3 generation readings span 63% of their median, over the 15% this host can resolve: re-run rather than score this row
  - Phi-3.5-mini-instruct-Q4_K_M: the 3 generation readings span 20% of their median, over the 15% this host can resolve: re-run rather than score this row
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
