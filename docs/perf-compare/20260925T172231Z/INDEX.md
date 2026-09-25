# llama.cpp vs Juno - 20260925T172231Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3512.110790 | 174.673297 | 138.2703736704673 | 56.81736688442882 | 56.42 / 56.82 | 0.03936959336936729 | 0.32527792089725555 | 128/128 | 64/64 | 634 | 48541096 | NOISY | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned | 3512.110790 | 174.673297 | 140.7466636014389 | 57.63029157622801 | 57.12 / 57.79 | 0.040074665070983964 | 0.3299318932316713 | 128/128 | 64/64 | 633 | 48742892 | NOISY | tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1334.923815 | 64.506204 | 66.08139916764934 | 27.82642492660014 | 25.62 / 29.44 | 0.049502000357712794 | 0.43137594837544835 | 128/128 | 64/64 | 6 | 141832193 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |
| qwen2.5-3b-instruct-q4_k_m-tuned | 1334.923815 | 64.506204 | 66.4071861897366 | 25.79320962869156 | 21 / 29.12 | 0.049746049507504365 | 0.3998562623324039 | 128/128 | 64/64 | 7 | 141974061 | yes | qwen2.5-3b-instruct-q4_k_m-tuned-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1144.462292 | 57.701862 | 43.16586287788809 | 24.41956622704166 | 24.32 / 24.55 | 0.037717156065014405 | 0.42320239556639716 | 128/128 | 64/64 | 9 | 212088564 | yes | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| Phi-3.5-mini-instruct-Q4_K_M-tuned | 1144.462292 | 57.701862 | 41.45811073641008 | 23.55617943949575 | 23.22 / 23.64 | 0.036224968726545057 | 0.40823950255705355 | 128/128 | 64/64 | 8 | 210519913 | yes | Phi-3.5-mini-instruct-Q4_K_M-tuned-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 618.606824 | 34.355557 | 43.866810442742484 | 19.977074141159044 | 19.88 / 20.27 | 0.07091226404373206 | 0.5814801413686597 | 128/128 | 64/64 | 10 | 224362932 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 618.606824 | 34.355557 | 44.99886125811389 | 21.670752071836485 | 21.24 / 21.93 | 0.07274226457306894 | 0.6307786560362414 | 128/128 | 64/64 | 636 | 225251799 | NOISY | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

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
  - tinyllama-1.1b-chat-v1.0.Q4_K_M: a single collection pause of 634 ms in a 1500 ms window exceeds the 75 ms ceiling: re-run rather than score this row
  - tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned: a single collection pause of 633 ms in a 1441 ms window exceeds the 72 ms ceiling: re-run rather than score this row
  - mistral-7b-instruct-v0.1-q4_k_m-tuned: a single collection pause of 636 ms in a 3514 ms window exceeds the 175 ms ceiling: re-run rather than score this row
- The Scorable column applies the collection-pause rule: a single pause above 200 ms,
  or above 5% of the measurement window when that is smaller, means re-run rather than
  score. Lock and park totals are recorded in each result JSON but are not gated on:
  the park figure sums every thread, so an idle worker pool exceeds wall time on a
  perfectly healthy run and the figure cannot discriminate.
- GC max ms and Alloc B/tok come from the recording taken alongside each run. A
  result whose GC max is a large fraction of its measurement window should be
  re-run rather than scored: one long pause looks exactly like a regression.
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran with jdk.incubator.vector. On CPUs without HW FMA this can be pathologically slow; re-run with --vector 0.
