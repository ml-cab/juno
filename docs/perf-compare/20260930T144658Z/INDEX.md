# llama.cpp vs Juno - 20260930T144658Z (gpu)

> **Attribution run, not a ratio reference** (Tier 01B step 2): `n_prompt=512` with `--device-spans`, for host-device staging and dequantization totals. Throughput is within each row spread of the unspanned [`20260930T141026Z`](../20260930T141026Z/INDEX.md) (0.98x to 1.01x), but read ratios there.

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3947.080331 | 183.047484 | 237.39368081742487 | 68.25992819217672 | 68.2 / 69.04 | 0.06014412196097381 | 0.3729083115514263 | 512/512 | 64/64 | 7 | 48299692 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned | 3947.080331 | 183.047484 | 242.32439421802712 | 67.6522093951566 | 67.46 / 67.91 | 0.061393327192971976 | 0.369588305268138 | 512/512 | 64/64 | 7 | 49854515 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1523.837455 | 69.802158 | 92.8924940845638 | 30.075842430432136 | 29.44 / 30.45 | 0.06095958186338436 | 0.43087267345562774 | 512/512 | 64/64 | 11 | 145120475 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |
| qwen2.5-3b-instruct-q4_k_m-tuned | 1523.837455 | 69.802158 | 93.79250083752214 | 29.97696656541283 | 29.61 / 30 | 0.06155020046906652 | 0.42945615757915145 | 512/512 | 64/64 | 11 | 145122261 | yes | qwen2.5-3b-instruct-q4_k_m-tuned-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1170.590452 | 59.186027 | 12.97041733211458 | 23.000210507989163 | 22.67 / 23.07 | 0.011080235030068894 | 0.388608792882637 | 512/512 | 64/64 | 10 | 210884120 | yes | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| Phi-3.5-mini-instruct-Q4_K_M-tuned | 1170.590452 | 59.186027 | 13.10614340947583 | 23.582739369892398 | 23.57 / 23.87 | 0.011196181710762691 | 0.398451130532759 | 512/512 | 64/64 | 8 | 212693258 | yes | Phi-3.5-mini-instruct-Q4_K_M-tuned-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 696.372629 | 37.012201 | 68.06904830612052 | 22.31027453131299 | 22.16 / 22.43 | 0.0977480237898904 | 0.602781621425675 | 512/512 | 64/64 | 29 | 225946671 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 696.372629 | 37.012201 | 67.0239274050249 | 21.96429691730283 | 21.76 / 22.16 | 0.09624721681159716 | 0.5934339575563968 | 512/512 | 64/64 | 26 | 225867917 | yes | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

GPU attention per row, as the engine resolved it (corrected 2026-09-30, see below):
- tinyllama-1.1b-chat-v1.0.Q4_K_M: on (requested: engine default)
- tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned: on (requested: auto)
- qwen2.5-3b-instruct-q4_k_m: on (requested: engine default)
- qwen2.5-3b-instruct-q4_k_m-tuned: on (requested: auto)
- Phi-3.5-mini-instruct-Q4_K_M: off (requested: engine default)
- Phi-3.5-mini-instruct-Q4_K_M-tuned: off (requested: auto)
- mistral-7b-instruct-v0.1-q4_k_m: on (requested: engine default)
- mistral-7b-instruct-v0.1-q4_k_m-tuned: on (requested: auto)

Correction (2026-09-30): this list first read `off` on every row. That was wrong. The harness
read it from the engine log, but the console turns library logging off unless `--verbose`, so the
activation line never reaches the log and a silent log proves nothing. The values are now
re-derived from this run's own recordings: `on`/`off` where the run carried `--device-spans`
(the attention kernel's `memcpy_gqa_*` copy sites present or absent), `unknown` otherwise.

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
  The last warmup ran under a discarded recording with the same settings, so the
  measured request does not pay for the first recording's recompilation.
- Build: Juno `ffd0ca7af4fa` plus uncommitted changes, jar sha256 `dc94bd797c8a3219` (juno-player-0.1.2-shaded.jar; filled in after the run, the harness recorded `missing`),
  OpenJDK Runtime Environment (build 25.0.3+9-2-24.04.2-Ubuntu); JVM -XX:+UseG1GC -XX:+AlwaysPreTouch -XX:+UnlockExperimentalVMOptions -XX:-UseFastUnorderedTimeStamps, -Xms equal to -Xmx per model.
  Reference tool build `ac4cdde` from /home/medion/Repo/llama.cpp/build-cuda/bin; GPU driver 580.173.02.
  A different reference build is a measurement boundary: its ratios are not comparable with this run.
- Clocks pinned: cpu governor performance (was schedutil), turbo off, gpu graphics clock locked at 1911 MHz.
- Clock state this run: CPU governor performance, turbo disabled,
  GPU clocks {"graphics_mhz": 1809, "sm_mhz": 1809, "memory_mhz": 5005, "active_throttle_reasons": "none"}. A throttled run and a regression look the same
  without this.
- Thread counts are not matched: the reference tool ran with -t 12, while Juno
  dispatches its kernels on the common pool at an effective parallelism of
  11. Juno has no thread-count control reaching the hot path yet, so this
  mismatch is recorded rather than removed.
- JFR timestamps: os-clock (kernel clocksource hpet, PERF_JFR_OS_CLOCK=auto). Every repetition's spans are checked against
  the engine's own request latency and its prefill against its prompt; a failing
  repetition is withheld and listed above.
- The Scorable column asks whether a row's own repetitions agree: a generation reading
  whose cycles span more than 15% of their median is not stable at the resolution a
  gate would read it at, and should be re-run rather than scored. Collection pauses and
  lock/park totals are recorded in every result JSON but are not gated on. Neither
  measures lost time reliably here: the park figure sums every thread, so an idle worker
  pool exceeds wall time on a healthy run, and the ~635 ms pauses that runs before
  2026-09-28 reported on rows that lost no time were JFR timestamps misread across
  CPU cores whose counters disagree, not pauses. A pause that does cost time appears in
  the dispersion anyway.
- GC max ms and Alloc B/tok come from the recording taken alongside each run. A
  result whose GC max is a large fraction of its measurement window should be
  re-run rather than scored: one long pause looks exactly like a regression.
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran with jdk.incubator.vector. On CPUs without HW FMA this can be pathologically slow; re-run with --vector 0.
