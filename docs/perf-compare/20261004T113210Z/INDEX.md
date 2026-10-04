# llama.cpp vs Juno - 20261004T113210Z (gpu)

> **GPU reference from 2026-10-04 at `n_prompt=128`**: the closing sweep of the packed K-quant prefill work, owner run with pinned clocks, jar `5ab4c78c505a4cef` (HEAD `59c53cc` plus the attention barrier fix and the decode GEMV scratch held at upload, uncommitted at the time of the run). Same method as [`20261003T042440Z`](../20261003T042440Z/INDEX.md), which it supersedes as the GPU reference. Paired with [`20261004T114812Z`](../20261004T114812Z/INDEX.md).

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3079.474760 | 147.599376 | 1613.2120859933793 | 53.5987332227496 | 52.77 / 54.66 | 0.5238594928420128 | 0.36313658414619315 | 128/128 | 64/64 | 8 | 48505386 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned | 3079.474760 | 147.599376 | 1627.7116897866367 | 54.637306096832035 | 54 / 54.93 | 0.5285679593576655 | 0.37017301547963205 | 128/128 | 64/64 | 10 | 48486445 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1210.410991 | 56.009940 | 745.8923011700532 | 24.193256104760295 | 23.78 / 24.91 | 0.6162306082116973 | 0.43194576006973573 | 128/128 | 64/64 | 0 | 143503799 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |
| qwen2.5-3b-instruct-q4_k_m-tuned | 1210.410991 | 56.009940 | 745.1885070182028 | 24.861030935257478 | 23.83 / 24.87 | 0.6156491576489682 | 0.44386819438223785 | 128/128 | 64/64 | 0 | 143549730 | yes | qwen2.5-3b-instruct-q4_k_m-tuned-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 978.458250 | 48.210622 | 296.87791931161985 | 26.2729589869771 | 26.03 / 26.37 | 0.3034139875785399 | 0.544962041497351 | 128/128 | 64/64 | 9 | 210261574 | yes | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| Phi-3.5-mini-instruct-Q4_K_M-tuned | 978.458250 | 48.210622 | 304.4318013864476 | 26.35459173354546 | 25.75 / 26.77 | 0.3111341760227865 | 0.5466552937969864 | 128/128 | 64/64 | 12 | 210828055 | yes | Phi-3.5-mini-instruct-Q4_K_M-tuned-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 522.919527 | 29.418003 | 381.37177753920395 | 18.452579826295693 | 18.32 / 18.48 | 0.7293125573778849 | 0.6272546721235869 | 128/128 | 64/64 | 15 | 224223279 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 522.919527 | 29.418003 | 381.02148078551085 | 18.201028384836505 | 18.01 / 18.42 | 0.7286426708358719 | 0.6187037367844618 | 128/128 | 64/64 | 14 | 224224432 | yes | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

GPU attention per row, as the engine resolved it (read from its log, not from the flag):
- tinyllama-1.1b-chat-v1.0.Q4_K_M: unknown (requested: engine default)
- tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned: unknown (requested: auto)
- qwen2.5-3b-instruct-q4_k_m: unknown (requested: engine default)
- qwen2.5-3b-instruct-q4_k_m-tuned: unknown (requested: auto)
- Phi-3.5-mini-instruct-Q4_K_M: unknown (requested: engine default)
- Phi-3.5-mini-instruct-Q4_K_M-tuned: unknown (requested: auto)
- mistral-7b-instruct-v0.1-q4_k_m: unknown (requested: engine default)
- mistral-7b-instruct-v0.1-q4_k_m-tuned: unknown (requested: auto)

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
- Build: Juno `59c53cc2465a` plus uncommitted changes, jar sha256 `5ab4c78c505a4cef`,
  OpenJDK Runtime Environment (build 25.0.3+9-2-24.04.2-Ubuntu); JVM -XX:+UseG1GC -XX:+AlwaysPreTouch -XX:+UnlockExperimentalVMOptions -XX:-UseFastUnorderedTimeStamps -Djava.util.concurrent.ForkJoinPool.common.parallelism=11, -Xms equal to -Xmx per model.
  Reference tool build `ac4cdde` from /home/medion/Repo/llama.cpp/build-cuda/bin; GPU driver 580.173.02.
  A different reference build is a measurement boundary: its ratios are not comparable with this run.
- Clocks pinned: cpu governor performance (was schedutil), turbo off, gpu graphics clock locked at 1911 MHz.
- Clock state this run: CPU governor performance, turbo disabled,
  GPU clocks {"graphics_mhz": 1809, "sm_mhz": 1809, "memory_mhz": 5005, "active_throttle_reasons": "none"}. A throttled run and a regression look the same
  without this.
- Thread counts are matched: the reference tool ran with -t 12, and Juno's CPU kernels
  ran on 12 threads (common pool parallelism 11 plus the calling thread).
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
