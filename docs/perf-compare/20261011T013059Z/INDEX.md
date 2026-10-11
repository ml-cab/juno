# llama.cpp vs Juno - 20261011T013059Z (cpu)

> **CPU reference from 2026-10-11** (supersedes [`20261001T180241Z`](../20261001T180241Z/INDEX.md) as reference). The closing sweep of the CPU work, pinned clocks, jar `6ae135b0e21839cf`: CPU matmuls run integer dot products over an 8-bit activation on a pool of `--threads 12` threads (the reference tool runs `-t 12`). A measurement boundary for CPU readings; compare ratios across it. Every row scorable, every prefill 128 of 128 tokens. Gate results: [`20261011T052951Z-cpu-hot-path-close-gate`](../20261011T052951Z-cpu-hot-path-close-gate/INDEX.md).

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 73.791644 | 32.500000 | 16.52428591444793 | 7.738769730105125 | 7.73 / 7.79 | 0.22393166785182247 | 0.23811599169554232 | 128/128 | 64/64 | 0 | 82268324 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 26.894079 | 12.045991 | 6.177736611331604 | 2.5919723552527496 | 2.58 / 2.6 | 0.2297061970901329 | 0.21517302771127336 | 128/128 | 64/64 | 15 | 165955731 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 18.933048 | 10.286024 | 4.85146207461898 | 2.2852459715631883 | 2.27 / 2.31 | 0.25624305577311063 | 0.22217000189414185 | 128/128 | 64/64 | 8 | 172239438 | yes | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 11.571235 | 5.757517 | 2.620057221818832 | 1.2900299447386592 | 1.29 / 1.29 | 0.22642848596704085 | 0.2240601190997194 | 128/128 | 64/64 | 0 | 208990124 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |

GPU attention per row, as the engine resolved it (read from its log, not from the flag):
- tinyllama-1.1b-chat-v1.0.Q4_K_M: n/a (requested: engine default)
- qwen2.5-3b-instruct-q4_k_m: n/a (requested: engine default)
- Phi-3.5-mini-instruct-Q4_K_M: n/a (requested: engine default)
- mistral-7b-instruct-v0.1-q4_k_m: n/a (requested: engine default)

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
- Build: Juno `aabc42cb1334` plus uncommitted changes, jar sha256 `6ae135b0e21839cf`,
  OpenJDK Runtime Environment (build 25.0.3+9-2-24.04.2-Ubuntu); JVM -XX:+UseG1GC -XX:+AlwaysPreTouch -XX:+UnlockExperimentalVMOptions -XX:-UseFastUnorderedTimeStamps -Djuno.threads=12 -Djava.util.concurrent.ForkJoinPool.common.parallelism=11, -Xms equal to -Xmx per model.
  Reference tool build `379ac6673` from <home>/Repo/llama.cpp-bin/llama-b9551; GPU driver 580.173.02.
  A different reference build is a measurement boundary: its ratios are not comparable with this run.
- Clocks pinned: cpu governor performance (was schedutil), turbo off.
- Clock state this run: CPU governor performance, turbo disabled,
  GPU clocks {"graphics_mhz": 139, "sm_mhz": 139, "memory_mhz": 405, "active_throttle_reasons": "Idle"}. A throttled run and a regression look the same
  without this.
- Thread counts are matched: the reference tool ran with -t 12, and Juno's CPU matmuls ran
  on 12 threads (-Djuno.threads=12, the property --threads sets). A build from
  before the worker pool runs them on common pool parallelism 11 plus the calling thread.
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
- Backend: cpu (llama -ngl 0 / juno --cpu), temperature 0, max_tokens=64.
- Juno ran with jdk.incubator.vector. On CPUs without HW FMA this can be pathologically slow; re-run with --vector 0.
