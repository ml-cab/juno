# llama.cpp vs Juno - 20261006T202000Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3850.558964 | 193.351223 | 2419.9125681808005 | 129.2511195585559 | 129.25 / 129.25 | 0.6284574761236666 | 0.6684784174266946 | 128/128 | 64/64 | 0 | 38805888 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 663.663004 | 37.203656 | 522.990926270864 | 36.18809114109173 | 36.19 / 36.19 | 0.7880368848628241 | 0.9727025521656186 | 128/128 | 64/64 | 0 | 193564097 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1199.620204 | 59.764780 | 359.4053045094996 | 48.22401679746187 | 48.22 / 48.22 | 0.2995992425861974 | 0.8068969181759201 | 128/128 | 64/64 | 9 | 180692914 | yes | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| Qwen3-1.7B-Q4_K_M | 2577.206928 | 109.983714 | 749.8673511018425 | 63.175347650626506 | 63.18 / 63.18 | 0.2909612507070843 | 0.5744063857547719 | 128/128 | 64/64 | 19 | 97592066 | yes | Qwen3-1.7B-Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1453.296874 | 70.873871 | 1070.827160356359 | 28.95281631725862 | 28.95 / 28.95 | 0.7368261636791762 | 0.40851185223477665 | 128/128 | 64/64 | 0 | 143169831 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |

GPU attention per row, as the engine resolved it (read from its log, not from the flag):
- tinyllama-1.1b-chat-v1.0.Q4_K_M: unknown (requested: engine default)
- mistral-7b-instruct-v0.1-q4_k_m: unknown (requested: engine default)
- Phi-3.5-mini-instruct-Q4_K_M: unknown (requested: engine default)
- Qwen3-1.7B-Q4_K_M: unknown (requested: engine default)
- qwen2.5-3b-instruct-q4_k_m: unknown (requested: engine default)

GPU clock while the measured requests ran (sampled every 100 ms; SM clock median over busy
samples, median across reps; lowest SM clock and hottest sample of any rep; clock event
reasons seen). The clock under "Clock state" below is read once before the run and says
nothing about a long window, which can heat the card into thermal slowdown:
- tinyllama-1.1b-chat-v1.0.Q4_K_M: prefill 1809 MHz (low 1809, 73 C, no event reasons); generate 1822 MHz (low 1822, 77 C, no event reasons)
- mistral-7b-instruct-v0.1-q4_k_m: prefill 1822 MHz (low 1822, 81 C, no event reasons); generate 1809 MHz (low 1721, 83 C, sw_thermal_slowdown)
- Phi-3.5-mini-instruct-Q4_K_M: prefill 1822 MHz (low 1822, 78 C, no event reasons); generate 1822 MHz (low 1809, 81 C, no event reasons)
- Qwen3-1.7B-Q4_K_M: prefill 1822 MHz (low 1822, 76 C, no event reasons); generate 1822 MHz (low 1822, 79 C, no event reasons)
- qwen2.5-3b-instruct-q4_k_m: prefill 1822 MHz (low 1822, 77 C, no event reasons); generate 1822 MHz (low 1822, 77 C, no event reasons)

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
- Juno readings are the median of 1 measured cycle(s), each preceded by
  2 discarded request(s) so the measured request runs on compiled code. The
  min/max column is that median's own spread; a difference smaller than the spread
  is not a result. Each cycle records only its measured request: the recording is
  started after the warmup requests return and stopped before the engine exits.
  The last warmup ran under a discarded recording with the same settings, so the
  measured request does not pay for the first recording's recompilation.
- Build: Juno `5e4c9132d3c1` plus uncommitted changes, jar sha256 `9659ff62d527578b`,
  OpenJDK Runtime Environment (build 25.0.3+9-2-24.04.2-Ubuntu); JVM -XX:+UseG1GC -XX:+AlwaysPreTouch -XX:+UnlockExperimentalVMOptions -XX:-UseFastUnorderedTimeStamps -Djava.util.concurrent.ForkJoinPool.common.parallelism=11, -Xms equal to -Xmx per model.
  Reference tool build `ac4cdde` from <reference-tool-bin>; GPU driver 580.173.02.
  A different reference build is a measurement boundary: its ratios are not comparable with this run.
- Clocks pinned: cpu governor performance (was schedutil), turbo off, gpu graphics clock locked at 1911 MHz.
- Clock state this run: CPU governor performance, turbo disabled,
  GPU clocks {"graphics_mhz": 1822, "sm_mhz": 1822, "memory_mhz": 5005, "active_throttle_reasons": "none"}. A throttled run and a regression look the same
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
