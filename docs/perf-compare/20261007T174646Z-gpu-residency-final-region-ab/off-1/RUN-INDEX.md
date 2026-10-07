# llama.cpp vs Juno - 20261007T174646Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3545.373833 | 194.136252 | 2386.3285003034644 | 64.28926407913663 | 64.29 / 64.29 | 0.6730823356599936 | 0.33115537884772095 | 128/128 | 64/64 | 7 | 47799410 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 629.247176 | 34.276336 | 475.3174446545009 | 22.137577549516646 | 22.14 / 22.14 | 0.7553747760554128 | 0.6458560083410504 | 128/128 | 64/64 | 14 | 223681346 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1177.783026 | 57.158315 | 643.1612339389953 | 28.837344444684167 | 28.84 / 28.84 | 0.5460778596235223 | 0.5045170496135893 | 128/128 | 64/64 | 10 | 209146574 | yes | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| Qwen3-1.7B-Q4_K_M | 2513.782006 | 109.062781 | 1440.5899846245156 | 31.446484067571387 | 31.45 / 31.45 | 0.5730767350494416 | 0.28833378150857336 | 128/128 | 64/64 | 0 | 111104862 | yes | Qwen3-1.7B-Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1360.679736 | 67.274482 | 1037.098739610661 | 27.19446793680576 | 27.19 / 27.19 | 0.7621916547823572 | 0.40423154706424586 | 128/128 | 64/64 | 0 | 142942413 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |

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
- tinyllama-1.1b-chat-v1.0.Q4_K_M: prefill 1873 MHz (low 1873, 52 C, no event reasons); generate 1873 MHz (low 1746, 55 C, no event reasons)
- mistral-7b-instruct-v0.1-q4_k_m: prefill 1860 MHz (low 1860, 63 C, no event reasons); generate 1860 MHz (low 1860, 64 C, no event reasons)
- Phi-3.5-mini-instruct-Q4_K_M: prefill 1860 MHz (low 1860, 64 C, no event reasons); generate 1860 MHz (low 1860, 65 C, no event reasons)
- Qwen3-1.7B-Q4_K_M: prefill 1860 MHz (low 1860, 64 C, no event reasons); generate 1860 MHz (low 1860, 66 C, no event reasons)
- qwen2.5-3b-instruct-q4_k_m: prefill 1847 MHz (low 1847, 66 C, no event reasons); generate 1847 MHz (low 1847, 68 C, no event reasons)

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
- Build: Juno `d05d1fe8d324` plus uncommitted changes, jar sha256 `533b1864d682bea5`,
  OpenJDK Runtime Environment (build 25.0.3+9-2-24.04.2-Ubuntu); JVM -XX:+UseG1GC -XX:+AlwaysPreTouch -XX:+UnlockExperimentalVMOptions -XX:-UseFastUnorderedTimeStamps -Djava.util.concurrent.ForkJoinPool.common.parallelism=11, -Xms equal to -Xmx per model.
  Reference tool build `ac4cdde` from <reference-tool-bin>; GPU driver 580.173.02.
  A different reference build is a measurement boundary: its ratios are not comparable with this run.
- Clocks pinned: cpu governor performance (was schedutil), turbo off, gpu graphics clock locked at 1911 MHz.
- Clock state this run: CPU governor performance, turbo disabled,
  GPU clocks {"graphics_mhz": 1847, "sm_mhz": 1847, "memory_mhz": 5005, "active_throttle_reasons": "none"}. A throttled run and a regression look the same
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
