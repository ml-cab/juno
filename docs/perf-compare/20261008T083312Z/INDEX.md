> **Superseded (2026-10-08).** Closing sweep of the first attention and long-context gate, taken before the decode-width
> attention fix. Prefill is unchanged by that fix; generation is not. The reference column is the re-run:
> `20261008T173153Z`, `20261008T174605Z`, `20261008T180014Z`, `20261008T181417Z`.

# llama.cpp vs Juno - 20261008T083312Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3396.679129 | 181.707225 | 2240.8697992147395 | 124.48424160695976 | 124.43 / 126.68 | 0.6597237225273213 | 0.6850814083312304 | 128/128 | 64/64 | 0 | 38848130 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned | 3396.679129 | 181.707225 | 2435.1822962247065 | 130.37259133989306 | 129.78 / 131.78 | 0.7169303321687723 | 0.7174871078455635 | 128/128 | 64/64 | 0 | 38966077 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1415.811124 | 67.419842 | 1065.907829117987 | 29.28410253152588 | 29.07 / 29.7 | 0.7528601880924245 | 0.4343543630898138 | 128/128 | 64/64 | 0 | 143556842 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |
| qwen2.5-3b-instruct-q4_k_m-tuned | 1415.811124 | 67.419842 | 1075.492923403604 | 29.22867430207447 | 29.2 / 29.34 | 0.7596302255099416 | 0.43353222782804013 | 128/128 | 64/64 | 0 | 143607338 | yes | qwen2.5-3b-instruct-q4_k_m-tuned-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1173.654009 | 58.039313 | 689.1797460733378 | 48.45324635989744 | 48.03 / 48.47 | 0.5872086158173194 | 0.8348349395503258 | 128/128 | 64/64 | 9 | 175042274 | yes | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| Phi-3.5-mini-instruct-Q4_K_M-tuned | 1173.654009 | 58.039313 | 689.4482153641025 | 48.46284745353294 | 48.38 / 48.48 | 0.5874373623548048 | 0.8350003635214107 | 128/128 | 64/64 | 9 | 175075249 | yes | Phi-3.5-mini-instruct-Q4_K_M-tuned-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 628.127111 | 35.996892 | 519.5230238012858 | 35.489321873575484 | 35.39 / 35.67 | 0.827098551715412 | 0.985899612488086 | 128/128 | 64/64 | 0 | 193563468 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 628.127111 | 35.996892 | 484.4212454941775 | 35.48425688049776 | 34.89 / 35.5 | 0.7712153113769635 | 0.9857589060882744 | 128/128 | 64/64 | 0 | 193564488 | yes | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

GPU attention per row, as the engine resolved it (read from its log, not from the flag):
- tinyllama-1.1b-chat-v1.0.Q4_K_M: unknown (requested: engine default)
- tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned: unknown (requested: auto)
- qwen2.5-3b-instruct-q4_k_m: unknown (requested: engine default)
- qwen2.5-3b-instruct-q4_k_m-tuned: unknown (requested: auto)
- Phi-3.5-mini-instruct-Q4_K_M: unknown (requested: engine default)
- Phi-3.5-mini-instruct-Q4_K_M-tuned: unknown (requested: auto)
- mistral-7b-instruct-v0.1-q4_k_m: unknown (requested: engine default)
- mistral-7b-instruct-v0.1-q4_k_m-tuned: unknown (requested: auto)

GPU clock while the measured requests ran (sampled every 100 ms; SM clock median over busy
samples, median across reps; lowest SM clock and hottest sample of any rep; clock event
reasons seen). The clock under "Clock state" below is read once before the run and says
nothing about a long window, which can heat the card into thermal slowdown:
- tinyllama-1.1b-chat-v1.0.Q4_K_M: prefill 1607 MHz (low 1607, 83 C, sw_thermal_slowdown); generate 1607 MHz (low 1607, 83 C, sw_thermal_slowdown)
- tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned: prefill 1822 MHz (low 1809, 80 C, no event reasons); generate 1809 MHz (low 1784, 83 C, sw_thermal_slowdown)
- qwen2.5-3b-instruct-q4_k_m: prefill 1809 MHz (low 1809, 81 C, no event reasons); generate 1822 MHz (low 1822, 80 C, no event reasons)
- qwen2.5-3b-instruct-q4_k_m-tuned: prefill 1822 MHz (low 1822, 80 C, no event reasons); generate 1822 MHz (low 1822, 79 C, no event reasons)
- Phi-3.5-mini-instruct-Q4_K_M: prefill 1809 MHz (low 1809, 80 C, no event reasons); generate 1809 MHz (low 1809, 82 C, no event reasons)
- Phi-3.5-mini-instruct-Q4_K_M-tuned: prefill 1809 MHz (low 1809, 81 C, no event reasons); generate 1809 MHz (low 1809, 83 C, no event reasons)
- mistral-7b-instruct-v0.1-q4_k_m: prefill 1809 MHz (low 1784, 83 C, no event reasons); generate 1632 MHz (low 1607, 84 C, sw_thermal_slowdown)
- mistral-7b-instruct-v0.1-q4_k_m-tuned: prefill 1607 MHz (low 1607, 83 C, sw_thermal_slowdown); generate 1645 MHz (low 1607, 84 C, sw_thermal_slowdown)

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
- Build: Juno `51fa5d46511a` plus uncommitted changes, jar sha256 `fefd30371caf535a`,
  OpenJDK Runtime Environment (build 25.0.3+9-2-24.04.2-Ubuntu); JVM -XX:+UseG1GC -XX:+AlwaysPreTouch -XX:+UnlockExperimentalVMOptions -XX:-UseFastUnorderedTimeStamps -Djava.util.concurrent.ForkJoinPool.common.parallelism=11, -Xms equal to -Xmx per model.
  Reference tool build `ac4cdde` from <home> GPU driver 580.173.02.
  A different reference build is a measurement boundary: its ratios are not comparable with this run.
- Clocks pinned: cpu governor performance (was schedutil), turbo off, gpu graphics clock locked at 1911 MHz.
- Clock state this run: CPU governor performance, turbo disabled,
  GPU clocks {"graphics_mhz": 1771, "sm_mhz": 1771, "memory_mhz": 5005, "active_throttle_reasons": "Idle"}. A throttled run and a regression look the same
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
