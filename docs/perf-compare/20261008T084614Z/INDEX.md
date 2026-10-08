> **Superseded (2026-10-08).** Closing sweep of the first attention and long-context gate, taken before the decode-width
> attention fix. Prefill is unchanged by that fix; generation is not. The reference column is the re-run:
> `20261008T173153Z`, `20261008T174605Z`, `20261008T180014Z`, `20261008T181417Z`.

# llama.cpp vs Juno - 20261008T084614Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3735.773124 | 181.970791 | 2993.3329763549905 | 131.91367669581206 | 129.84 / 131.94 | 0.8012619816564082 | 0.7249167625798366 | 512/512 | 64/64 | 0 | 38977680 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned | 3735.773124 | 181.970791 | 3040.0624115312894 | 130.5926029709258 | 130.31 / 131.39 | 0.8137706200627641 | 0.7176569506197609 | 512/512 | 64/64 | 0 | 39034370 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1493.382199 | 67.933542 | 1236.0129324419377 | 28.970504041630658 | 28.86 / 29.24 | 0.8276601483997853 | 0.42645360728623066 | 512/512 | 64/64 | 0 | 143557034 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |
| qwen2.5-3b-instruct-q4_k_m-tuned | 1493.382199 | 67.933542 | 1244.2359463580235 | 28.67400142694824 | 28.66 / 29.24 | 0.8331664507526539 | 0.4220890090928608 | 512/512 | 64/64 | 0 | 143515757 | yes | qwen2.5-3b-instruct-q4_k_m-tuned-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1160.922956 | 57.875357 | 693.6305039328165 | 47.71240414431702 | 47.61 / 47.85 | 0.5974819434381281 | 0.8243993059829768 | 512/512 | 64/64 | 9 | 174713296 | yes | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| Phi-3.5-mini-instruct-Q4_K_M-tuned | 1160.922956 | 57.875357 | 687.226311534202 | 47.40698084992861 | 47.36 / 47.7 | 0.5919654771080277 | 0.8191220461919329 | 512/512 | 64/64 | 9 | 174629134 | yes | Phi-3.5-mini-instruct-Q4_K_M-tuned-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 664.202372 | 35.993565 | 535.4965771827517 | 34.99806490504115 | 34.95 / 35.26 | 0.8062250298358642 | 0.9723422757662696 | 512/512 | 64/64 | 0 | 193561877 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 664.202372 | 35.993565 | 539.0700846048809 | 35.30961085522837 | 34.87 / 35.34 | 0.8116051783760883 | 0.9809978771268801 | 512/512 | 64/64 | 0 | 193563463 | yes | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

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
- tinyllama-1.1b-chat-v1.0.Q4_K_M: prefill 1809 MHz (low 1733, 83 C, no event reasons); generate 1822 MHz (low 1822, 81 C, no event reasons)
- tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned: prefill 1809 MHz (low 1809, 82 C, no event reasons); generate 1809 MHz (low 1809, 80 C, no event reasons)
- qwen2.5-3b-instruct-q4_k_m: prefill 1809 MHz (low 1695, 85 C, sw_power_cap); generate 1809 MHz (low 1809, 79 C, no event reasons)
- qwen2.5-3b-instruct-q4_k_m-tuned: prefill 1809 MHz (low 1683, 85 C, sw_power_cap, sw_thermal_slowdown); generate 1822 MHz (low 1822, 79 C, no event reasons)
- Phi-3.5-mini-instruct-Q4_K_M: prefill 1796.5 MHz (low 1620, 84 C, sw_thermal_slowdown); generate 1777.5 MHz (low 1657, 83 C, sw_thermal_slowdown)
- Phi-3.5-mini-instruct-Q4_K_M-tuned: prefill 1759 MHz (low 1607, 84 C, sw_thermal_slowdown); generate 1771 MHz (low 1683, 83 C, sw_thermal_slowdown)
- mistral-7b-instruct-v0.1-q4_k_m: prefill 1607 MHz (low 1607, 87 C, sw_power_cap, sw_thermal_slowdown); generate 1607 MHz (low 1607, 85 C, sw_thermal_slowdown)
- mistral-7b-instruct-v0.1-q4_k_m-tuned: prefill 1632 MHz (low 1607, 87 C, sw_power_cap, sw_thermal_slowdown); generate 1607 MHz (low 1607, 85 C, sw_thermal_slowdown)

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
