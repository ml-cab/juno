# llama.cpp vs Juno - 20261008T174605Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 4085.324026 | 187.881689 | 3051.953821196696 | 147.68846165755312 | 145.65 / 149.45 | 0.7470530615866249 | 0.7860716094454161 | 512/512 | 64/64 | 0 | 39014822 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned | 4085.324026 | 187.881689 | 3052.222016435691 | 146.64037954562838 | 146.1 / 147.32 | 0.7471187100486044 | 0.7804931940207775 | 512/512 | 64/64 | 0 | 39016550 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1525.255781 | 69.643044 | 1257.7632426857958 | 30.33129736316501 | 30.33 / 30.77 | 0.8246244717467461 | 0.4355251525646267 | 512/512 | 64/64 | 0 | 143560762 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |
| qwen2.5-3b-instruct-q4_k_m-tuned | 1525.255781 | 69.643044 | 1269.6912970039189 | 30.618258407876375 | 30.57 / 30.95 | 0.8324448350370939 | 0.43964560779216333 | 512/512 | 64/64 | 0 | 142148802 | yes | qwen2.5-3b-instruct-q4_k_m-tuned-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1182.863532 | 59.409359 | 702.4005733125178 | 51.36239004429825 | 50.97 / 51.56 | 0.5938137023506764 | 0.8645504834397936 | 512/512 | 64/64 | 10 | 179436906 | yes | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| Phi-3.5-mini-instruct-Q4_K_M-tuned | 1182.863532 | 59.409359 | 696.9400585377379 | 50.98130756497587 | 50.86 / 51.14 | 0.5891973500606136 | 0.8581359641496195 | 512/512 | 64/64 | 9 | 179035666 | yes | Phi-3.5-mini-instruct-Q4_K_M-tuned-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 668.781428 | 36.285054 | 538.4209287748403 | 37.41159012125849 | 37.1 / 37.59 | 0.8050775727803857 | 1.031046836012935 | 512/512 | 64/64 | 0 | 192698742 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 668.781428 | 36.285054 | 538.2816154862846 | 37.324822390474175 | 36.99 / 37.35 | 0.8048692636337452 | 1.0286555558240087 | 512/512 | 64/64 | 0 | 193562712 | yes | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

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
- tinyllama-1.1b-chat-v1.0.Q4_K_M: prefill 1822 MHz (low 1822, 80 C, no event reasons); generate 1822 MHz (low 1822, 80 C, no event reasons)
- tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned: prefill 1822 MHz (low 1822, 78 C, no event reasons); generate 1822 MHz (low 1822, 79 C, no event reasons)
- qwen2.5-3b-instruct-q4_k_m: prefill 1809 MHz (low 1683, 84 C, sw_power_cap, sw_thermal_slowdown); generate 1822 MHz (low 1822, 78 C, no event reasons)
- qwen2.5-3b-instruct-q4_k_m-tuned: prefill 1809 MHz (low 1797, 83 C, sw_power_cap); generate 1822 MHz (low 1822, 77 C, no event reasons)
- Phi-3.5-mini-instruct-Q4_K_M: prefill 1809 MHz (low 1733, 83 C, sw_thermal_slowdown); generate 1809 MHz (low 1759, 83 C, sw_thermal_slowdown)
- Phi-3.5-mini-instruct-Q4_K_M-tuned: prefill 1809 MHz (low 1607, 84 C, sw_thermal_slowdown); generate 1809 MHz (low 1670, 83 C, sw_thermal_slowdown)
- mistral-7b-instruct-v0.1-q4_k_m: prefill 1607 MHz (low 1607, 86 C, sw_thermal_slowdown); generate 1607 MHz (low 1607, 85 C, sw_thermal_slowdown)
- mistral-7b-instruct-v0.1-q4_k_m-tuned: prefill 1607 MHz (low 1607, 85 C, sw_thermal_slowdown); generate 1607 MHz (low 1607, 84 C, sw_thermal_slowdown)

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
- Build: Juno `51fa5d46511a` plus uncommitted changes, jar sha256 `43b22345f1175c8f`,
  OpenJDK Runtime Environment (build 25.0.3+9-2-24.04.2-Ubuntu); JVM -XX:+UseG1GC -XX:+AlwaysPreTouch -XX:+UnlockExperimentalVMOptions -XX:-UseFastUnorderedTimeStamps -Djava.util.concurrent.ForkJoinPool.common.parallelism=11, -Xms equal to -Xmx per model.
  Reference tool build `ac4cdde` from <home> GPU driver 580.173.02.
  A different reference build is a measurement boundary: its ratios are not comparable with this run.
- Clocks pinned: cpu governor performance (was schedutil), turbo off, gpu graphics clock locked at 1911 MHz.
- Clock state this run: CPU governor performance, turbo disabled,
  GPU clocks {"graphics_mhz": 1126, "sm_mhz": 1126, "memory_mhz": 5005, "active_throttle_reasons": "Idle"}. A throttled run and a regression look the same
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
