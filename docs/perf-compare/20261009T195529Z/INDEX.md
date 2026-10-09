> **Published by hand (2026-10-09).** Part L of the context-policy final gate (`20261009T195800Z-context-policy-final-gate`),
> run by the owner with clocks pinned, `--juno-jar` set to the candidate jar `16f8794d833b1dff`, which is this tree's
> own build (same sha256). The harness does not publish a `--juno-jar` run, so its result files and this INDEX were
> copied here unchanged apart from local paths (replaced by `<repo>/`). A reading for the record, not a gate, and not
> made the reference column.

# llama.cpp vs Juno - 20261009T195529Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 4248.979798 | 206.478606 | 3120.95609124437 | 152.29739550274834 | 150.44 / 152.88 | 0.7345189291588083 | 0.7375940706551861 | 512/512 | 64/64 | 0 | 39008266 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned | 4248.979798 | 206.478606 | 3073.72854198287 | 150.68205348607816 | 147.84 / 150.69 | 0.7234038964905594 | 0.7297707806399959 | 512/512 | 64/64 | 0 | 39044862 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1604.293921 | 75.419175 | 1249.607478033841 | 31.035299405035175 | 30.86 / 31.48 | 0.7789143009748032 | 0.4115040956764003 | 512/512 | 64/64 | 0 | 142185654 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |
| qwen2.5-3b-instruct-q4_k_m-tuned | 1604.293921 | 75.419175 | 1274.1780384485014 | 30.906035801795262 | 30.69 / 31.21 | 0.7942297990222836 | 0.40979016015217967 | 512/512 | 64/64 | 0 | 142156768 | yes | qwen2.5-3b-instruct-q4_k_m-tuned-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1226.216981 | 64.336905 | 716.2342721895843 | 53.184975970263324 | 51.73 / 53.3 | 0.5841007613558602 | 0.826663576220574 | 512/512 | 64/64 | 10 | 180660091 | yes | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| Phi-3.5-mini-instruct-Q4_K_M-tuned | 1226.216981 | 64.336905 | 707.0158523876948 | 53.05962635618385 | 51.53 / 53.09 | 0.5765829892610946 | 0.8247152447912104 | 512/512 | 64/64 | 10 | 180698700 | yes | Phi-3.5-mini-instruct-Q4_K_M-tuned-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 673.195440 | 38.650424 | 539.6326581582666 | 39.20198904669107 | 38.91 / 39.39 | 0.8015988019144435 | 1.0142706079159978 | 512/512 | 64/64 | 0 | 193563615 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 673.195440 | 38.650424 | 540.3554539651653 | 39.42189469326716 | 39.35 / 39.6 | 0.8026724809145549 | 1.019960212940152 | 512/512 | 64/64 | 0 | 193567391 | yes | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

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
- tinyllama-1.1b-chat-v1.0.Q4_K_M: prefill 1860 MHz (low 1860, 61 C, no event reasons); generate 1860 MHz (low 1860, 64 C, no event reasons)
- tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned: prefill 1847 MHz (low 1847, 65 C, no event reasons); generate 1860 MHz (low 1847, 68 C, no event reasons)
- qwen2.5-3b-instruct-q4_k_m: prefill 1835 MHz (low 1822, 77 C, sw_power_cap); generate 1847 MHz (low 1847, 72 C, no event reasons)
- qwen2.5-3b-instruct-q4_k_m-tuned: prefill 1822 MHz (low 1809, 79 C, no event reasons); generate 1835 MHz (low 1835, 74 C, no event reasons)
- Phi-3.5-mini-instruct-Q4_K_M: prefill 1809 MHz (low 1809, 81 C, no event reasons); generate 1809 MHz (low 1809, 81 C, no event reasons)
- Phi-3.5-mini-instruct-Q4_K_M-tuned: prefill 1809 MHz (low 1746, 83 C, sw_thermal_slowdown); generate 1809 MHz (low 1746, 83 C, sw_thermal_slowdown)
- mistral-7b-instruct-v0.1-q4_k_m: prefill 1638.5 MHz (low 1607, 86 C, sw_power_cap, sw_thermal_slowdown); generate 1607 MHz (low 1607, 85 C, sw_power_cap, sw_thermal_slowdown)
- mistral-7b-instruct-v0.1-q4_k_m-tuned: prefill 1619.5 MHz (low 1607, 85 C, sw_power_cap, sw_thermal_slowdown); generate 1607 MHz (low 1607, 85 C, sw_power_cap, sw_thermal_slowdown)

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
- Build: Juno `0f8760386887` plus uncommitted changes, jar sha256 `16f8794d833b1dff`,
  OpenJDK Runtime Environment (build 25.0.3+9-2-24.04.2-Ubuntu); JVM -XX:+UseG1GC -XX:+AlwaysPreTouch -XX:+UnlockExperimentalVMOptions -XX:-UseFastUnorderedTimeStamps -Djava.util.concurrent.ForkJoinPool.common.parallelism=11, -Xms equal to -Xmx per model.
  Reference tool build `ac4cdde` from <home>/Repo/llama.cpp/build-cuda/bin; GPU driver 580.173.02.
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
