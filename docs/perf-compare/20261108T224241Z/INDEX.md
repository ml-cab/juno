# llama.cpp vs Juno - 20261108T224241Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Alloc B/tok in request | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|------------------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 4105.184442 | 187.206450 | 3145.672911566711 | 134.02943081905906 | 130.67 / 144.62 | 0.7662683506698116 | 0.7159445137657333 | 512/512 | 64/64 | 0 | 34437888 | 585285 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned | 4105.184442 | 187.206450 | 3158.975510431057 | 138.6373679695691 | 137.44 / 138.92 | 0.7695087894496744 | 0.740558714561219 | 512/512 | 64/64 | 0 | 34457580 | 633687 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1470.256566 | 69.169935 | 1181.366309204911 | 55.00238577145345 | 54.8 / 56.87 | 0.8035103100535386 | 0.7951776414341498 | 512/512 | 64/64 | 0 | 113545069 | 1835317 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |
| qwen2.5-3b-instruct-q4_k_m-tuned | 1470.256566 | 69.169935 | 1270.071766447019 | 55.46643726487766 | 53.99 / 56.03 | 0.8638436282603338 | 0.801886502638432 | 512/512 | 64/64 | 0 | 113544212 | 1864730 | yes | qwen2.5-3b-instruct-q4_k_m-tuned-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1165.560413 | 56.908695 | 830.702936281775 | 55.45210757903992 | 55.4 / 55.71 | 0.7127068893354522 | 0.974404835307503 | 512/512 | 64/64 | 0 | 100509365 | 601601 | yes | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| Phi-3.5-mini-instruct-Q4_K_M-tuned | 1165.560413 | 56.908695 | 819.0484039690062 | 57.85009113893178 | 56.09 / 58.65 | 0.7027078089078909 | 1.0165422197597709 | 512/512 | 64/64 | 0 | 100509083 | 578537 | yes | Phi-3.5-mini-instruct-Q4_K_M-tuned-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 622.484691 | 34.028896 | 538.5598592958512 | 37.06747013790597 | 36.93 / 37.45 | 0.8651776775918353 | 1.0892939382431321 | 512/512 | 64/64 | 0 | 166843486 | 760247 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 622.484691 | 34.028896 | 535.1371423965904 | 36.74834476268103 | 36.62 / 36.88 | 0.8596792019686641 | 1.0799158680517 | 512/512 | 64/64 | 0 | 166844024 | 797327 | yes | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

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
- tinyllama-1.1b-chat-v1.0.Q4_K_M: prefill 1885 MHz (low 1885, 51 C, no event reasons); generate 1873 MHz (low 1873, 56 C, no event reasons)
- tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned: prefill 1873 MHz (low 1860, 56 C, no event reasons); generate 1860 MHz (low 1860, 61 C, no event reasons)
- qwen2.5-3b-instruct-q4_k_m: prefill 1835 MHz (low 1835, 70 C, no event reasons); generate 1835 MHz (low 1835, 70 C, no event reasons)
- qwen2.5-3b-instruct-q4_k_m-tuned: prefill 1835 MHz (low 1822, 76 C, no event reasons); generate 1835 MHz (low 1835, 74 C, no event reasons)
- Phi-3.5-mini-instruct-Q4_K_M: prefill 1809 MHz (low 1809, 80 C, no event reasons); generate 1822 MHz (low 1822, 80 C, no event reasons)
- Phi-3.5-mini-instruct-Q4_K_M-tuned: prefill 1809 MHz (low 1809, 82 C, no event reasons); generate 1809 MHz (low 1809, 81 C, no event reasons)
- mistral-7b-instruct-v0.1-q4_k_m: prefill 1683 MHz (low 1607, 85 C, sw_power_cap, sw_thermal_slowdown); generate 1645 MHz (low 1607, 84 C, sw_thermal_slowdown)
- mistral-7b-instruct-v0.1-q4_k_m-tuned: prefill 1721 MHz (low 1607, 87 C, sw_power_cap, sw_thermal_slowdown); generate 1607 MHz (low 1607, 85 C, sw_thermal_slowdown)

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
- Build: Juno `d8012b11f98f` plus uncommitted changes, jar sha256 `796de19984b3e9ae`,
  OpenJDK Runtime Environment (build 25.0.4.1+1-1-24.04.4-Ubuntu); JVM -XX:+UseG1GC -XX:+AlwaysPreTouch -XX:+UnlockExperimentalVMOptions -XX:-UseFastUnorderedTimeStamps -Djuno.threads=12 -Djava.util.concurrent.ForkJoinPool.common.parallelism=11, -Xms equal to -Xmx per model.
  Reference tool build `ac4cdde` from /home/medion/Repo/llama.cpp/build-cuda/bin; GPU driver 580.178.04.
  A different reference build is a measurement boundary: its ratios are not comparable with this run.
- Clocks pinned: cpu governor performance (was schedutil), turbo off, gpu graphics clock locked at 1911 MHz.
- Clock state this run: CPU governor performance, turbo disabled,
  GPU clocks {"graphics_mhz": 1771, "sm_mhz": 1771, "memory_mhz": 5005, "active_throttle_reasons": "Idle"}. A throttled run and a regression look the same
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
- Alloc B/tok sums each thread's running allocation total since the thread started, so
  model load and the warmup requests are in it. Alloc B/tok in request is each thread's
  growth inside the recording of the measured request (its last running total minus
  its first; a thread sampled once adds nothing), which is what an allocation gate reads.
  Both are generation-lane figures, read by the extractor in host.json (jfr_extractor_jar).
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran with jdk.incubator.vector. On CPUs without HW FMA this can be pathologically slow; re-run with --vector 0.
