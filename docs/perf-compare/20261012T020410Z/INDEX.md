# llama.cpp vs Juno - 20261012T020410Z (gpu)

> **GPU reference from 2026-10-12 at `n_prompt=2048`** (TinyLlama, Qwen2.5-3B, Mistral 7B): the closing sweep of the GPU host-overhead work (Qwen2 in the device-resident decode region, host KV reuse, direct K/V write), owner run with pinned clocks, jar `921ab742ba3f362a` (HEAD `198467ad` plus the uncommitted work). Same method as the 2026-10-08 reference sweeps, which it supersedes as the GPU reference. Gate summary and the A/B against the pre-work build: [`20261012T002824Z-gpu-host-overhead-close-gate`](../20261012T002824Z-gpu-host-overhead-close-gate/INDEX.md). Local paths are replaced by `<repo>` and `<home>`.

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3297.818558 | 181.937585 | 2349.7791172635502 | 148.9991932555087 | 147.65 / 152.07 | 0.712525287834089 | 0.8189577390263187 | 2048/2048 | 64/64 | 17 | 34456841 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned | 3297.818558 | 181.937585 | 2361.107066889025 | 152.03679501390388 | 149.56 / 152.56 | 0.7159602705131679 | 0.8356535842437607 | 2048/2048 | 64/64 | 17 | 34418346 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1356.476284 | 67.251573 | 922.0854673023433 | 56.97922060504797 | 56.85 / 57.42 | 0.6797652698971501 | 0.8472548382630095 | 2048/2048 | 64/64 | 22 | 113541937 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |
| qwen2.5-3b-instruct-q4_k_m-tuned | 1356.476284 | 67.251573 | 935.134982472276 | 57.604718246938866 | 57.17 / 57.99 | 0.6893854271559648 | 0.8565557008895371 | 2048/2048 | 64/64 | 22 | 113570556 | yes | qwen2.5-3b-instruct-q4_k_m-tuned-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 613.132214 | 35.961893 | 447.2850894392084 | 38.12544440827472 | 37.66 / 38.15 | 0.7295083820847952 | 1.0601623337312835 | 2048/2048 | 64/64 | 8 | 166777052 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 613.132214 | 35.961893 | 447.228415905102 | 37.9627304049901 | 37.72 / 38.17 | 0.7294159492737108 | 1.0556377108677257 | 2048/2048 | 64/64 | 8 | 166765592 | yes | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

GPU attention per row, as the engine resolved it (read from its log, not from the flag):
- tinyllama-1.1b-chat-v1.0.Q4_K_M: unknown (requested: engine default)
- tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned: unknown (requested: auto)
- qwen2.5-3b-instruct-q4_k_m: unknown (requested: engine default)
- qwen2.5-3b-instruct-q4_k_m-tuned: unknown (requested: auto)
- mistral-7b-instruct-v0.1-q4_k_m: unknown (requested: engine default)
- mistral-7b-instruct-v0.1-q4_k_m-tuned: unknown (requested: auto)

GPU clock while the measured requests ran (sampled every 100 ms; SM clock median over busy
samples, median across reps; lowest SM clock and hottest sample of any rep; clock event
reasons seen). The clock under "Clock state" below is read once before the run and says
nothing about a long window, which can heat the card into thermal slowdown:
- tinyllama-1.1b-chat-v1.0.Q4_K_M: prefill 1607 MHz (low 1607, 87 C, sw_power_cap, sw_thermal_slowdown); generate 1809 MHz (low 1809, 82 C, no event reasons)
- tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned: prefill 1727 MHz (low 1607, 86 C, sw_power_cap, sw_thermal_slowdown); generate 1809 MHz (low 1809, 82 C, no event reasons)
- qwen2.5-3b-instruct-q4_k_m: prefill 1607 MHz (low 1607, 85 C, sw_power_cap, sw_thermal_slowdown); generate 1683 MHz (low 1607, 83 C, sw_thermal_slowdown)
- qwen2.5-3b-instruct-q4_k_m-tuned: prefill 1607 MHz (low 1607, 86 C, sw_thermal_slowdown); generate 1708 MHz (low 1645, 83 C, sw_thermal_slowdown)
- mistral-7b-instruct-v0.1-q4_k_m: prefill 1607 MHz (low 1607, 91 C, sw_thermal_slowdown); generate 1607 MHz (low 1607, 88 C, sw_thermal_slowdown)
- mistral-7b-instruct-v0.1-q4_k_m-tuned: prefill 1607 MHz (low 1607, 91 C, sw_thermal_slowdown); generate 1607 MHz (low 1607, 89 C, sw_thermal_slowdown)

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
- Build: Juno `198467ad0d91` plus uncommitted changes, jar sha256 `921ab742ba3f362a`,
  OpenJDK Runtime Environment (build 25.0.3+9-2-24.04.2-Ubuntu); JVM -XX:+UseG1GC -XX:+AlwaysPreTouch -XX:+UnlockExperimentalVMOptions -XX:-UseFastUnorderedTimeStamps -Djuno.threads=12 -Djava.util.concurrent.ForkJoinPool.common.parallelism=11, -Xms equal to -Xmx per model.
  Reference tool build `ac4cdde` from <home> GPU driver 580.173.02.
  A different reference build is a measurement boundary: its ratios are not comparable with this run.
- Clocks pinned: cpu governor performance (was schedutil), turbo off, gpu graphics clock locked at 1911 MHz.
- Clock state this run: CPU governor performance, turbo disabled,
  GPU clocks {"graphics_mhz": 1670, "sm_mhz": 1670, "memory_mhz": 5005, "active_throttle_reasons": "Idle"}. A throttled run and a regression look the same
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
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran with jdk.incubator.vector. On CPUs without HW FMA this can be pathologically slow; re-run with --vector 0.
