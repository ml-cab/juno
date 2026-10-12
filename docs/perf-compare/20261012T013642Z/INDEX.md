# llama.cpp vs Juno - 20261012T013642Z (gpu)

> **GPU reference from 2026-10-12 at `n_prompt=128`**: the closing sweep of the GPU host-overhead work (Qwen2 in the device-resident decode region, host KV reuse, direct K/V write), owner run with pinned clocks, jar `921ab742ba3f362a` (HEAD `198467ad` plus the uncommitted work). Same method as the 2026-10-08 reference sweeps, which it supersedes as the GPU reference. Gate summary and the A/B against the pre-work build: [`20261012T002824Z-gpu-host-overhead-close-gate`](../20261012T002824Z-gpu-host-overhead-close-gate/INDEX.md). Local paths are replaced by `<repo>` and `<home>`.

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3532.874081 | 194.869526 | 2405.0639975316526 | 151.15205379388232 | 145.57 / 152.06 | 0.680766973967803 | 0.7756577280014645 | 128/128 | 64/64 | 0 | 34431832 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned | 3532.874081 | 194.869526 | 2617.800404662859 | 152.48987117600882 | 152.47 / 153.41 | 0.7409832178116792 | 0.7825229234457562 | 128/128 | 64/64 | 0 | 34457802 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1473.984171 | 71.472612 | 1119.5598030799265 | 59.194671131439534 | 58.76 / 59.4 | 0.7595466933137958 | 0.8282147451311774 | 128/128 | 64/64 | 0 | 113567904 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |
| qwen2.5-3b-instruct-q4_k_m-tuned | 1473.984171 | 71.472612 | 1101.7513697997313 | 59.19931036503382 | 58.61 / 59.35 | 0.7474648584945566 | 0.8282796543805314 | 128/128 | 64/64 | 0 | 113543078 | yes | qwen2.5-3b-instruct-q4_k_m-tuned-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1180.280876 | 60.149471 | 820.627413766548 | 58.554076039625976 | 57.83 / 58.67 | 0.6952814626190283 | 0.9734761597425516 | 128/128 | 64/64 | 0 | 100461478 | yes | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| Phi-3.5-mini-instruct-Q4_K_M-tuned | 1180.280876 | 60.149471 | 821.6927607827821 | 58.43732363883944 | 58.4 / 58.47 | 0.6961840842219849 | 0.9715351218772886 | 128/128 | 64/64 | 0 | 100502262 | yes | Phi-3.5-mini-instruct-Q4_K_M-tuned-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 653.555922 | 36.618091 | 512.7937370161277 | 39.023618432873484 | 38.71 / 39.07 | 0.7846210549923343 | 1.0656923222150898 | 128/128 | 64/64 | 0 | 166822344 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 653.555922 | 36.618091 | 541.9700345699397 | 38.835765781991725 | 38.38 / 39.05 | 0.8292634437637911 | 1.0605622718560541 | 128/128 | 64/64 | 0 | 166795977 | yes | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

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
- tinyllama-1.1b-chat-v1.0.Q4_K_M: prefill 1657 MHz (low 1657, 47 C, no event reasons); generate 1860 MHz (low 1860, 58 C, no event reasons)
- tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned: prefill 1860 MHz (low 1607, 54 C, no event reasons); generate 1860 MHz (low 1860, 63 C, no event reasons)
- qwen2.5-3b-instruct-q4_k_m: prefill 1860 MHz (low 1860, 65 C, no event reasons); generate 1847 MHz (low 1847, 71 C, no event reasons)
- qwen2.5-3b-instruct-q4_k_m-tuned: prefill 1847 MHz (low 1847, 69 C, no event reasons); generate 1835 MHz (low 1835, 74 C, no event reasons)
- Phi-3.5-mini-instruct-Q4_K_M: prefill 1835 MHz (low 1835, 73 C, no event reasons); generate 1822 MHz (low 1822, 79 C, no event reasons)
- Phi-3.5-mini-instruct-Q4_K_M-tuned: prefill 1835 MHz (low 1835, 76 C, no event reasons); generate 1809 MHz (low 1809, 80 C, no event reasons)
- mistral-7b-instruct-v0.1-q4_k_m: prefill 1809 MHz (low 1809, 82 C, no event reasons); generate 1714.5 MHz (low 1620, 84 C, sw_thermal_slowdown)
- mistral-7b-instruct-v0.1-q4_k_m-tuned: prefill 1809 MHz (low 1809, 83 C, no event reasons); generate 1664 MHz (low 1607, 84 C, sw_thermal_slowdown)

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
  GPU clocks {"graphics_mhz": 1797, "sm_mhz": 1797, "memory_mhz": 5005, "active_throttle_reasons": "Idle"}. A throttled run and a regression look the same
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
