# llama.cpp vs Juno - 20261008T002907Z (gpu)

> **`--gpu-residency off` half of the residency-default pair (2026-10-08), unpinned.** Full GPU sweep on the four sweep models, `n_prompt=128`, `n_gen=64`, `--juno-reps 3`, default and tuned lanes, flags as `scripts/performance-tests/compare-llama-cpp.sh --gpu --gpu-residency off`; jar sha256 `94ac42bac3d03b4a` (HEAD `2acc808`, tree clean); governor `schedutil`, turbo on, GPU clock recorded, not fixed. Taken immediately before its pair [`20261008T004314Z`](../20261008T004314Z/INDEX.md) (`--gpu-residency on`), which carries the side-by-side reading. Not a reference column and not scorable against a pinned gate: the reference tool reads 12% to 28% faster here (TinyLlama tg 189.3 against 147.6 t/s at `20261004T113210Z`) than in the pinned reference sweeps, so compare the two halves of this pair with each other, never with a pinned run. Local paths made repository-relative.

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3451.437394 | 189.323607 | 2289.7684024482637 | 64.44819816261288 | 64.23 / 65.86 | 0.6634245796921628 | 0.3404129003448201 | 128/128 | 64/64 | 7 | 48195609 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned | 3451.437394 | 189.323607 | 2268.7307295252544 | 65.00829318922251 | 64.88 / 65.24 | 0.6573292430189317 | 0.34337130070220195 | 128/128 | 64/64 | 11 | 48249335 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1471.717140 | 71.515013 | 1079.5248602894528 | 29.21401426464262 | 28.98 / 29.23 | 0.733513819299171 | 0.40850183813352064 | 128/128 | 64/64 | 0 | 143341108 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |
| qwen2.5-3b-instruct-q4_k_m-tuned | 1471.717140 | 71.515013 | 1069.5429618153025 | 28.98083722334017 | 28.88 / 29.39 | 0.7267313349461314 | 0.40524130539331893 | 128/128 | 64/64 | 0 | 143553317 | yes | qwen2.5-3b-instruct-q4_k_m-tuned-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1189.391182 | 60.264607 | 684.1639311512154 | 31.134764050821655 | 30.97 / 31.17 | 0.5752219635601901 | 0.5166343165702824 | 128/128 | 64/64 | 9 | 207582567 | yes | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| Phi-3.5-mini-instruct-Q4_K_M-tuned | 1189.391182 | 60.264607 | 696.3169939159793 | 31.249834336205378 | 31.24 / 31.29 | 0.5854398489360747 | 0.5185437339067918 | 128/128 | 64/64 | 9 | 208591748 | yes | Phi-3.5-mini-instruct-Q4_K_M-tuned-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 658.892244 | 36.815573 | 517.4248204878464 | 22.255398654098645 | 21.81 / 22.3 | 0.7852950542362833 | 0.604510451435827 | 128/128 | 64/64 | 14 | 224139701 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 658.892244 | 36.815573 | 512.7572460540596 | 22.13863817047393 | 22.13 / 22.38 | 0.778211081892868 | 0.6013389543189761 | 128/128 | 64/64 | 14 | 223588036 | yes | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

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
- tinyllama-1.1b-chat-v1.0.Q4_K_M: prefill 1607 MHz (low 1607, 53 C, no event reasons); generate 1873 MHz (low 1746, 59 C, no event reasons)
- tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned: prefill 1607 MHz (low 1607, 55 C, no event reasons); generate 1860 MHz (low 1746, 62 C, no event reasons)
- qwen2.5-3b-instruct-q4_k_m: prefill 1847 MHz (low 1847, 65 C, no event reasons); generate 1847 MHz (low 1847, 68 C, no event reasons)
- qwen2.5-3b-instruct-q4_k_m-tuned: prefill 1847 MHz (low 1835, 68 C, no event reasons); generate 1847 MHz (low 1835, 70 C, no event reasons)
- Phi-3.5-mini-instruct-Q4_K_M: prefill 1835 MHz (low 1835, 74 C, no event reasons); generate 1835 MHz (low 1835, 75 C, no event reasons)
- Phi-3.5-mini-instruct-Q4_K_M-tuned: prefill 1835 MHz (low 1822, 75 C, no event reasons); generate 1835 MHz (low 1822, 76 C, no event reasons)
- mistral-7b-instruct-v0.1-q4_k_m: prefill 1822 MHz (low 1822, 80 C, no event reasons); generate 1822 MHz (low 1822, 81 C, no event reasons)
- mistral-7b-instruct-v0.1-q4_k_m-tuned: prefill 1809 MHz (low 1809, 83 C, no event reasons); generate 1809 MHz (low 1733, 83 C, sw_thermal_slowdown)

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
- Build: Juno `2acc808e2e58`, jar sha256 `94ac42bac3d03b4a`,
  OpenJDK Runtime Environment (build 25.0.3+9-2-24.04.2-Ubuntu); JVM -XX:+UseG1GC -XX:+AlwaysPreTouch -XX:+UnlockExperimentalVMOptions -XX:-UseFastUnorderedTimeStamps -Djava.util.concurrent.ForkJoinPool.common.parallelism=11, -Xms equal to -Xmx per model.
  Reference tool build `ac4cdde` from ../llama.cpp/build-cuda/bin; GPU driver 580.173.02.
  A different reference build is a measurement boundary: its ratios are not comparable with this run.
- Clocks not pinned (not requested). This run can be read against the 15% noise floor
  only; it is not usable for a gate tighter than that (see --pin-clocks).
- Clock state this run: CPU governor schedutil, turbo enabled,
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
