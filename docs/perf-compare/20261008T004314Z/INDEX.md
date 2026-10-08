# llama.cpp vs Juno - 20261008T004314Z (gpu)

> **`--gpu-residency on` half of the residency-default pair (2026-10-08), unpinned.** Same command, jar and session as its pair [`20261008T002907Z`](../20261008T002907Z/INDEX.md) (`--gpu-residency off`), run directly after it: `scripts/performance-tests/compare-llama-cpp.sh --gpu --gpu-residency on`; jar sha256 `94ac42bac3d03b4a` (HEAD `2acc808`, tree clean); governor `schedutil`, turbo on, GPU clock recorded, not fixed. Purpose: the two full sweeps on which the default of `--gpu-residency` is put to the owner. Not a reference column. Local paths made repository-relative.

## Region on against off (this run against its pair; Juno medians of three)

| Model, lane | Region | tg off | tg on | tg on/off | pp on/off | GPU tg ratio off / on | GPU pp ratio off / on | alloc/token off / on | GC max ms off / on |
|---|---|---|---|---|---|---|---|---|---|
| tinyllama-1.1b | whole layer | 64.45 | 128.51 | **1.994** | 0.995 | 0.340x / 0.701x | 0.663x / 0.676x | 48.2M / 38.9M | 8 / 0 |
| tinyllama-1.1b tuned | whole layer | 65.01 | 130.48 | **2.007** | 1.014 | 0.343x / 0.712x | 0.657x / 0.682x | 48.2M / 38.9M | 12 / 0 |
| qwen2.5-3b | declined (qwen2) | 29.21 | 29.06 | **0.995** | 0.987 | 0.409x / 0.420x | 0.734x / 0.740x | 143.3M / 143.6M | 0 / 0 |
| qwen2.5-3b tuned | declined (qwen2) | 28.98 | 29.31 | **1.011** | 0.995 | 0.405x / 0.423x | 0.727x / 0.739x | 143.6M / 143.6M | 0 / 0 |
| Phi-3.5-mini | whole layer | 31.13 | 48.60 | **1.561** | 0.995 | 0.517x / 0.816x | 0.575x / 0.578x | 207.6M / 175.0M | 9 / 10 |
| Phi-3.5-mini tuned | whole layer | 31.25 | 48.57 | **1.554** | 0.979 | 0.519x / 0.816x | 0.585x / 0.579x | 208.6M / 175.0M | 9 / 10 |
| mistral-7b | whole layer | 22.26 | 35.22 | **1.583** | 0.998 | 0.605x / 0.977x | 0.785x / 0.824x | 224.1M / 193.6M | 14 / 0 |
| mistral-7b tuned | whole layer | 22.14 | 34.85 | **1.574** | 0.988 | 0.601x / 0.967x | 0.778x / 0.809x | 223.6M / 193.6M | 14 / 0 |

How to read it:
- Every row of both runs is scorable (repetition spread under 15%). On every region model every on repetition is above every off repetition (tg min/max columns of the two INDEX tables).
- Prefill does not run the decode region; pp on/off sits within 0.979 to 1.014. The pp ratio columns move with the reference tool's own reading between the two runs.
- Qwen2.5-3B is declined with the console notice (`--gpu-residency=on has no effect here: architecture qwen2 uses the split-half RoPE layout (and Q/K/V biases) ...`) and runs the existing path.
- Greedy text over the 64 generated tokens: each mode is identical across its three repetitions and both lanes on every model. On against off it is identical on Qwen2.5-3B and Phi-3.5-mini, and parts on TinyLlama (at character 93 of 223) and Mistral 7B (at character 152 of 263), as at every earlier gate of this region: the region's norms sum in a different order than the CPU norms of the default path.
- The GPU ratios divide by the reference tool's reading in the same run (one repetition), unpinned; they are readings for this decision, not scores against the end-of-plan targets, which the pinned closing sweeps score.

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3370.938765 | 183.371061 | 2279.4505591002494 | 128.51422989975245 | 127.49 / 129.85 | 0.676206457016507 | 0.7008424840806939 | 128/128 | 64/64 | 0 | 38873012 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned | 3370.938765 | 183.371061 | 2300.5286632841107 | 130.48479507270167 | 130.29 / 130.54 | 0.6824593455004843 | 0.7115888099306011 | 128/128 | 64/64 | 0 | 38937219 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1440.017909 | 69.211791 | 1065.367796882902 | 29.05829911476838 | 28.46 / 29.15 | 0.7398295467191318 | 0.419846079619127 | 128/128 | 64/64 | 0 | 143556601 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |
| qwen2.5-3b-instruct-q4_k_m-tuned | 1440.017909 | 69.211791 | 1064.6322785893815 | 29.308781030413172 | 29.02 / 29.48 | 0.7393187764787594 | 0.4234651438280678 | 128/128 | 64/64 | 0 | 143555253 | yes | qwen2.5-3b-instruct-q4_k_m-tuned-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1178.309247 | 59.523766 | 680.6075547555558 | 48.59877786344865 | 48.31 / 48.75 | 0.5776136922360552 | 0.8164600651015369 | 128/128 | 64/64 | 9 | 175032293 | yes | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| Phi-3.5-mini-instruct-Q4_K_M-tuned | 1178.309247 | 59.523766 | 681.8813466472584 | 48.56739347963255 | 48.23 / 48.6 | 0.5786947258398784 | 0.8159328070678953 | 128/128 | 64/64 | 9 | 175032721 | yes | Phi-3.5-mini-instruct-Q4_K_M-tuned-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 626.619783 | 36.055616 | 516.2907624212105 | 35.22352688906089 | 34.98 / 35.74 | 0.8239298796942236 | 0.9769220664281784 | 128/128 | 64/64 | 0 | 193562646 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 626.619783 | 36.055616 | 506.6522009019106 | 34.84815047206603 | 32.09 / 35.47 | 0.8085480456366482 | 0.9665110276320347 | 128/128 | 64/64 | 0 | 193586786 | yes | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

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
- tinyllama-1.1b-chat-v1.0.Q4_K_M: prefill 1809 MHz (low 1809, 78 C, no event reasons); generate 1822 MHz (low 1822, 81 C, no event reasons)
- tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned: prefill 1822 MHz (low 1607, 76 C, no event reasons); generate 1822 MHz (low 1822, 79 C, no event reasons)
- qwen2.5-3b-instruct-q4_k_m: prefill 1822 MHz (low 1822, 78 C, no event reasons); generate 1822 MHz (low 1822, 78 C, no event reasons)
- qwen2.5-3b-instruct-q4_k_m-tuned: prefill 1822 MHz (low 1822, 77 C, no event reasons); generate 1822 MHz (low 1822, 77 C, no event reasons)
- Phi-3.5-mini-instruct-Q4_K_M: prefill 1822 MHz (low 1822, 80 C, no event reasons); generate 1822 MHz (low 1809, 82 C, no event reasons)
- Phi-3.5-mini-instruct-Q4_K_M-tuned: prefill 1822 MHz (low 1822, 80 C, no event reasons); generate 1809 MHz (low 1809, 83 C, no event reasons)
- mistral-7b-instruct-v0.1-q4_k_m: prefill 1809 MHz (low 1746, 83 C, no event reasons); generate 1638.5 MHz (low 1607, 84 C, sw_thermal_slowdown)
- mistral-7b-instruct-v0.1-q4_k_m-tuned: prefill 1809 MHz (low 1809, 84 C, no event reasons); generate 1683 MHz (low 1607, 84 C, sw_thermal_slowdown)

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
