# llama.cpp vs Juno - 20261027T172646Z (cpu)

> **CPU reference at `n_prompt=2048` from 2026-10-27** (TinyLlama, Qwen2.5-3B, Mistral 7B; supersedes [`20261012T071213Z`](../20261012T071213Z/INDEX.md) as reference). The closing sweep of the CPU decode and prefill kernel work, owner run, pinned clocks, jar `4e480ab9131887eb` (`c2633a68` plus the in-recording allocation field): the vector Q6_K decode kernel, the split Q4_K kernel, pooled vector prefill attention and the padded prefill tile, `--threads 12` against `-t 12`. A measurement boundary for CPU readings; compare ratios across it. Gate results: [`20261027T154748Z-cpu-kernels-close-gate-2`](../20261027T154748Z-cpu-kernels-close-gate-2/INDEX.md). Local paths are replaced by `<repo>` and `<home>`. pp ratio 0.378x (TinyLlama) to 0.411x. Phi-3.5-mini at 2048 is [`20261027T202604Z`](../20261027T202604Z/INDEX.md).

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Alloc B/tok in request | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|------------------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 70.661296 | 36.081143 | 26.730716857457118 | 12.240597382204156 | 11.59 / 12.31 | 0.3782936115049053 | 0.3392519295246316 | 2048/2048 | 64/64 | 8 | 139104436 | 300373 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 26.210621 | 14.806270 | 10.396053344936668 | 3.970275708564476 | 3.96 / 4.01 | 0.39663514057666427 | 0.2681482715474239 | 2048/2048 | 64/64 | 2 | 242765020 | 1288588 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 10.813914 | 6.937544 | 4.446814880829738 | 1.9415467428961928 | 1.92 / 1.96 | 0.41121234003060664 | 0.2798608185975026 | 2048/2048 | 64/64 | 9 | 254395580 | 388339 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |

GPU attention per row, as the engine resolved it (read from its log, not from the flag):
- tinyllama-1.1b-chat-v1.0.Q4_K_M: n/a (requested: engine default)
- qwen2.5-3b-instruct-q4_k_m: n/a (requested: engine default)
- mistral-7b-instruct-v0.1-q4_k_m: n/a (requested: engine default)

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
- Build: Juno `c2633a68a974` plus uncommitted changes, jar sha256 `4e480ab9131887eb`,
  OpenJDK Runtime Environment (build 25.0.3+9-2-24.04.2-Ubuntu); JVM -XX:+UseG1GC -XX:+AlwaysPreTouch -XX:+UnlockExperimentalVMOptions -XX:-UseFastUnorderedTimeStamps -Djuno.threads=12 -Djava.util.concurrent.ForkJoinPool.common.parallelism=11, -Xms equal to -Xmx per model.
  Reference tool build `379ac6673` from <home>/Repo/llama.cpp-bin/llama-b9551; GPU driver 580.173.02.
  A different reference build is a measurement boundary: its ratios are not comparable with this run.
- Clocks pinned: cpu governor performance (was schedutil), turbo off.
- Clock state this run: CPU governor performance, turbo disabled,
  GPU clocks {"graphics_mhz": 1607, "sm_mhz": 1607, "memory_mhz": 4513, "active_throttle_reasons": "none"}. A throttled run and a regression look the same
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
- Juno pp/tg from JFR (--jfr 120m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: cpu (llama -ngl 0 / juno --cpu), temperature 0, max_tokens=64.
- Juno ran with jdk.incubator.vector. On CPUs without HW FMA this can be pathologically slow; re-run with --vector 0.
