# llama.cpp vs Juno - 20261027T202604Z (cpu)

> **CPU reference at `n_prompt=2048` from 2026-10-27, Phi-3.5-mini** at an explicit 8 GiB heap (supersedes [`20261012T162344Z`](../20261012T162344Z/INDEX.md) as reference). The closing sweep of the CPU decode and prefill kernel work, owner run, pinned clocks, jar `4e480ab9131887eb` (`c2633a68` plus the in-recording allocation field): the vector Q6_K decode kernel, the split Q4_K kernel, pooled vector prefill attention and the padded prefill tile, `--threads 12` against `-t 12`. A measurement boundary for CPU readings; compare ratios across it. Gate results: [`20261027T154748Z-cpu-kernels-close-gate-2`](../20261027T154748Z-cpu-kernels-close-gate-2/INDEX.md). Local paths are replaced by `<repo>` and `<home>`. pp ratio 0.388x. `NOISY` is the generation lane only (three readings 2.52 to 2.96 t/s); the prefill lane's three readings span 6.34 to 6.54 t/s (3.1%) and are scorable.

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Alloc B/tok in request | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|------------------------|----------|---------|
| Phi-3.5-mini-instruct-Q4_K_M | 16.752080 | 5.249058 | 6.504622352140633 | 2.7746934037816775 | 2.52 / 2.96 | 0.3882874456270883 | 0.528607876647901 | 2048/2048 | 64/64 | 2 | 250062575 | 358996 | NOISY | Phi-3.5-mini-instruct-Q4_K_M-*.json |

GPU attention per row, as the engine resolved it (read from its log, not from the flag):
- Phi-3.5-mini-instruct-Q4_K_M: n/a (requested: engine default)

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
  request; the generation run prefilled 10 tokens against the reference 0.
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
  GPU clocks {"graphics_mhz": 316, "sm_mhz": 316, "memory_mhz": 405, "active_throttle_reasons": "Idle"}. A throttled run and a regression look the same
  without this.
- Thread counts are matched: the reference tool ran with -t 12, and Juno's CPU matmuls ran
  on 12 threads (-Djuno.threads=12, the property --threads sets). A build from
  before the worker pool runs them on common pool parallelism 11 plus the calling thread.
- Rows marked NOISY are not scorable and should be re-run:
  - Phi-3.5-mini-instruct-Q4_K_M: the 3 generation readings span 15% of their median, over the 15% this host can resolve: re-run rather than score this row
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
