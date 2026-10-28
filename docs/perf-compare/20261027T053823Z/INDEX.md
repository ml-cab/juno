# llama.cpp vs Juno - 20261027T053823Z (cpu)

> **Not scored: interrupted closing gate attempt.** Run 2 of the CPU kernels closing gate, first attempt, pinned, jar `4e480ab9131887eb`. The host reset at about 10:56Z during the attempt's last sweep, so the gate was re-run whole and scored on its complete run ([`20261027T154748Z-cpu-kernels-close-gate-2`](../20261027T154748Z-cpu-kernels-close-gate-2/INDEX.md)). This sweep is a valid pinned reading of the same jar, kept because it records how far the reference tool's CPU decode reading moves between sessions: the complete run, hours later on the same jar, read it 11% to 34% higher (128 and 512 sweeps) and Juno's 1% to 7% higher. Local paths are replaced by `<repo>` and `<home>`.

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Alloc B/tok in request | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|------------------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 75.316174 | 27.747479 | 32.295220566197344 | 11.678186080812964 | 10.94 / 11.91 | 0.4287952885949483 | 0.4208737694985899 | 128/128 | 64/64 | 0 | 136365946 | 311005 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 27.203233 | 12.636817 | 12.286786492878084 | 4.208724098596993 | 4.19 / 4.32 | 0.4516664064480161 | 0.3330525478525955 | 128/128 | 64/64 | 5 | 219961285 | 1287964 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 19.222076 | 10.087790 | 9.970526003449628 | 3.034077744813703 | 3.03 / 3.06 | 0.518701830304366 | 0.30076733802088496 | 128/128 | 64/64 | 0 | 251383794 | 396044 | yes | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 11.668904 | 6.184489 | 5.306051864192559 | 2.1129650672953817 | 2.1 / 2.11 | 0.45471724372679384 | 0.3416555623747381 | 128/128 | 64/64 | 0 | 294315653 | 348677 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |

GPU attention per row, as the engine resolved it (read from its log, not from the flag):
- tinyllama-1.1b-chat-v1.0.Q4_K_M: n/a (requested: engine default)
- qwen2.5-3b-instruct-q4_k_m: n/a (requested: engine default)
- Phi-3.5-mini-instruct-Q4_K_M: n/a (requested: engine default)
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
  GPU clocks {"graphics_mhz": 139, "sm_mhz": 139, "memory_mhz": 405, "active_throttle_reasons": "Idle"}. A throttled run and a regression look the same
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
