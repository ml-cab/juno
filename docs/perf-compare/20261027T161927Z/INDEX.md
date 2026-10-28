# llama.cpp vs Juno - 20261027T161927Z (cpu)

> **CPU reference at `n_prompt=512` from 2026-10-27** (supersedes [`20261012T045931Z`](../20261012T045931Z/INDEX.md) as reference). The closing sweep of the CPU decode and prefill kernel work, owner run, pinned clocks, jar `4e480ab9131887eb` (`c2633a68` plus the in-recording allocation field): the vector Q6_K decode kernel, the split Q4_K kernel, pooled vector prefill attention and the padded prefill tile, `--threads 12` against `-t 12`. A measurement boundary for CPU readings; compare ratios across it. Gate results: [`20261027T154748Z-cpu-kernels-close-gate-2`](../20261027T154748Z-cpu-kernels-close-gate-2/INDEX.md). Local paths are replaced by `<repo>` and `<home>`. pp ratio 0.422x (Qwen2.5-3B, binding) to 0.512x.

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Alloc B/tok in request | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|------------------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 79.334436 | 42.026554 | 33.99688394535062 | 12.281795468809456 | 12.25 / 12.34 | 0.4285261944176501 | 0.2922389370494059 | 512/512 | 64/64 | 0 | 156947512 | 309316 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 28.178681 | 15.397339 | 11.901184468150289 | 4.307132418662267 | 4.29 / 4.33 | 0.42234710943888 | 0.279732258844354 | 512/512 | 64/64 | 0 | 247626035 | 1290007 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 19.527194 | 12.129070 | 9.999600221451677 | 3.106513410966319 | 3.1 / 3.12 | 0.5120858747781005 | 0.2561213193564155 | 512/512 | 64/64 | 3 | 268787787 | 348956 | yes | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 11.858136 | 7.097583 | 5.171878777779029 | 2.181141565406965 | 2.14 / 2.19 | 0.4361460163535845 | 0.30730765183118885 | 512/512 | 64/64 | 0 | 260269837 | 348151 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |

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
