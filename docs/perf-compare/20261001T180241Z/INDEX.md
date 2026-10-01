# llama.cpp vs Juno - 20261001T180241Z (cpu)

> **CPU reference from 2026-10-01** (supersedes [`20260927T094414Z`](../20260927T094414Z/INDEX.md) as reference). The first CPU sweep with pinned clocks (governor performance, turbo off) and with Juno's common-pool parallelism set from `--threads` (12 threads, matching the reference tool's `-t 12`). At this default the pool size equals the JVM's own default, so thread count did not change; the clock pinning did, and it lowers both engines' absolute throughput. Compare ratios across this boundary, never absolute t/s. Every row scorable, every prefill 128 of 128 tokens.

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 59.131602 | 25.888680 | 6.108500223713679 | 3.1397535034852755 | 3.02 / 3.26 | 0.10330347930897726 | 0.12127901088372506 | 128/128 | 64/64 | 10 | 65153068 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| qwen2.5-3b-instruct-q4_k_m | 26.153034 | 10.944721 | 1.992856166215052 | 1.083258688978605 | 1.08 / 1.09 | 0.07619980787755072 | 0.09897545026306337 | 128/128 | 64/64 | 7 | 183314471 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 18.526862 | 8.933504 | 0.9058059950870286 | 0.8975008617507123 | 0.9 / 0.9 | 0.048891495769063785 | 0.10046459505147279 | 128/128 | 64/64 | 9 | 199531445 | yes | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 11.436533 | 5.565919 | 0.8569259356077699 | 0.5096100932449008 | 0.51 / 0.51 | 0.0749288211390436 | 0.09155902075558427 | 128/128 | 64/64 | 14 | 273132702 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |

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
- Build: Juno `807bfea458e8` plus uncommitted changes, jar sha256 `30d4d4937d1e3994`,
  OpenJDK Runtime Environment (build 25.0.3+9-2-24.04.2-Ubuntu); JVM -XX:+UseG1GC -XX:+AlwaysPreTouch -XX:+UnlockExperimentalVMOptions -XX:-UseFastUnorderedTimeStamps -Djava.util.concurrent.ForkJoinPool.common.parallelism=11, -Xms equal to -Xmx per model.
  Reference tool build `379ac6673` from /home/medion/Repo/llama.cpp-bin/llama-b9551; GPU driver 580.173.02.
  A different reference build is a measurement boundary: its ratios are not comparable with this run.
- Clocks pinned: cpu governor performance (was schedutil), turbo off.
- Clock state this run: CPU governor performance, turbo disabled,
  GPU clocks {"graphics_mhz": 151, "sm_mhz": 151, "memory_mhz": 405, "active_throttle_reasons": "Idle"}. A throttled run and a regression look the same
  without this.
- Thread counts are matched: the reference tool ran with -t 12, and Juno's CPU kernels
  ran on 12 threads (common pool parallelism 11 plus the calling thread).
- Repetitions whose readings were withheld (the median is taken over the rest):
  - Phi-3.5-mini-instruct-Q4_K_M-generate-rep2: the recorded spans leave 40 ms (forward passes) and -61 ms (token span) of the 84121 ms request unaccounted for (allowed -25 to 492): JFR timestamps disagree with the engine clock, readings withheld
  - mistral-7b-instruct-v0.1-q4_k_m-generate-rep2: the recorded spans leave 39 ms (forward passes) and -31 ms (token span) of the 137439 ms request unaccounted for (allowed -25 to 492): JFR timestamps disagree with the engine clock, readings withheld
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
- Backend: cpu (llama -ngl 0 / juno --cpu), temperature 0, max_tokens=64.
- Juno ran with jdk.incubator.vector. On CPUs without HW FMA this can be pathologically slow; re-run with --vector 0.
