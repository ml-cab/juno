# llama.cpp vs Juno - 20260930T141026Z (gpu)

> **GPU reference from 2026-09-30 at `n_prompt=512`**, the first parity-corrected 512-token sweep (Tier 01B step 2). Same build and method as [`20260930T135225Z`](../20260930T135225Z/INDEX.md) (`n_prompt=128`). Staged bytes for the same shape are in [`20260930T144658Z`](../20260930T144658Z/INDEX.md).

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3659.567667 | 173.571413 | 242.34785109978196 | 65.26413370302487 | 63.61 / 66.96 | 0.06622308238351314 | 0.37600738839998304 | 512/512 | 64/64 | 7 | 49824602 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned | 3659.567667 | 173.571413 | 245.468090446538 | 65.21660766389398 | 64.57 / 65.3 | 0.06707570751048994 | 0.37573357580429434 | 512/512 | 64/64 | 7 | 49840996 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1438.588262 | 65.062352 | 91.74318771836182 | 29.136394021566684 | 27.89 / 30.01 | 0.06377306846005798 | 0.4478226366849861 | 512/512 | 64/64 | 11 | 145005537 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |
| qwen2.5-3b-instruct-q4_k_m-tuned | 1438.588262 | 65.062352 | 95.65714558108684 | 30.286071064014173 | 30.18 / 30.39 | 0.06649376204981633 | 0.4654930252754184 | 512/512 | 64/64 | 11 | 145020778 | yes | qwen2.5-3b-instruct-q4_k_m-tuned-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1170.423079 | 59.128229 | 13.058197839714241 | 23.75892779392031 | 23.43 / 23.79 | 0.011156818482143277 | 0.4018203858925034 | 512/512 | 64/64 | 11 | 213261402 | yes | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| Phi-3.5-mini-instruct-Q4_K_M-tuned | 1170.423079 | 59.128229 | 13.105832176882041 | 24.231946036879805 | 23.99 / 24.29 | 0.011197516874051697 | 0.40982025754364815 | 512/512 | 64/64 | 9 | 213264721 | yes | Phi-3.5-mini-instruct-Q4_K_M-tuned-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 694.892970 | 37.076586 | 68.34549557757082 | 22.35689948607322 | 22.3 / 22.5 | 0.09835398907197293 | 0.6029923975760125 | 512/512 | 64/64 | 28 | 226240856 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 694.892970 | 37.076586 | 67.44299579914743 | 22.125450174089128 | 22.03 / 22.41 | 0.09705522823053948 | 0.5967499319945242 | 512/512 | 64/64 | 27 | 225888952 | yes | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

GPU attention per row, as the engine resolved it (corrected 2026-09-30, see below):
- tinyllama-1.1b-chat-v1.0.Q4_K_M: unknown (requested: engine default)
- tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned: unknown (requested: auto)
- qwen2.5-3b-instruct-q4_k_m: unknown (requested: engine default)
- qwen2.5-3b-instruct-q4_k_m-tuned: unknown (requested: auto)
- Phi-3.5-mini-instruct-Q4_K_M: unknown (requested: engine default)
- Phi-3.5-mini-instruct-Q4_K_M-tuned: unknown (requested: auto)
- mistral-7b-instruct-v0.1-q4_k_m: unknown (requested: engine default)
- mistral-7b-instruct-v0.1-q4_k_m-tuned: unknown (requested: auto)

Correction (2026-09-30): this list first read `off` on every row. That was wrong. The harness
read it from the engine log, but the console turns library logging off unless `--verbose`, so the
activation line never reaches the log and a silent log proves nothing. The values are now
re-derived from this run's own recordings: `on`/`off` where the run carried `--device-spans`
(the attention kernel's `memcpy_gqa_*` copy sites present or absent), `unknown` otherwise.

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
- Build: Juno `ffd0ca7af4fa` plus uncommitted changes, jar sha256 `dc94bd797c8a3219` (juno-player-0.1.2-shaded.jar; filled in after the run, the harness recorded `missing`),
  OpenJDK Runtime Environment (build 25.0.3+9-2-24.04.2-Ubuntu); JVM -XX:+UseG1GC -XX:+AlwaysPreTouch -XX:+UnlockExperimentalVMOptions -XX:-UseFastUnorderedTimeStamps, -Xms equal to -Xmx per model.
  Reference tool build `ac4cdde` from /home/medion/Repo/llama.cpp/build-cuda/bin; GPU driver 580.173.02.
  A different reference build is a measurement boundary: its ratios are not comparable with this run.
- Clocks pinned: cpu governor performance (was schedutil), turbo off, gpu graphics clock locked at 1911 MHz.
- Clock state this run: CPU governor performance, turbo disabled,
  GPU clocks {"graphics_mhz": 1771, "sm_mhz": 1771, "memory_mhz": 5005, "active_throttle_reasons": "Idle"}. A throttled run and a regression look the same
  without this.
- Thread counts are not matched: the reference tool ran with -t 12, while Juno
  dispatches its kernels on the common pool at an effective parallelism of
  11. Juno has no thread-count control reaching the hot path yet, so this
  mismatch is recorded rather than removed.
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
