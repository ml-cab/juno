# llama.cpp vs Juno - 20260930T135225Z (gpu)

> **Superseded as the GPU reference on 2026-10-03** by [`20261003T042440Z`](../20261003T042440Z/INDEX.md), the pinned closing sweep of the prefill-throughput work. Kept as the before-reading of that work.

> **GPU reference from 2026-09-30 at `n_prompt=128`** (supersedes [`20260927T232837Z`](../20260927T232837Z/INDEX.md) + [`20260927T234659Z`](../20260927T234659Z/INDEX.md) as reference). Tier 01B step 2 re-baseline: pinned clocks, operating-system JFR clock, warm-up recording, fixed heap. A measurement boundary: compare ratios across it, not absolute t/s. Its 512-token pair is [`20260930T141026Z`](../20260930T141026Z/INDEX.md).

> **Mislabelled lanes in earlier runs.** From commit `8776a3f` (2026-09-17) the engine defaults `--gpu-attention` to `auto`, which is on under CUDA for Llama-family handlers. Every GPU run of this harness since then, including its default lanes, ran the attention kernel on `tinyllama`, `qwen2.5-3b` and `mistral-7b`, while the harness help and the tuned-lane comment described the default lane as `--gpu-attention off`. `Phi-3.5-mini` ran scalar attention on every lane, because its handler has no kernel. The result data recorded the flag as passed (empty, not `off`), so only the description was wrong, and no published figure changes. Runs before `8776a3f` were described correctly. From this run on, each result records the resolved value; see the list below the table.

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3535.584706 | 182.421218 | 239.07967176901286 | 68.75879245000783 | 67.12 / 68.97 | 0.0676209712535771 | 0.376923217616099 | 128/128 | 64/64 | 8 | 49842206 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned | 3535.584706 | 182.421218 | 238.13327076034366 | 63.30082429166283 | 60.64 / 68.25 | 0.06735329247131992 | 0.34700362702140725 | 128/128 | 64/64 | 8 | 49840953 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1426.082454 | 69.052837 | 95.24045584082215 | 29.23328726526586 | 28.93 / 29.55 | 0.06678467684227055 | 0.4233466506997513 | 128/128 | 64/64 | 0 | 144439190 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |
| qwen2.5-3b-instruct-q4_k_m-tuned | 1426.082454 | 69.052837 | 92.17893887092528 | 28.723868758855577 | 28.18 / 29.04 | 0.06463787462805799 | 0.415969422934145 | 128/128 | 64/64 | 0 | 144307823 | yes | qwen2.5-3b-instruct-q4_k_m-tuned-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1158.766116 | 57.831504 | 43.067756206472225 | 22.803109675316 | 22.4 / 22.97 | 0.037166910226146294 | 0.3943025530741168 | 128/128 | 64/64 | 10 | 213276751 | yes | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| Phi-3.5-mini-instruct-Q4_K_M-tuned | 1158.766116 | 57.831504 | 42.73697378933685 | 22.901287706487135 | 22.8 / 22.94 | 0.03688144932720561 | 0.396000209617359 | 128/128 | 64/64 | 10 | 210852185 | yes | Phi-3.5-mini-instruct-Q4_K_M-tuned-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 633.990725 | 35.198321 | 64.28386711103391 | 21.462337851643497 | 21.43 / 21.86 | 0.10139559551290583 | 0.6097545917500865 | 128/128 | 64/64 | 17 | 224369059 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 633.990725 | 35.198321 | 64.0368359731937 | 21.454629757343653 | 21.33 / 21.57 | 0.1010059507939863 | 0.6095356013527933 | 128/128 | 64/64 | 19 | 226242265 | yes | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

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
  GPU clocks {"graphics_mhz": 1278, "sm_mhz": 1278, "memory_mhz": 5005, "active_throttle_reasons": "Idle"}. A throttled run and a regression look the same
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
