# Device copy and dequantization spans: same-hour A/B (2026-09-29 to 2026-09-30, GPU)

Gate: prefill with the new spans **>= 0.98x** prefill without them, Juno absolute t/s, same-hour
interleaved A/B (A B A B A B), pinned clocks, median of three. Models: TinyLlama 1.1B Q4_K_M and
Mistral 7B Q4_K_M, `n_prompt` 512, GTX 1080. Each invocation: `compare-llama-cpp.sh --gpu
--pin-clocks --models tinyllama,mistral-7b --n-prompt 512 --juno-jar <jar> --juno-reps 1
--juno-warmup 2 --reps 1 --no-publish --no-tuned-lane`. A (baseline) is the shaded jar of commit
`38b1c6d` (sha256 `9a0a3f3bf695987d`); B (candidate) is the working tree at each stage. Every
reading, per pass, is in `ab-readings.json`. The harness names the checkout's commit in each run's
metadata for both sides; the jar hash is what tells them apart.

## Result (scored: run 4)

| Model | Prefill A t/s | Prefill B t/s | B/A | Generation A t/s | Generation B t/s | B/A |
|---|---|---|---|---|---|---|
| TinyLlama | 245.18 (234.99 / 252.74) | 244.95 (238.27 / 248.02) | **0.999** | 65.31 (64.20 / 66.30) | 65.30 (63.86 / 66.26) | 1.000 |
| Mistral 7B | 66.07 (64.70 / 66.55) | 66.04 (65.83 / 66.55) | **1.000** | 21.77 (21.73 / 21.91) | 21.73 (21.40 / 21.94) | 0.998 |

Median (min / max) of three passes each; all six invocations recorded `clock_pinned` (governor
performance, turbo off, GPU graphics clock locked at 1911 MHz). **Gate met.** In the scored design
the two events are off in `juno-perf.jfc` and are turned on for a breakdown run with
`juno-perf-spans.jfc` (`--device-spans`); this A/B therefore measures the cost a throughput sweep
carries, which is none.

## How it got there: four runs

| Run | Candidate | Harness | TinyLlama prefill B/A | Mistral prefill B/A | TinyLlama gen B/A | Mistral gen B/A |
|---|---|---|---|---|---|---|
| 1 `20260929T045606Z` | one JFR event per copy | warmup unrecorded | 0.802 | 0.948 | 0.818 | 0.902 |
| 2 `20260929T053002Z` | totals per site and phase; decode counted, not timed | warmup unrecorded | 0.896 | 0.975 | 1.016 | 1.000 |
| 3 `20260930T022300Z` | same as run 2 | last warmup under a discarded recording | 0.933 | 1.006 | 1.026 | 0.984 |
| 4 `20260930T030253Z` | spans opt-in (off in `juno-perf.jfc`), small copies timed 1 in 16 | as run 3 | **0.999** | **1.000** | 1.000 | 0.998 |

- **Run 1.** A TinyLlama prefill window issues 22,726 host-to-device copies (every KV-mirror row of
  every layer is its own copy) and a decoded token several hundred; one event per copy, at a few
  microseconds each on this host's operating-system JFR clock, cost a fifth of both.
- **Run 2.** Per-site totals committed at the end of the recording chunk removed the decode cost.
  Prefill still missed, and not in the copies: the per-layer `juno.SwiGlu` spans read 251 and 161 ms
  for the first two of 22 layers, then 35 ms like the baseline. `jdk.Deoptimization` named the
  cause: the harness warms the engine up with no recording, so the recording-only branches in
  `DeviceStaging.copy` and `DeviceSpanTally.staging` compiled as uncommon traps; the measured
  request's recording tripped them and deoptimized the batched-layer method they are inlined into.
  The baseline pays a smaller share of the same effect from JFR instrumenting its own event classes
  (85 and 42 ms for the first two layers).
- **Run 3.** The harness now runs the last warmup under a discarded recording with the measurement
  settings, which moves that recompilation out of the measured request for both builds (every layer
  flat, no Juno method deoptimized inside the window). The baseline rose with it (TinyLlama prefill
  about 240 to 248 t/s). The remaining TinyLlama gap was the counting and timing work itself: the
  same candidate jar with the two events disabled in the settings matched the baseline in three
  interleaved unpinned rounds (248.7 against 246.5 t/s, median).
- **Run 4.** Owner decision: the spans are opt-in. Cost of turning them on, from an unpinned probe on
  the final build: 241.5 against 258.0 t/s TinyLlama prefill (about 6%); read throughput from a run
  without `--device-spans` and staged bytes and the per-term breakdown from a run with it.

## What a spans run reports (final build, TinyLlama, one 511-token prefill window)

H2D 506 MB (22,726 copies; 11,242 each of K and V rows, the rest matmul activations and attention
tables), D2H 897 MB (176 copies), device dequantization 39 ms (154 Q4_K/Q6_K weight matrices);
estimated H2D 195 ms and D2H 99 ms. Unpinned, indicative; the step-2 re-baseline takes the
reference figures.

See `docs/performance.md` ("Measurement boundary") for the harness change and how to read the
figures.
