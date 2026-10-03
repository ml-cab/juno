# Decode no-regression gate after the prefill-throughput work (2026-10-03, pinned)

Purpose: the closing decode gate for the prefill-throughput work. Gate: Juno generation t/s **>= 0.95x** the
build the 2026-09-30 reference sweeps measured, on every sweep model, from a same-hour interleaved A/B with
pinned clocks, median of three; prefill recorded, not gated. Owner run, 2026-10-02 23:02 to 23:24 -0500
(`dist/tier01b-close/run-gate.sh`, part A).

Each invocation: `compare-llama-cpp.sh --gpu --pin-clocks --models <four sweep models> --n-prompt 512
--juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish --juno-jar <jar>`, alternating
baseline, candidate, three times. All six runs pinned (`clock_pinned: true`), every row scorable, every
prefill 512 of 512, GC pause at most 30 ms. `ab-readings.json` holds every reading.

Builds: A = commit `ffd0ca7`, rebuilt from that commit (jar `bd19a9306ea7901f`); B = HEAD `1ac490a`
(jar `66f02ee7c2908f78`).

| Model | Prefill A | Prefill B | B/A | Generation A (min / max) | Generation B (min / max) | B/A |
|---|---|---|---|---|---|---|
| TinyLlama | 249.66 | 1016.66 | 4.07x | 67.61 (66.54 / 69.23) | 68.53 (62.49 / 68.75) | **1.014** |
| Qwen2.5-3B | 94.39 | 431.95 | 4.58x | 30.34 (29.89 / 30.53) | 30.16 (29.79 / 30.23) | **0.994** |
| Phi-3.5-mini | 12.53 | 174.27 | 13.91x | 23.50 (23.09 / 23.73) | 31.98 (31.91 / 32.05) | **1.361** |
| Mistral 7B | 67.68 | 200.09 | 2.96x | 22.58 (22.25 / 23.00) | 22.69 (21.74 / 22.82) | **1.005** |

**Gate met** on all four models. Phi-3.5-mini's generation rose because its decode path now runs the GPU
attention kernel; on the other three, generation is flat. Prefill moved 2.96x to 13.91x across the whole
body of prefill work measured here (GPU attention on Phi-3, the prefill-window device region, the residual
kept on the device).
