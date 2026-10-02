# Residual stream across layers and the non-allocating matmul: same-hour A/B (2026-10-02, GPU)

Gate: the build that keeps the prefill-window residual on the device across layers (one upload and one
download per window instead of one of each per layer) and writes batched matmuls into the caller's rows
(`MatVec.sgemmInto`), against the build before it. Generation **>= 0.95x** on both models (no-regression
gate); prefill recorded, not gated. Juno absolute t/s, same-hour interleaved A/B (A B A B A B), pinned
clocks, median of three. Models: TinyLlama 1.1B Q4_K_M and Mistral 7B Q4_K_M, `n_prompt` 512, GTX 1080.
Each invocation: `compare-llama-cpp.sh --gpu --pin-clocks --models
tinyllama-1.1b-chat-v1.0.Q4_K_M,mistral-7b-instruct-v0.1-q4_k_m --n-prompt 512 --juno-warmup 2
--juno-reps 1 --reps 1 --no-tuned-lane --no-publish --juno-jar <jar>`, driven by
`dist/tier01b-item2-ab/run-gate.sh`.

- A (baseline): the shaded jar of commit `61c31cc` (the prefill-window device region), sha256
  `d287e929f78c11ff`.
- B (candidate): the shaded jar with this change (uncommitted at the time of the run), sha256
  `ba4ee78b6a77541e`. It was rebuilt from the same sources as `caaf5810752a1ecb`, the jar the
  `20261002T050741Z` breakdown measured, after a `mvn clean` removed that one; jar builds here carry entry
  timestamps, so the hash differs while the sources do not.

Both jars were run from the same checkout, so the jar hash is what tells the sides apart; each run's
`host.json` names the jar it launched. Every reading is in `ab-readings.json`. All six invocations
recorded `clock_pinned` (governor performance, turbo off, GPU graphics clock locked at 1860 MHz), every
prefill was 512 of 512 tokens, every row was scorable, no repetition was withheld, and the largest GC
pause was 16 ms (Mistral 7B; 13 to 16 ms on both sides alike). Run between 13:52 and 14:00 UTC, taken by
the owner.

## Result

| Model | Prefill A t/s | Prefill B t/s | B/A | Generation A t/s | Generation B t/s | B/A |
|---|---|---|---|---|---|---|
| TinyLlama | 959.54 (941.69 / 965.94) | 1029.42 (983.10 / 1030.55) | **1.073** | 67.91 (67.67 / 68.62) | 67.73 (66.98 / 68.22) | **0.997** |
| Mistral 7B | 194.63 (191.17 / 196.39) | 204.18 (200.47 / 207.49) | **1.049** | 22.89 (22.47 / 22.94) | 22.73 (22.69 / 22.88) | **0.993** |

Median (min / max) of three passes each. **Gate met**: generation 0.997x and 0.993x against `>= 0.95x`.
Prefill rose 1.073x and 1.049x; every candidate prefill reading is above every baseline reading on both
models. The gain is the size the breakdowns predict: matmul and region staging plus the host side of the
region were 11.1% (TinyLlama) and 6.5% (Mistral 7B) of a 512-token window before the change
(`20261002T014419Z`) and 3.5% and 2.1% after (`20261002T050741Z`); removing 7.6 and 4.4 points of a window
is worth about 1.08x and 1.05x. The unpinned informational A/B taken when the change landed read 1.045x and 1.047x.
Generation is not expected to move: the change touches prefill windows and the batched matmul's output
handling, not the decode path.
