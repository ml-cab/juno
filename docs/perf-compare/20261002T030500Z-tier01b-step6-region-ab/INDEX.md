# Prefill-window device region: same-hour A/B (2026-10-02, GPU)

Gate: the build with the prefill-window device region (the layer's norms, matmuls, SwiGLU, residual
adds, and on the LLaMA family RoPE and attention on the device) against the build without it. Prefill
**>= 1.25x** on TinyLlama (gated; Mistral 7B recorded); generation **>= 0.95x** on both (no-regression
gate). Juno absolute t/s, same-hour interleaved A/B (A B A B A B), pinned clocks, median of three.
Models: TinyLlama 1.1B Q4_K_M and Mistral 7B Q4_K_M, `n_prompt` 512, GTX 1080. Each invocation:
`compare-llama-cpp.sh --gpu --pin-clocks --models
tinyllama-1.1b-chat-v1.0.Q4_K_M,mistral-7b-instruct-v0.1-q4_k_m --n-prompt 512 --juno-warmup 2
--juno-reps 1 --reps 1 --no-tuned-lane --no-publish --juno-jar <jar>`, driven by
`target/tier01b-item6-ab/run-gate.sh`.

- A (baseline): the shaded jar of commit `c42de6e`, before the region, sha256 `92643aec3c3b226a`.
- B (candidate): the shaded jar with the region (uncommitted at the time of the run), sha256
  `fb6d32e6c41cb3dd`.

Both jars were run from the same checkout, so the jar hash is what tells the sides apart; each run's
`host.json` names the jar it launched. Every reading is in `ab-readings.json`. All six invocations
recorded `clock_pinned` (governor performance, turbo off, GPU graphics clock locked at 1911 MHz), every
prefill was 512 of 512 tokens, every row was scorable, no repetition was withheld, and the largest GC
pause was 14.7 ms (Mistral 7B, 13.3 to 14.7 ms on both sides alike). Run between 03:05 and 03:16 UTC,
taken by the owner.

## Result

| Model | Prefill A t/s | Prefill B t/s | B/A | Generation A t/s | Generation B t/s | B/A |
|---|---|---|---|---|---|---|
| TinyLlama | 240.00 (234.29 / 253.18) | 935.81 (924.60 / 960.03) | **3.899** | 66.96 (63.09 / 68.91) | 68.48 (65.34 / 69.22) | **1.023** |
| Mistral 7B | 68.35 (68.10 / 68.46) | 191.76 (190.52 / 197.34) | **2.806** | 22.83 (22.57 / 22.84) | 22.89 (22.24 / 23.00) | **1.003** |

Median (min / max) of three passes each. **Gate met**: TinyLlama prefill 3.899x against `>= 1.25x`,
generation 1.023x and 1.003x against `>= 0.95x`. Every candidate prefill reading is above every baseline
reading on both models, by far more than the noise floor. The unpinned informational A/B taken when the
region landed read 3.60x and 2.73x; pinned, the gain is slightly larger. Generation is unaffected, as
expected: the region runs on prefill windows only, and the decode path's code is unchanged.
