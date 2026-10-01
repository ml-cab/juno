# Device kernel span: same-hour A/B (2026-10-01, GPU)

Gate: prefill with `juno.DeviceCompute` in the build, its recording **off** (the default
`juno-perf.jfc`), **>= 0.98x** prefill of the build without it. Juno absolute t/s, same-hour
interleaved A/B (A B A B A B), pinned clocks, median of three. Models: TinyLlama 1.1B Q4_K_M and
Mistral 7B Q4_K_M, `n_prompt` 512, GTX 1080. Each invocation: `compare-llama-cpp.sh --gpu
--pin-clocks --models tinyllama-1.1b-chat-v1.0.Q4_K_M,mistral-7b-instruct-v0.1-q4_k_m --n-prompt 512
--juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --juno-jar <jar>`.

- A (baseline): the shaded jar of the tree before this change (commit `807bfea` plus uncommitted
  plan documents), sha256 `fc01f42184810aa6`.
- B (candidate): the shaded jar with `juno.DeviceCompute`, the host packing site and the Phi-3
  RoPE factor fix, sha256 `30d4d4937d1e3994`. The RoPE change touches only Phi-3 rotation, which
  neither model runs.

Both jars were run from the same checkout, so each run's metadata names the same commit; the jar
hash is what tells the sides apart. Every reading is in `ab-readings.json`. All six invocations
recorded `clock_pinned` (governor performance, turbo off, GPU graphics clock locked at 1911 MHz),
every prefill was 512 of 512 tokens, every row's repetition was scorable, and none was withheld.
Run between 17:23 and 17:34 UTC.

## Result

| Model | Prefill A t/s | Prefill B t/s | B/A | Generation A t/s | Generation B t/s | B/A |
|---|---|---|---|---|---|---|
| TinyLlama | 252.76 (249.60 / 254.69) | 251.36 (247.45 / 259.28) | **0.994** | 68.18 (64.06 / 68.26) | 68.94 (67.56 / 69.63) | 1.011 |
| Mistral 7B | 67.13 (66.97 / 68.05) | 68.23 (65.31 / 68.60) | **1.016** | 22.45 (21.62 / 22.68) | 22.35 (21.56 / 22.51) | 0.996 |

Median (min / max) of three passes each. **Gate met** on both models. With the event off, every new
kernel site adds one enabled check before the plain call, and the host packing one more; this A/B
measures the cost a throughput sweep carries, which is within the spread of either side.

Not measured here: the cost with `--device-spans` on. That run times each prefill-width kernel
between two stream events (or, for the attention kernel and the FP32 GEMM, between two drains of
the default stream), and is read for attribution, not throughput.
