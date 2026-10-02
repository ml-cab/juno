# Prefill window spans: same-hour A/B (2026-10-01, GPU)

Gate: prefill with `juno.WindowStep`, the per-op events on the Phi-3 and Qwen3 window paths and the
`juno.MatVec` window width in the build, device spans **off** (the default `juno-perf.jfc`, which
enables `juno.WindowStep` like the other per-op events), **>= 0.98x** prefill of the build without
them. Juno absolute t/s, same-hour interleaved A/B (A B A B A B), pinned clocks, median of three.
Models: TinyLlama 1.1B Q4_K_M and Phi-3.5-mini Q4_K_M (the handler whose window path gained the most
spans), `n_prompt` 512, GTX 1080. Each invocation: `compare-llama-cpp.sh --gpu --pin-clocks --models
tinyllama-1.1b-chat-v1.0.Q4_K_M,Phi-3.5-mini-instruct-Q4_K_M --n-prompt 512 --juno-warmup 2
--juno-reps 1 --reps 1 --no-tuned-lane --no-publish --juno-jar <jar>`.

- A (baseline): the shaded jar of commit `d47957f`, before the spans, sha256 `2c163485a0071c66`.
- B (candidate): the shaded jar with the spans (the change committed as `c42de6e`), sha256
  `513f57643c8fe97c`.

Both jars were run from the same checkout, so each run's metadata names the same commit; the jar
hash is what tells the sides apart. Every reading is in `ab-readings.json`. All six invocations
recorded `clock_pinned` (governor performance, turbo off, GPU graphics clock locked at 1911 MHz),
every prefill was 512 of 512 tokens, every row was scorable, and the largest GC pause in any run was
9 ms. Run between 23:59 and 00:09 UTC.

## Result

| Model | Prefill A t/s | Prefill B t/s | B/A | Generation A t/s | Generation B t/s | B/A |
|---|---|---|---|---|---|---|
| TinyLlama | 251.67 (245.36 / 253.76) | 249.88 (248.12 / 253.47) | **0.993** | 68.25 (67.64 / 69.05) | 68.20 (66.31 / 69.11) | 0.999 |
| Phi-3.5-mini | 89.13 (88.85 / 89.78) | 89.58 (88.38 / 89.86) | **1.005** | 32.66 (32.31 / 32.68) | 32.25 (31.72 / 32.31) | 0.987 |

Median (min / max) of three passes each. **Gate met** on both models, so `juno.WindowStep` stays
enabled in the base `juno-perf.jfc`, and the fallback (moving it to `juno-perf-spans.jfc`) is not
needed. The unpinned informational A/B taken when the spans landed read 0.974x on TinyLlama with a 5%
spread on the candidate side; pinned, both sides agree to within 3%.

Generation is not gated by this A/B: the spans are on the window path only, and the decode path's code
is unchanged. Phi-3.5-mini's 0.987 (1.3% lower; the candidate median sits just below the baseline
minimum) is recorded, not attributed. It is well inside the 0.95x decode no-regression gate that the
tier's closing A/B applies.
