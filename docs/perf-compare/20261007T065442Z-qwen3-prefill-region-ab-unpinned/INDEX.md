# Qwen3 prefill region A/B (unpinned) - 20261007T065442Z (gpu)

**Purpose.** Same-session A/B of Qwen3-1.7B's prefill and generation before and after its per-head Q/K norm,
RoPE and attention moved into the prefill-window device region.

**Pinned: no.** Clocks not pinned on any of the six runs; indicative, not scorable against a 0.95x bound. The
pinned form is `bash dist/qwen3-prefill-region-ab/run-gate.sh` (owner, if wanted; this change is also read by the
tier's closing no-regression gate).

**Command.** `PIN=0 bash dist/qwen3-prefill-region-ab/run-gate.sh` (copied here as `run-gate.sh`): A = baseline jar
(HEAD `d05d1fe`, sha256 `652bf47e12609973`), B = candidate (this working tree, `533b1864d682bea5`), A B A B A B,
each run `compare-llama-cpp.sh --gpu --models Qwen3-1.7B-Q4_K_M --n-prompt 512 --juno-warmup 2 --juno-reps 1
--reps 1 --no-tuned-lane --no-publish --juno-jar <jar>`, default flags otherwise. `console.txt` is the script's
output; each `baseline-N`/`candidate-N` directory holds one run's JSON (`RUN-INDEX.md`: the harness's index).

## Reading

| Metric | Baseline (3 runs) | Candidate (3 runs) | B/A (medians) |
|---|---|---|---|
| pp t/s at 512 | 849.70, 848.80, 849.10 (median 849.10) | 1720.89, 1687.69, 1690.02 (median 1690.02) | **1.990** |
| tg t/s | 37.67, 37.27, 36.93 (median 37.27) | 37.45, 36.79, 37.16 (median 37.16) | **0.997** |

Every candidate prefill reading is above every baseline one. Generation does not run through the prefill region.
Greedy text is identical on all six runs, in both the prefill and the generation request.
