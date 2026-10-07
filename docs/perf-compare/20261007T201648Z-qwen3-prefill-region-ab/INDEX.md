# Qwen3 prefill region A/B (pinned) - 20261007T201648Z (gpu)

**Purpose.** Same-session pinned A/B of Qwen3-1.7B's prefill and generation before and after its per-head Q/K norm,
RoPE and attention moved into the prefill-window device region. The pinned form of
`20261007T065442Z-qwen3-prefill-region-ab-unpinned`. Bound: pp and tg B/A `>= 0.95` (the tier's no-regression bound,
read for the model the change touches).

**Pinned: yes**, on all six runs (`clock_pinned: true`); runs started 20:16Z to 20:20Z; owner run.

**Command.** `bash dist/qwen3-prefill-region-ab/run-gate.sh` (copied here as `run-gate.sh`): A = baseline jar (HEAD
`d05d1fe`, sha256 `652bf47e12609973`), B = candidate (`533b1864d682bea5`, `d05d1fe` plus the Qwen3 prefill change),
A B A B A B, each run `compare-llama-cpp.sh --gpu --pin-clocks --models Qwen3-1.7B-Q4_K_M --n-prompt 512
--juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish --juno-jar <jar>`. Re-scored from the run files
with the script's method (medians of three), since its console output was not kept. Each `baseline-N`/`candidate-N`
directory holds one run's JSON (`RUN-INDEX.md`: the harness's index); local paths made repository-relative.

## Reading (every row scorable; prompt tokens 512 of 512 on all six runs)

| Metric | Baseline (3 runs) | Candidate (3 runs) | B/A (medians) | Bound |
|---|---|---|---|---|
| pp t/s at 512 | 815.96, 813.44, 835.72 (median 815.96) | 1656.62, 1721.50, 1725.45 (median 1721.50) | **2.110** | >= 0.95, met |
| tg t/s | 36.59, 37.83, 36.71 (median 36.71) | 36.21, 36.02, 36.43 (median 36.21) | **0.986** | >= 0.95, met |
| GPU pp ratio at 512 | 0.286x, 0.293x, 0.296x | 0.585x, 0.608x, 0.613x | | reading |
| Allocated bytes per generated token | 111.1M to 111.9M | 111.5M to 111.8M | | reading |

Every candidate prefill reading is above every baseline one; the unpinned reading (1.990, 0.997) agrees. Generation does
not run through the prefill region; 0.986 sits inside the baseline's own spread (36.59 to 37.83). Greedy text is
identical on all six runs, in both the prefill and the generation request.
