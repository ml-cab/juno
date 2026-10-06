# Phi-3 prefill region attention: pinned A/B - 20261006T192743Z (gpu, clocks pinned, owner run)

**Purpose.** The pinned form of the same-session A/B of the change that moved Phi-3's prefill-window
RoPE and attention inside the prefill-window device region, on the model it touches (Phi-3.5-mini), at
`n_prompt=512`: is prefill faster, and do prefill and generation stay at or above 0.95x the pre-change
build (the tier's no-regression bound, read here for this change)?

**Pinned: yes**, on all six runs (`clock_pinned: true` in every `host.json`; CPU governor performance,
turbo off; the GPU clock is recorded, not locked, as on every run on this card). Owner run.

**Command.** `bash dist/phi3-prefill-region-ab/run-gate.sh` (copied here as `run-gate.sh`): baseline and
candidate alternated A B A B A B from 19:27Z to 19:31Z, each run `compare-llama-cpp.sh --gpu --pin-clocks
--models Phi-3.5-mini-instruct-Q4_K_M --n-prompt 512 --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane
--no-publish --juno-jar <jar>`, GPU attention and the prefill region at their defaults. The console output was
not kept; the scores below were recomputed from the run files with the script's own method (median of three).

**Jars.** Baseline: commit 5e4c913, sha256 `a4f334f20440e3e1` (Phi-3 rotates and attends on the host side of
the region). Candidate: 5e4c913 plus the uncommitted change, sha256 `326cbb9ff2bd86e2`.

## Results (Juno t/s, median of three; all six readings)

| Metric | Baseline | Candidate | B/A | Bound |
|---|---|---|---|---|
| pp t/s | 378.93 (378.93, 375.88, 380.39) | **718.79** (718.79, 723.40, 706.62) | **1.897** | >= 0.95, met |
| tg t/s | 31.21 (31.00, 31.23, 31.21) | 30.89 (30.67, 31.17, 30.89) | **0.990** | >= 0.95, met |
| GPU pp ratio against the reference tool | 0.303x (0.303, 0.302, 0.309) | 0.577x (0.577, 0.583, 0.575) | | reading |
| GPU tg ratio against the reference tool | 0.513x | 0.511x | | reading |
| GC max pause, ms | 8, 8, 10 | 8, 8, 8 | | reading |
| Allocated bytes per generated token | 206.7M (205.4M, 206.7M, 212.2M) | 207.6M (212.2M, 207.6M, 205.9M) | | reading |

Every candidate prefill reading is above every baseline reading. Every row scorable, every prefill 512 of
512 tokens. Generation does not run through the prefill region; its 0.990 is inside the spread of either side.

## How to read it

Each `baseline-N` / `candidate-N` directory is one harness run (JSON and Markdown only; local paths made
repository-relative, the reference tool's install directory replaced by `<reference-tool-bin>`). The t/s
figures are `prompt_eval_tps` and `token_gen_tps` in `*-juno.json`; the ratio, GC and allocation columns are
each run's `INDEX.md` row. The ratios against the reference tool are from a gate A/B with one repetition per
run, not the closing sweeps the end-of-plan targets are scored on.
