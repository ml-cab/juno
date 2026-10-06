# Phi-3 prefill region attention: baseline against candidate - 20261006T184535Z (gpu, unpinned)

**Purpose.** A same-session A/B of the change that moved Phi-3's prefill-window RoPE and attention
inside the prefill-window device region, on the model it touches (Phi-3.5-mini), at `n_prompt=512`.
It reads whether prefill got faster and whether generation is unaffected.

**Pinned: no. Indicative, not a gate reading.** The tier's 0.95x no-regression gate is scored at the
tier's close on a pinned same-hour A/B by the owner. `run-gate.sh` here runs pinned by default and is
that gate's form for this change; this reading is its `PIN=0` run.

**Command.** `PIN=0 bash dist/phi3-prefill-region-ab/run-gate.sh` (the script is copied here as
`run-gate.sh`): baseline and candidate alternated A B A B A B, each run
`compare-llama-cpp.sh --gpu --models Phi-3.5-mini-instruct-Q4_K_M --n-prompt 512 --juno-warmup 2
--juno-reps 1 --reps 1 --no-tuned-lane --no-publish --juno-jar <jar>`, GPU attention and the prefill
region at their defaults. Console output: `console.txt`.

**Jars.** Baseline: commit 5e4c913 built from a `git archive` of that commit, sha256
`a4f334f20440e3e1` (Phi-3 rotates and attends on the host side of the region). Candidate: this working
tree on 5e4c913 plus the uncommitted change, sha256 `326cbb9ff2bd86e2`.

## Results (Juno t/s, median of three; all six readings)

| Metric | Baseline | Candidate | B/A |
|---|---|---|---|
| pp t/s | 375.67 (373.23, 375.67, 381.62) | **705.19** (712.31, 680.26, 705.19) | **1.877** |
| tg t/s | 31.03 (30.67, 31.03, 31.19) | 31.26 (31.51, 31.26, 31.20) | 1.007 |
| GPU pp ratio against the reference tool | 0.306x (0.305, 0.306, 0.312) | 0.578x (0.585, 0.556, 0.578) | |
| GPU tg ratio against the reference tool | 0.515x | 0.519x | |
| GC max pause, ms | 9, 8, 9 | 8, 8, 8 | |
| Allocated bytes per generated token | 207.5M (212.2M, 207.5M, 207.5M) | 207.6M (208.0M, 207.6M, 207.5M) | |

Every candidate prefill reading is above every baseline reading. Generation does not run through the
prefill region and reads the same within noise.

## How to read it

Each `baseline-N` / `candidate-N` directory is one harness run (JSON and Markdown only; local paths made
repository-relative, the reference tool's install directory replaced by `<reference-tool-bin>`). The
t/s figures are `prompt_eval_tps` and `token_gen_tps` in `*-juno.json`; the ratio, GC and allocation
columns are each run's `INDEX.md` row. Unpinned: the GPU and CPU clocks moved freely, so read the B/A
ratios, not absolute t/s, and do not compare these ratios with the pinned sweeps the program targets are
scored on.
