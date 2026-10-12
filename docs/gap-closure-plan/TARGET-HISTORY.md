# Program target: history of readings

Moved out of [`README.md`](README.md) on 2026-10-11 (plan review). The README keeps the end-of-plan targets, the
machine-read tables `check-plan-thresholds.sh` reads, and the provenance of the current reference column. This file
keeps every earlier column and the reason each one stopped being the reference. It is history: nothing here is
scored against, and nothing outside this plan tree may point at it (README execution rule 5).

**When a new reference column is taken**, add it to the right of the table below in the same change that updates
the README's current-reference table and marks the superseded run in `docs/perf-compare/` (README, "Rule: the
reference column moves in the same change that supersedes it").

| Metric | Target at end of plan | Reading when this plan was written | Parity-corrected reading (2026-09-25) | After the CPU RoPE table (2026-09-27) | Reference 2026-09-27, late | Reference 2026-09-30 (pinned GPU re-baseline) | Reference 2026-10-01 (pinned CPU) | Reference 2026-10-03 (Tier 01B closing sweeps, pinned) | Reference 2026-10-04 (Tier 01C closing sweeps, pinned) | Reference 2026-10-08 (Tier 02 closing sweeps, pinned) | Reference 2026-10-11 (Tier 02C closing sweep, pinned CPU; GPU unchanged) | **Current reference** (2026-10-12: Tier 02D closing sweeps, pinned GPU; CPU unchanged) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPU tg, Phi-3.5-mini | >= **0.70x** (was 0.50x, met 2026-09-30) | 0.330x | 0.423x | 0.415x (0.414x tuned) | 0.416x (0.417x tuned) | 0.394x (0.396x tuned) | 0.394x (0.396x tuned; GPU unchanged) | 0.535x (0.537x tuned) | 0.545x (0.547x tuned) | **0.858x** (0.895x tuned) | 0.858x (GPU unchanged: Tier 02C's pinned GPU A/B read tg 1.127x the pre-tier build on this model; ratio not re-measured) | **0.973x** (0.972x tuned) |
| GPU tg, mistral-7b | >= **0.70x** (was 0.60x, met 2026-09-27) | 0.513x | 0.581x (0.631x tuned) | **0.646x** (0.638x tuned) | 0.639x (0.643x tuned) | 0.610x (0.610x tuned) | 0.610x (0.610x tuned) | 0.611x (0.607x tuned) | 0.627x (0.619x tuned) | **1.030x** (1.037x tuned) | 1.030x (GPU unchanged: Tier 02C's pinned GPU A/B read tg 1.021x; ratio not re-measured) | 1.066x (1.061x tuned) |
| GPU pp, every sweep model | retired 2026-10-04: met at `n_prompt=128` (was >= 0.25x unqualified, before that 0.15x); replaced by the per-length rows of the end-of-plan target table below | see the caveat below — not 0.016x to 0.031x | 0.036x to 0.073x, median 0.045x | 0.039x to 0.091x, median 0.051x | 0.040x to 0.101x, median 0.064x (Phi-3.5-mini 0.040x binding) | 0.037x to 0.101x at `n_prompt=128`, median 0.067x; 0.011x to 0.098x at 512 (Phi-3.5-mini binding at both) | 0.037x to 0.101x at `n_prompt=128`; 0.011x to 0.098x at 512 (GPU unchanged) | 0.173x to 0.299x at `n_prompt=128`, median 0.289x; 0.146x to 0.285x at 512, median 0.268x (Phi-3.5-mini binding at both) | 0.303x to 0.729x at `n_prompt=128`, median 0.570x; 0.226x to 0.468x at 512, median 0.373x (Phi-3.5-mini binding at both); 0.099x to 0.180x at 2048, median 0.138x (Phi-3.5-mini binding) | **0.584x to 0.807x** at `n_prompt=128`, median 0.681x; **0.594x to 0.825x** at 512, median 0.776x; **0.540x to 0.706x** at 2048, median 0.689x (Phi-3.5-mini binding at all three) | unchanged (Tier 02C's pinned GPU A/B read pp 0.973x to 1.039x the pre-tier build) | **0.681x to 0.785x** at `n_prompt=128`, median 0.728x (TinyLlama binding); **0.726x to 0.833x** at 512, median 0.798x; **0.602x to 0.730x** at 2048, median 0.697x (Phi-3.5-mini binding at 512 and 2048) |
| CPU tg, every sweep model | >= **0.25x** (restated again 2026-10-11 from Tier 02C's closing sweep; was 0.40x from 2026-10-10, and a 0.25x placeholder before that) | 0.106x to 0.147x | 0.079x to 0.128x | 0.090x to 0.135x | 0.090x to 0.135x (CPU not re-measured) | 0.090x to 0.135x (CPU not re-measured) | 0.092x to 0.121x (mistral-7b binding; pp 0.049x to 0.103x, Phi-3.5-mini binding) | 0.092x to 0.121x (CPU not re-measured) | 0.092x to 0.121x (CPU not re-measured) | **0.092x to 0.121x** (CPU not re-measured; Tier 02's pinned CPU A/B 0.997x to 1.001x the pre-tier build) | **0.215x to 0.238x** (Qwen2.5-3B binding; pp 0.224x to 0.256x, TinyLlama binding; Tier 02C closing sweep `20261011T013059Z`, pinned) | 0.215x to 0.238x (CPU not re-measured; the Tier 02D pinned CPU A/B read tg 0.999x to 1.005x and pp 0.983x to 1.030x the pre-tier build) |

## Provenance of superseded columns

Each paragraph below was written in the README while its column was current or just superseded, and is kept as
written; "the column to its left" refers to the table above.

**The 2026-10-08 column** was the reference until then. Its GPU rows are Tier 02's closing sweeps (2026-10-08,
owner run, clocks pinned, HEAD `51fa5d4` plus Tier 02's decode-width attention path and test-only changes, jar
`43b22345f1175c8f`): `docs/perf-compare/20261008T173153Z/` (`n_prompt=128`), `20261008T174605Z/` (512),
`20261008T180014Z/` (2048; TinyLlama, Qwen2.5-3B, Mistral 7B) and `20261008T181417Z/` (2048, Phi-3.5-mini at an
explicit 8 GiB heap); every row scorable, prompt parity exact, same harness as the column to its left. It is a
measurement boundary for GPU generation: from Tier 02 on the engine's defaults run the device-resident decode region
(`--gpu-residency auto`), which the column to its left did not. Its CPU row is unchanged (no CPU path changed; Tier 02's
pinned CPU A/B read 0.997x to 1.001x). Made the reference by owner decision (Tier 02 decision 14). With it every GPU
end-of-plan row is met and retired; the CPU rows are what remains.

**The column to its left** was the reference until then. Its GPU rows are Tier 01C's closing sweeps
(2026-10-04, owner run, clocks pinned, HEAD `59c53cc` plus Tier 01C's two out-of-tier fixes, jar
`5ab4c78c505a4cef`): `docs/perf-compare/20261004T113210Z/` (`n_prompt=128`) and `20261004T114812Z/`
(`n_prompt=512`), same method and harness as the column to its left, so ratios compare across the two
directly; every row scorable, every prefill at its `n_prompt`. Its CPU row is unchanged (no CPU path
changed). Its 2048 figures are from the 2048 re-run on the same code (jar `a90122b2855bb826`, which adds two
fixes that do not touch throughput: an out-of-heap request fails instead of hanging, and prompt encoding
is recorded so the harness's span check can subtract it): `docs/perf-compare/20261004T220015Z/`
(TinyLlama, Qwen2.5-3B, Mistral 7B at their fixed heaps) and `20261004T222758Z/` (Phi-3.5-mini at an
explicit 8 GiB heap, since its 2048-token live set of about 6.1 GB does not fit the fixed 6 GiB). The
interrupted first attempt, `20261004T120622Z-partial`, is superseded by them. Made the reference by
owner decision (Tier 01C decision 6).

**The column left of that** was the reference before it. Its GPU rows are Tier 01B's closing sweeps
(2026-10-03, owner run, clocks pinned, HEAD `1ac490a`): `docs/perf-compare/20261003T042440Z/`
(`n_prompt=128`) and `20261003T044014Z/` (`n_prompt=512`), same method and harness as the 2026-09-30 GPU
re-baseline, so ratios compare across the two directly; every row scorable, every prefill at its
`n_prompt`. Its CPU row is unchanged from the column to its left, described next, which was the reference
until then. **That column's** CPU row is Tier 01B step 3a's pinned
CPU sweep (2026-10-01, `docs/perf-compare/20261001T180241Z/`): clocks pinned, Juno's common-pool
parallelism set from `--threads` so its kernels run the reference tool's `-t 12`. Pinning is a
measurement boundary for CPU readings (both engines' absolute t/s fell; tinyllama's reference prefill
by 16%), so compare CPU ratios across it, never absolute t/s. Its GPU rows are unchanged from the
column to its left, Tier 01B's step 2 re-baseline (2026-09-30): GPU `docs/perf-compare/20260930T135225Z/` (`n_prompt=128`) and `20260930T141026Z/`
(`n_prompt=512`), both with pinned clocks, the operating-system JFR clock and the warm-up recording;
CPU unchanged from `20260927T094414Z/`. It is a measurement boundary against every column to its left
(turbo off lowers both engines' absolute throughput, and the harness changes of 2026-09-28 and
2026-09-29 apply), so compare ratios across it, never absolute t/s. Tier 01B's execution record derives
it. The column to its left, described next, was the reference until then.

The previous reference was from late 2026-09-27:
GPU `docs/perf-compare/20260927T232837Z/` with the `qwen2.5-3b` tuned and both `Phi-3.5-mini` rows
from its re-run `20260927T234659Z/`; CPU `20260927T094414Z/`, which the change behind it did not touch.
It follows the per-request device-memory fix and the extracted FP16 pack loop, a GPU-prefill
measurement boundary recorded in Tier 01's "Out-of-tier changes" (GPU pp +10% to +34% on tinyllama,
qwen2.5-3b and mistral-7b; generation within the gate). The column to its left was the reference for
most of that day and is now superseded; it is kept, not overwritten, for the same reason the older
columns are.

**The 2026-09-27 (RoPE table) column** was taken after the scalar CPU RoPE stopped
recomputing its angles for every head of every layer, which is a measurement boundary (Tier 01's
"Out-of-tier changes"). Sources: GPU `docs/perf-compare/20260927T091155Z/` with the `tinyllama`,
`qwen2.5-3b` and `Phi-3.5-mini` rows from its re-run `20260927T093054Z/`; CPU `20260927T094414Z/`.
Juno's GPU throughput rose where the rotation runs on the CPU path (tinyllama tg +26%, mistral-7b
+17%, qwen2.5-3b +12%; pp +22% to +35%) and held on Phi-3.5-mini, which has its own rotation; its
ratio moved only with the reference tool's own reading. CPU throughput did not move (the rotation was
never a visible share of a CPU forward pass); the CPU ratios moved with the reference tool's readings.
The mistral-7b GPU tg row met its end-of-plan target on the default flags from this column on.

**The 2026-09-25 column was the reference until the morning of 2026-09-27, and the two columns
before it are not comparable with it.**
Those readings come from `docs/perf-compare/20260925T172231Z/` (GPU) and `20260925T174146Z/` (CPU),
the first sweep taken with prompt-token parity, warm repeated measurement, a fixed heap, `min_tokens`
generation parity, and prefill and generation measured in two separate runs as the reference tool
measures them. Tier 01's execution record derives them and explains each correction. The generation
figures moved up mainly because the measurement stopped penalizing Juno, not because the engine got
faster — score a later tier against the right-hand column, never against the middle one.

**The pp row's starting reading is not yet known, and the target is provisional until it is.** The
0.016x-to-0.031x figures widely quoted from `docs/perf-compare/20260918T204809Z/` were taken with
`raw_prompt: 0`, meaning llama-bench prefilled 128 tokens while Juno prefilled 20 to 30 — they are
not a like-for-like prefill measurement (see "Benchmark parity preconditions" below). The only
parity-corrected prefill figures on record are worse and narrower: at a real 512-token prompt,
`20260915T043143Z` puts mistral-7b at 0.0166x and tinyllama at 0.0075x, and the best GPU prefill
number anywhere in this repository — TinyLlama at 119.56 t/s with `--gpu-attention on` and
`--prefill-batch 32`, `20260916T040113Z-prefill/` — is about **0.028x** of llama.cpp's 4225 t/s on
the same shape. Tier 01's re-baseline establishes the real starting reading under parity; **the
0.15x target is re-derived from that number in Tier 01's own file before any later tier is scored
against it.** A target anchored to a measurement that was never like-for-like is not a target.
