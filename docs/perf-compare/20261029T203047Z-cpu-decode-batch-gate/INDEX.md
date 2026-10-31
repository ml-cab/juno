# CPU decode batch: pinned throughput gate (2026-10-29)

**Purpose.** This is the scored run of the owner's threshold for the CPU decode-batch matmul. The gate requires CPU
aggregate token-generation t/s at `--parallel` 2, 4 and 8 to be at least 0.90x the previous build's, on TinyLlama and
Qwen2.5-3B. The unpinned reading of the same A/B, and the change itself, are in `../20261029T182639Z-cpu-decode-batch/`.

**Pinned: yes, by hand.** Owner run: governor `performance`, turbo off for every run, restored on exit.
`runs/clocks.txt` records the state before each of the 48 runs: all `performance,no_turbo=1`. Started by
`wait-and-run.sh` once the prepared gate's `READY` file existed; 2026-10-29, 20:30 to 22:37 UTC.

**Builds.**

- Baseline (A): `06d6d22d`, built from `git archive` (player shaded jar sha256 first 16 `758520d7ad418890`). A CPU
  decode batch runs the prefill gemm.
- Candidate (B): `06d6d22d` plus the uncommitted decode-batch work (`1c181de29e52d249`). A CPU decode batch runs
  `CpuIntegerDecodeBatch` and the serial-head attention.

The gate checked that the tree's own jar equals the candidate. OpenJDK 25.0.3.

## Method

`run-gate.sh` (copied here) runs the tree's `compare-parallel.sh` against each jar (`--juno-jar`):

- `--cpu`, P concurrent sessions at `--parallel P`, 64 tokens, the script's greedy prompt, batch window 50 ms;
- two untimed warmup rounds per server;
- heap 4 GB (TinyLlama) and 6 GB (Qwen2.5-3B);
- A B A B A B: for each repetition, every P value, baseline then candidate.

It reads the median of three `aggregate_token_gen_tps` per side and gates on B/A unrounded.

## Throughput

| Model | P | A runs | A median | B runs | B median | B/A | Gate |
|---|---|---|---|---|---|---|---|
| TinyLlama 1.1B | 1 | 8.91 / 8.71 / 9.14 | 8.91 | 9.20 / 8.74 / 7.96 | 8.74 | 0.981 | report |
| TinyLlama 1.1B | 2 | 9.42 / 7.65 / 8.68 | 8.68 | 8.94 / 8.86 / 9.18 | 8.94 | **1.029** | >= 0.90: met |
| TinyLlama 1.1B | 4 | 8.33 / 6.97 / 8.06 | 8.06 | 9.26 / 8.54 / 9.77 | 9.26 | **1.148** | >= 0.90: met |
| TinyLlama 1.1B | 8 | 8.03 / 6.81 / 8.92 | 8.03 | 9.75 / 8.49 / 9.91 | 9.75 | **1.214** | >= 0.90: met |
| Qwen2.5-3B | 1 | 3.51 / 3.44 / 3.55 | 3.51 | 3.55 / 3.39 / 3.53 | 3.53 | 1.004 | report |
| Qwen2.5-3B | 2 | 3.41 / 3.50 / 3.52 | 3.50 | 3.49 / 3.54 / 3.53 | 3.53 | **1.009** | >= 0.90: met |
| Qwen2.5-3B | 4 | 2.90 / 2.92 / 2.94 | 2.92 | 3.55 / 3.47 / 3.54 | 3.54 | **1.212** | >= 0.90: met |
| Qwen2.5-3B | 8 | 3.52 / 3.56 / 3.26 | 3.52 | 3.61 / 3.65 / 3.83 | 3.65 | **1.039** | >= 0.90: met |

**GATES MET.** The smallest ratio is 1.009 (Qwen2.5-3B, P=2).

- At P=1 both builds run the single-request path, so 0.981 and 1.004 are the run-to-run spread.
- TinyLlama's readings spread by up to 24% within a side even pinned (baseline P=8: 6.81 to 8.92). Qwen2.5-3B's
  spread is within 9%.
- Pinned, with turbo off, every reading is lower than the unpinned A/B's (TinyLlama P=1 8.7 to 8.9 t/s, against
  10.4 to 10.6). The ratios agree with it in direction at every point. Qwen2.5-3B at P=8 reads 1.039 pinned against
  1.134 unpinned.
- Against its own P=1, the candidate's P=8 reads 1.12x (TinyLlama) and 1.03x (Qwen2.5-3B); the baseline's reads
  0.90x and 1.00x. Batching adds little or no aggregate throughput on this host's CPU in either build.

## Replies: batched against single

Each build's replies at P=2, 4 and 8 against its own single reply (P=1, run 1); every session sends the same prompt,
greedy.

| Model | Build | P=2 | P=4 | P=8 |
|---|---|---|---|---|
| TinyLlama 1.1B | baseline | 0 of 6 | 4 of 12 | 5 of 24 |
| TinyLlama 1.1B | candidate | 0 of 6 | **0 of 12** | **0 of 24** |
| Qwen2.5-3B | baseline | 0 of 6 | 6 of 12 | 8 of 24 |
| Qwen2.5-3B | candidate | 0 of 6 | **0 of 12** | **0 of 24** |

The baseline differs in 23 of 84 batched replies; the candidate in none.

## Files

- `gate.log`: the gate's full output, scoring included.
- `run-gate.sh`, `wait-and-run.sh`: the gate as prepared.
- `runs/clocks.txt`: the clock state before each run.
- `runs/<model>-p<P>-<baseline|candidate>-<run>/parallel-<P>.json` and `replies/`: per-run results and replies,
  with local paths replaced by `<repo>`.
