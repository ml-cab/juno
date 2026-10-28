# CPU decode and prefill kernels: closing gate, run 2 (2026-10-27 15:47Z to 2026-10-28 02:30Z)

**Purpose.** The re-run of the owner's pinned closing gate for the CPU kernel work on the final build (owner decision 8
of the tier record): the closing sweeps (Part S) and the CPU same-hour A/B against the build before the work (Part C).
Parts K (decode kernels), A (GPU) and L (LoRA) stand from run 1,
[`20261013T230239Z-cpu-kernels-close-gate`](../20261013T230239Z-cpu-kernels-close-gate/INDEX.md): what changed after run 1 is
the prefill matmul tile's leftover-row handling (same output bits) and the metrics extractor.

**Pinned:** yes. CPU governor `performance`, turbo off (`compare-llama-cpp.sh --pin-clocks`); every sweep and every A/B
run reports `clock_pinned: true`.

**Builds.**

| Side | Jar sha256 (first 16) | Build |
|---|---|---|
| baseline | `b9a0440b3a1b597a` | HEAD `01027ee9`, before any change to the CPU kernels (the jar of the step 2 sweeps and of run 1) |
| candidate | `4e480ab9131887eb` | `c2633a68` plus the in-recording allocation field (extractor and harness); the padded prefill tile |

Both sides' recordings were extracted by the candidate's jar (`host.json`: `jfr_extractor_jar_sha256_16`
`4e480ab9131887eb`), so both read allocation by one definition.

**Command.** `bash dist/cpu-kernels-close/wait-and-run.sh`, which ran `PART=SC run-gate.sh`; the full log is `gate.log`
here. Part S runs `compare-llama-cpp.sh --cpu --pin-clocks --n-gen 64 --threads 12 --reps 3 --juno-reps 3 --juno-warmup 2
--jfr 120m --publish` per prompt length. Part C alternates the two jars A B A B A B through `compare-llama-cpp.sh --cpu
--pin-clocks --threads 12 --n-prompt <128|512> --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish
--juno-jar <jar>`; per-run results are in `ab/`.

**An interrupted first attempt.** The gate first started at 2026-10-27 05:38Z. The host reset at about 10:56Z during
Part S's last sweep (Phi-3.5-mini at 2048, Juno repetition 2 of 3), with nothing in the kernel log (a machine-level
reset; the kernel does not reboot on panic on this host). Its three finished sweeps,
[`20261027T053823Z`](../20261027T053823Z/INDEX.md) (128), [`20261027T060952Z`](../20261027T060952Z/INDEX.md) (512) and
[`20261027T071820Z`](../20261027T071820Z/INDEX.md) (2048), are kept, marked not scored. The gate was re-run whole from
15:47Z; its log is `gate-attempt-interrupted.log` here.

## Part S: closing sweeps (published separately)

| Model | CPU tg ratio @128 | pp ratio @128 | pp ratio @512 | pp ratio @2048 | pp t/s @512 against the pre-tier sweep | pp t/s @2048 against the pre-tier sweep |
|---|---|---|---|---|---|---|
| TinyLlama 1.1B | 0.337x | 0.436x | 0.429x | 0.378x | 12.38 -> 34.00 (2.75x) | 6.10 -> 26.73 (4.38x) |
| Qwen2.5-3B | 0.275x | 0.431x | 0.422x | 0.397x | 5.14 -> 11.90 (2.32x) | 3.18 -> 10.40 (3.27x) |
| Phi-3.5-mini | **0.259x** | 0.516x | 0.512x | 0.388x | 3.81 -> 10.00 (2.63x) | 1.68 -> 6.50 (3.88x) |
| Mistral 7B | 0.309x | 0.455x | 0.436x | 0.411x | 2.25 -> 5.17 (2.30x) | 1.29 -> 4.45 (3.44x) |

Bars: CPU tg ratio >= 0.25x (Phi-3.5-mini binding, 0.259x); CPU pp ratio at 512 >= 0.30x (Qwen2.5-3B binding, 0.422x);
pp t/s at 2048 >= 1.50x the step 2 sweeps of the pre-tier jar (3.27x to 4.38x). All met. The 512 column against the step 2
sweep is reported; the 512 bar is Part C's same-hour A/B.

Sweeps: [`20261027T154748Z`](../20261027T154748Z/INDEX.md) (128), [`20261027T161927Z`](../20261027T161927Z/INDEX.md) (512),
[`20261027T172646Z`](../20261027T172646Z/INDEX.md) (2048) and [`20261027T202604Z`](../20261027T202604Z/INDEX.md) (2048,
Phi-3.5-mini at 8 GiB). Pre-tier sweeps: `20261012T045931Z` (512), `20261012T071213Z` and `20261012T162344Z` (2048).

**Phi-3.5-mini at 2048 is flagged `NOISY` on its generation lane only.** Its three generation readings (a 10-token prompt)
span 2.52 to 2.96 t/s, 15.9% of their median against the harness's 15%. The 2048 bar reads the prefill lane, whose three
readings span 6.34 to 6.54 t/s (3.1%), so the gated figure is scorable; the generation figure of that sweep is not gated.

**The reference tool's decode reading moves between sessions.** On the same jar, the interrupted attempt's 128 sweep read
the reference tool's CPU tg at 27.75 t/s on TinyLlama (Juno 11.68), this run's at 37.08 (Juno 12.51); across the four
models the reference tool's tg at 128 rose 13% to 34% between the two sessions and Juno's 2% to 7%. Within a sweep, Juno's
three repetitions agree to about 2%; the reference tool's own repetitions vary more (standard deviation 1.7% to 4.8% of
the mean on three models, and on TinyLlama 16.5% in this run and 7.2% in the interrupted one). On Phi-3.5-mini, which
binds, the reference tool read 10.09 and 11.93 t/s, each with a 3.1% deviation. Earlier pinned 128 sweeps read the
reference tool at 32.50 t/s on TinyLlama (`20261011T013059Z`) and 30.02 (`20261013T231340Z`). The CPU tg ratios here
(0.259x to 0.337x) are therefore the low readings of this build; the interrupted attempt read 0.301x to 0.421x. The
harness's noise check reads Juno's repetitions only. `host-specific`.

## Part C: CPU same-hour A/B, pinned (per-run results in `ab/cpu128-*`, `ab/cpu512-*`)

Medians of three; B/A is candidate over baseline; the bracket is the three runs' range. Allocation is the gate's figure,
`jfr.allocated_bytes_per_token_in_recording` (each thread's growth inside the measured recording); the harness's
whole-thread figure, which also counts model load and the warmups, is printed beside it and not gated.

| `n_prompt` | Model | pp t/s A -> B (B/A, bar) | tg t/s A -> B (B/A, bar) | Allocation per token in the recording A -> B (B/A, bar <= 1.10) | Whole-thread, reported |
|---|---|---|---|---|---|
| 128 | TinyLlama | 16.63 [14.21-16.63] -> 34.13 [33.52-35.26] (2.052, >= 0.95) | 7.98 -> 12.14 (1.522, >= 0.95) | 0.306 MB -> 0.300 MB (0.981) | 79.5 -> 126.5 MB (1.59x) |
| 128 | Qwen2.5-3B | 6.15 [6.08-6.26] -> 12.43 [11.74-12.49] (2.020, >= 1.10) | 2.61 -> 4.30 (1.646, >= 1.10) | 1.277 MB -> 1.269 MB (0.994) | 145.1 -> 200.2 MB (1.38x) |
| 128 | Phi-3.5-mini | 4.95 [4.86-4.98] -> 10.04 [9.77-10.27] (2.027, >= 0.95) | 2.34 -> 3.08 (1.314, >= 0.95) | 0.348 MB -> 0.352 MB (1.013) | 166.3 -> 244.4 MB (1.47x) |
| 128 | Mistral 7B | 2.63 [2.61-2.63] -> 5.03 [5.03-5.27] (1.916, >= 1.10) | 1.30 -> 2.12 (1.633, >= 1.10) | 0.347 MB -> 0.348 MB (1.003) | 202.2 -> 271.6 MB (1.34x) |
| 512 | TinyLlama | 12.58 [12.54-12.61] -> 33.76 [33.46-33.86] (2.684, >= 1.10) | 7.98 -> 12.12 (1.519) | 0.299 MB -> 0.296 MB (0.988, not gated at 512) | 79.6 -> 140.8 MB (1.77x) |
| 512 | Qwen2.5-3B | 5.18 [5.15-5.19] -> 11.74 [11.36-11.94] (2.267, >= 1.10) | 2.62 -> 4.31 (1.643) | 1.289 MB -> 1.269 MB (0.985) | 150.8 -> 197.4 MB (1.31x) |
| 512 | Phi-3.5-mini | 3.91 [3.89-3.93] -> 9.39 [9.20-9.94] (2.402, >= 1.10) | 2.33 -> 3.08 (1.324) | 0.342 MB -> 0.342 MB (1.002) | 166.3 -> 229.7 MB (1.38x) |
| 512 | Mistral 7B | 2.28 [2.24-2.30] -> 5.12 [5.09-5.18] (2.252, >= 1.10) | 1.30 -> 2.16 (1.660) | 0.347 MB -> 0.348 MB (1.001) | 196.1 -> 272.6 MB (1.39x) |

In-span GC pause total: 0.0 ms on every run of both builds (bar <= 5 ms, since the baseline reads under 5 ms).

**Every bar met.** The gate printed `GATES MET`. The measured requests allocate what the pre-tier build's do (0.981x to
1.013x); the whole-thread figure's 1.31x to 1.77x is the vector kernels' one-time allocation before C2 compiles them,
during model load and the warmups, outside the measured request (run 1's record traces it).

## How to read this

Against run 1, the candidate's prefill rose on every model (pp B/A at 128: 1.34 to 1.46 in run 1, 1.92 to 2.05 here; at
512: 1.56 to 2.02, now 2.25 to 2.68), and the pinned CPU pp ratios at 512 rose from 0.312x to 0.369x to 0.422x to 0.512x.
Run 1's candidate boxed vectors in the prefill tile's leftover-row path (tens of GB allocated per request); the padded
tile removes that allocation and its time. Decode is unchanged within noise (tg B/A 1.31 to 1.66 in both runs), since
decode does not reach the prefill tile. `expected-general` for the mechanism (a boxed vector allocates on every lane
operation); the magnitudes are `host-specific`.
