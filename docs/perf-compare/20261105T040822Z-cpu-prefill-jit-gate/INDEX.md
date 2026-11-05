# CPU batched matmul without loop-carried receivers: pinned gate (2026-11-05 04:08Z to 09:11Z)

**Purpose.** The owner's pinned no-regression gate for the change to `CpuIntegerGemm`'s four-row prefill kernels
(owner decision (a) of 2026-11-04): no Vector API method is called on a vector carried from one loop iteration to the
next, so C2 cannot box the accumulators through a polluted type profile. The diagnosis is in
[`20261104T191116Z-cpu-prefill-jit`](../20261104T191116Z-cpu-prefill-jit/INDEX.md); the change, the reproducer and the
unpinned readings are in [`20261104T234612Z-cpu-prefill-jit-fix`](../20261104T234612Z-cpu-prefill-jit-fix/INDEX.md).
This gate measures what the change costs on the fast path. It does not measure the slow compile, which none of the
server JVMs here got (the baseline's prefill readings are its usual ones in every run).

**Pinned:** yes. CPU A/B: governor `performance`, turbo off (`compare-llama-cpp.sh --pin-clocks`); every run's
`host.json` reads `clock_pinned: true`. LoRA: pinned by hand by the gate script (the same two settings), restored on
exit. Owner run, started by `wait-and-run.sh` once `READY` existed.

**Builds.**

| Side | Player shaded jar sha256 (first 16) | Build |
|---|---|---|
| baseline (A) | `15083b9c320c1c36` | `80ec949e`: this tree with `CpuIntegerGemm.java` and `CpuKernelVectorEscapeTest.java` taken from `HEAD` |
| candidate (B) | `ea700f227c83b911` | `80ec949e` plus the uncommitted change (the final form, the offset negated once per sub-block) |

The gate checked that the tree's own jar equals the candidate before it started. Both sides' recordings were
extracted by the candidate's jar (`host.json`: `jfr_extractor_jar_sha256_16` `ea700f227c83b911`), so both read
allocation by one definition. Host: Xeon E5-1650 v2, 12 threads, 128-bit integer vectors, GTX 1080 (LoRA GPU lane),
OpenJDK 25.0.4.1, kernel clock source `hpet`.

**Command.** `bash dist/cpu-prefill-jit-gate/wait-and-run.sh`, which ran `run-gate.sh` (both copied here); the full log
is `gate.log`. The CPU A/B alternates the two jars A B A B A B through `compare-llama-cpp.sh --cpu --pin-clocks
--threads 12 --n-prompt <512|128> --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish --juno-jar
<jar>` on the four sweep models; per-run results are in `ab/` (`host.json`, the merged `*-juno.json` and the greedy
reply per model). LoRA runs each tree's own `compare-lora.sh --<gpu|cpu> --reps 1 --skip-build --no-publish`, A B A B
A B; per-run results are in `lora/`. Scored on the median of three, B/A unrounded. The comment at the top of
`run-gate.sh` quotes the offset step as `p.neg().add(f0)`, the form before the negation was hoisted; the candidate jar
is the final form (`o.mul(x).add(f0)`, `o` negated once per sub-block).

## CPU same-hour A/B, pinned

Medians of three; the bracket is the three runs' range; B/A is candidate over baseline. Allocation is
`jfr.allocated_bytes_per_token_in_recording` (each thread's growth inside the measured recording).

| `n_prompt` | Model | pp t/s A -> B (B/A; bar) | tg t/s A -> B (B/A; bar) | Allocation per token in the recording A -> B (B/A; bar) |
|---|---|---|---|---|
| 512 | TinyLlama 1.1B | 33.31 [31.32-33.43] -> 32.13 [31.83-33.59] (**0.965**; >= 0.95) | 12.02 -> 11.73 (0.976; reported) | 0.304 MB -> 0.302 MB (0.991; reported) |
| 512 | Qwen2.5-3B | 11.09 [11.05-11.28] -> 11.43 [11.04-11.74] (**1.031**; >= 0.95) | 4.218 -> 4.204 (0.997; reported) | 1.270 MB -> 1.275 MB (1.003; reported) |
| 512 | Phi-3.5-mini | 9.57 [8.26-9.58] -> 9.55 [8.63-9.69] (**0.999**; >= 0.95) | 3.018 -> 3.035 (1.006; reported) | 0.380 MB -> 0.353 MB (0.927; reported) |
| 512 | Mistral 7B | 4.88 [4.67-5.13] -> 5.08 [4.99-5.17] (**1.040**; >= 0.95) | 2.114 -> 2.087 (0.987; reported) | 0.348 MB -> 0.343 MB (0.988; reported) |
| 128 | TinyLlama 1.1B | 34.43 [33.25-34.46] -> 34.60 [33.97-34.74] (**1.005**; >= 0.95) | 11.88 [11.61-11.91] -> 12.06 [11.95-12.12] (**1.015**; >= 0.95) | 0.305 MB -> 0.303 MB (**0.995**; <= 1.10) |
| 128 | Qwen2.5-3B | 11.87 [11.70-12.28] -> 11.92 [11.90-11.95] (**1.004**; >= 0.95) | 4.235 [4.234-4.247] -> 4.264 [4.180-4.268] (**1.007**; >= 0.95) | 1.289 MB -> 1.268 MB (**0.984**; <= 1.10) |
| 128 | Phi-3.5-mini | 9.66 [9.58-9.95] -> 9.82 [9.69-10.11] (**1.016**; >= 0.95) | 3.040 [3.028-3.040] -> 3.050 [3.015-3.051] (**1.003**; >= 0.95) | 0.351 MB -> 0.348 MB (**0.992**; <= 1.10) |
| 128 | Mistral 7B | 5.07 [5.06-5.34] -> 5.26 [4.96-5.31] (**1.037**; >= 0.95) | 2.113 [2.103-2.134] -> 2.130 [2.130-2.131] (**1.008**; >= 0.95) | 0.348 MB -> 0.343 MB (**0.985**; <= 1.10) |

- **GC pause in the token span** (bar: <= 5 ms, since every baseline median is under 5 ms): median 0.0 ms on every model
  of both builds. One run reads above zero: Qwen2.5-3B at 128, candidate run 1, 5.04 ms (its other two runs and all
  three baseline runs read 0.0).
- **Greedy output** (temperature 0, the generation lane's 64-token reply): the candidate's equals the baseline's in 3 of
  3 runs on every model at both prompt lengths (gated).
- **Readings withheld.** The harness withholds a run's JFR decode reading when the recorded spans and the engine
  clock disagree by more than its bounds (`JFR timestamps disagree with the engine clock`; token-span residual -28 to
  -56 ms against the -25 ms floor; the kernel clock source here is `hpet`). That happened in five generation runs:
  Mistral 7B at 512 (baseline runs 1 and 2, candidate run 1), Phi-3.5-mini at 512 (candidate run 1) and Mistral 7B at
  128 (candidate run 1, the gate log's `na`). The gate takes the median of the readings left, the lower one when two
  are left. The only gated one is Mistral 7B's tg at 128 (candidate from two readings); its request-timed tg
  (`api_token_gen_tps`, median of three per side) reads 1.941 -> 1.960 t/s, 1.010.
- **Whole-thread allocation per token** (`jfr.allocated_bytes_per_token`, which also counts model load and the
  warmups), reported and not gated: 0.911x to 1.317x by model, 109 to 276 MB per token. As in earlier gates, this is the
  vector kernels' allocation while still interpreted, before C2 compiles them, outside the measured request.

## LoRA, pinned by hand

Each tree's own `compare-lora.sh` (the baseline's from `dist/cpu-prefill-jit-gate/baseline-tree/`). Training's frozen
forward runs the float-activation kernels, which the change does not touch: a control.

| Lane | Training ms per pass A -> B (B/A; bar <= 1.0526) | Playback t/s A -> B (B/A; bar >= 0.80) | Runs |
|---|---|---|---|
| GPU | 2800 [2800-2800] -> 2800 [2800-2800] (**1.000**) | 12.73 [12.64-12.84] -> 12.43 [11.95-12.80] (**0.977**) | 15 passes, final loss 1.1797, recall correct, in all six |
| CPU | 51,467 [51,467-51,533] -> 51,533 [51,467-52,133] (**1.001**) | 0.842 [0.837-0.845] -> 0.847 [0.844-0.856] (**1.006**) | 15 passes, final loss 1.1795, recall correct, in all six |

**Every bar met.** The gate printed `GATE: PASS`.

## How to read this

The change costs nothing measurable on the fast path. Pinned, the candidate's CPU prefill reads 0.965x to 1.040x the
baseline's at 512 and 1.004x to 1.037x at 128, inside the runs' own spreads (TinyLlama at 512, the lowest, has
overlapping ranges: 31.32 to 33.43 against 31.83 to 33.59). Decode, which the change does not reach, reads 1.003x to
1.015x at 128. The measured requests allocate what the baseline's do (0.984x to 0.995x at 128). Greedy output is
identical in all 24 run pairs, as the change is meant to be bit for bit. LoRA training and playback are unchanged.
What the change buys, the slow compile gone, is shown by the reproducer and the kit runs in
`20261104T234612Z-cpu-prefill-jit-fix/`, not here. `expected-general` for the absence of cost (the operations are the
same, reordered); the magnitudes are `host-specific`.

## Files

- `gate.log`: the gate's full output.
- `run-gate.sh`, `wait-and-run.sh`: the scripts as run.
- `ab/<cpu512|cpu128>-<baseline|candidate>-<1..3>-*`: per run, `host.json`, each model's merged `*-juno.json` and its
  greedy reply (`*-generate-rep1-juno-response.json`). The JFR recordings and server logs are not published.
- `lora/lora-<gpu|cpu>-<baseline|candidate>-<1..3>.json` and `.log`: each LoRA run's `current.json` and log.
- Paths are scrubbed to `<repo>` and `<home>`.
