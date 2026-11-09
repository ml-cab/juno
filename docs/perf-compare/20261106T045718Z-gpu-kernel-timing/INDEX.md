# Longest single GPU kernels at the sweep models' shapes, on the reference host's Linux

**Purpose.** The first reading of the kit's `gpu.kernel-timing` probe: the device time of single launches of the
longest kernels Juno runs, each launch timed alone by a pair of CUDA events. A display driver's watchdog holds every
single kernel to a time limit (two seconds by default on Windows, for a GPU that drives a display), so the reading is
per launch. The probe's check holds the slowest launch to 500 ms, a 4x margin. This is the Linux side of the
comparison a Windows run of the same probe on the same GPU will make; nothing here scores a threshold, which applies
on Windows.

**Pinned:** no. Not a throughput gate; GPU clocks are not pinned (the GTX 1080 refuses clock locking), and the
check uses the slowest launch, which a slow repetition can only make stricter.

**Build.** Juno `d8012b11` plus the uncommitted CUDA library and kit-probe changes (the kit reports "modified tree").
Kit jar SHA-256 `666c282aac7502a1b91684848d9058072174316f543fa4f5b444c18cdddc932f`. Host: Xeon E5-1650 v2, GTX 1080
(8 GiB, driver 580.178.04, CUDA 12.0 runtime), OpenJDK 25.0.4.1, Linux, clock source `hpet`. The GPU also drives this
host's desktop (8% utilization before the run, no other compute process).

**Commands.**
- 2048-token window, 5 launches per kernel: `./juno-kit.sh --probes env,gpu.kernel-timing,launcher.tinyllama --yes
  --models-dir models`, 2026-11-06 04:57:18Z to 04:58:06Z. `kernel-timing-2048.json` is the probe's report from the
  zip, `env.json` the environment reading (with the CUDA libraries loaded), `SUMMARY.txt` the kit's summary.
- 4096 and 8192 tokens, 3 launches per kernel, minutes later with the same jar: `java -Xmx3g
  --enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector -cp juno-kit/target/juno-kit-0.1.2-shaded.jar
  cab.ml.juno.kit.ProbeMain GpuKernelTimingProbe gpu.kernel-timing - tokens=<n> reps=3`
  (`kernel-timing-4096.json`, `kernel-timing-8192.json`).

## Single-launch device time, ms (median, min to max)

| Kernel and shape | 2048 tokens (n=5) | 4096 tokens (n=3) | 8192 tokens (n=3) |
|---|---|---|---|
| Prefill attention, TinyLlama 1.1B, 32/4/64 heads | 15.99 (15.78 to 16.82) | 60.32 (57.29 to 63.94) | 215.48 (215.25 to 217.04) |
| Prefill attention, Qwen2.5-3B, 16/2/128 heads | 19.66 (19.39 to 20.15) | 75.16 (74.59 to 75.29) | 301.03 (300.84 to 301.13) |
| Prefill attention, Phi-3.5-mini, 32/32/96 heads | 34.67 (34.18 to 35.17) | 139.83 (139.46 to 140.18) | 564.86 (563.84 to 565.39) |
| Prefill attention, Mistral 7B, 32/8/128 heads | **37.04** (36.55 to **37.37**) | 149.34 (148.91 to 149.73) | 596.03 (595.86 to **596.51**) |
| Tiled K-quant GEMM, Mistral 7B `ffn_gate`/`ffn_up` 14336x4096 Q4_K | 22.91 (22.65 to 23.02) | 45.81 (45.79 to 45.96) | 91.03 (90.89 to 91.63) |
| Tiled K-quant GEMM, Mistral 7B `ffn_down` 4096x14336 Q6_K | 24.74 (24.63 to 25.24) | 48.56 (48.29 to 49.06) | 97.79 (97.67 to 98.72) |
| LM head GEMV, Qwen2.5-3B output 151936x2048 Q6_K, one row | 1.89 (1.88 to 1.93) | 1.95 (1.89 to 1.95) | 1.89 (1.89 to 2.01) |
| **Check: slowest launch <= 500 ms** | **PASS** (37.37) | PASS (149.73) | **FAIL** (596.51) |

## How to read it

- The tokens are the prefill window's rows. On the GPU the static schedule's adaptive window covers a whole prompt in
  one window, up to a device-memory ceiling (and at most 65,536 rows), so a prompt's length is the window's: one
  attention launch per layer covers all of it, row `b` over keys `[0, b + 1)`.
- Attention grows with the square of the window (about 4x per doubling), the GEMM linearly (2x), and the LM head not
  at all (one row per token). At 2048 tokens, the length the plan names, the longest kernel is 13x under the 500 ms
  bar. Extrapolated on this GPU, a Mistral 7B attention launch reaches 2 s near 16,000 tokens in one window; a faster
  GPU moves that point out, a longer context brings it back.
- Shapes are those of the four sweep models' files: their heads, Mistral 7B's FFN (the largest matrices of the
  sweep, in the types its Q4_K_M file holds), and Qwen2.5-3B's output matrix (the largest vocabulary; every sweep
  model's output is Q6_K). Weights and activations are seeded random values with finite scales.
- The probe's own device memory, sampled every other second, peaked at 152 MiB in the kit run; the sampling can miss
  the LM head's phase (about 260 MiB of weights), and either rounds up to the kit's 512 MiB minimum.
