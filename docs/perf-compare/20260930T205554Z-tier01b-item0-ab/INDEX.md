# GPU attention on the Phi-3 and Qwen3 handlers: pinned same-build flag A/B

The gate reading for the GPU attention kernel in the Phi-3 and Qwen3 handlers: one build, the only
difference between lanes `--gpu-attention`, alternated off, on, off, on, off, on in the same hour.
`off` is the scalar CPU attention these two handlers ran before the change; `on` is the kernel over an
FP16 KV mirror.

- Build: working tree on `ffd0ca7` with the change, shaded jar sha256 `2564f8f506273981` (all twelve
  runs, and the two default sweeps `20260930T211637Z` and `20260930T215547Z`).
- Clocks pinned on every run: CPU governor performance, turbo off, GPU graphics clock locked at 1911
  MHz. JFR on the operating-system clock (kernel clocksource `hpet`). GTX 1080, driver 580.173.02.
  Reference tool build `ac4cdde`.
- Harness, per run: `compare-llama-cpp.sh --gpu --pin-clocks --models Phi-3.5-mini,Qwen3-1.7B
  --n-prompt {512|128} --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish
  --gpu-attention {off|on}`. So each side's figure is a median of three runs, one repetition each.
- Every row scorable; every Juno prefill 512/512 or 128/128 tokens; no repetition withheld.
- Runs 01 to 06 at `n_prompt=512` (2026-09-30 20:55 to 21:16 UTC), 07 to 12 at 128 (21:46 to 21:55 UTC).
  Resolved GPU attention reads `unknown` (no `--device-spans`); the kernel being taken is shown by
  decode moving with it and by `GpuAttentionHandlerParityTest`.

Juno t/s, median of three (min to max):

| Model | `n_prompt` | pp off | pp on | pp gain | tg off | tg on | tg gain |
|---|---|---|---|---|---|---|---|
| Phi-3.5-mini Q4_K_M | 512 | 13.69 (12.96 to 13.75) | 89.71 (89.42 to 89.83) | **6.55x** | 24.08 (23.65 to 24.09) | 32.03 (31.84 to 32.35) | 1.33x |
| Qwen3-1.7B Q4_K_M | 512 | 28.80 (28.32 to 29.07) | 175.56 (175.46 to 176.05) | **6.10x** | 34.96 (34.47 to 35.19) | 38.43 (38.23 to 39.05) | 1.10x |
| Phi-3.5-mini Q4_K_M | 128 | 44.63 (44.43 to 44.65) | 98.07 (96.46 to 98.16) | **2.20x** | 24.20 (24.14 to 24.35) | 31.87 (31.72 to 32.62) | 1.32x |
| Qwen3-1.7B Q4_K_M | 128 | 82.12 (81.94 to 82.41) | 177.72 (173.02 to 177.91) | **2.16x** | 34.87 (34.73 to 35.19) | 38.63 (38.48 to 38.83) | 1.11x |

Every spread is under 3% of its median. The gain grows with prompt length because scalar attention
is quadratic in it: with the kernel, prefill t/s is nearly flat from 128 to 512 tokens (Phi-3.5-mini
98.1 to 89.7, Qwen3-1.7B 177.7 to 175.6), where without it Phi-3.5-mini fell 3.3x.

Ratios against the reference tool in the same runs (Juno/reference, median):

| Model | pp @128 off / on | pp @512 off / on | tg (on) |
|---|---|---|---|
| Phi-3.5-mini Q4_K_M | 0.0383 / 0.0847 | 0.0112 / 0.0723 | 0.52 to 0.54 |
| Qwen3-1.7B Q4_K_M | 0.0348 / 0.0730 | 0.0103 / 0.0629 | 0.33 to 0.34 |

The same build's published default sweeps, `20260930T211637Z` (128) and `20260930T215547Z` (512), are
item 0's ratio reading for the four standard models against the step 2 reference.
