# Packed K-quant prefill: end-to-end smoke, greedy agreement and determinism

**Purpose.** The packed K-quant prefill matmul checked end to end on the four sweep models (TinyLlama,
Qwen2.5-3B, Phi-3.5-mini, Mistral 7B, all Q4_K_M), both schedules, over `/v1/chat/completions`: a
512-token prompt and exactly 512 greedy tokens. Read for: correct output with the packed path active,
TTFT and peak device memory per cell, and greedy token agreement with the FP16 prefill route over 512
tokens. Also published here: the determinism study that found and confirmed the fix for a race in the
GPU attention kernel. Without that fix, greedy agreement could not be read at all.

**Pinned: no.** TTFT figures are single unpinned readings, for reading only. Token agreement and
determinism do not depend on clocks.

**Builds.**

| Side | Source | Shaded jar sha256 (first 16) |
|---|---|---|
| candidate | HEAD `59c53cc` plus the attention-kernel barrier fix (working tree) | `cff55ad0807c6ea7` |
| baseline (FP16 route) | HEAD `f02bdae` (the last build whose prefill dequantizes K-quant weights to FP16) plus only the same attention fix, from `git archive` | `463af3e52bc9801d` |
| determinism study, before the fix | HEAD `59c53cc` / HEAD `f02bdae` | `425285a3daef839e` / `c9f8e7a74c4187b7` |

Host: GTX 1080 (8 GiB), desktop session sharing the card; JDK 25. One engine at a time, 2026-10-04.

**Re-run on the closing build** (jar `5ab4c78c505a4cef`, which adds the decode GEMV scratch held at upload),
2026-10-04 08:17Z, with the script's restated rule (agreement recorded, not asserted): 48 of 48 checks pass,
and every agreeing prefix is the same as below. The decode change does not move greedy output.

**Command.** `scripts/performance-tests/smoke-packed-kquant-matmul.sh --baseline-jar <baseline jar>`:
`--local --gpu`, default `--gpu-layers`, `--mmq` and `--prefill-batch`, temperature 0, `min_tokens` =
`max_tokens` = 512. The prompt is calibrated per model to 507 to 512 prompt tokens. Exit 1: all 48
candidate checks passed, and the 8 agreement checks failed (below).

## Results

| Model | Schedule | TTFT ms (cand / base) | Peak VRAM MiB (cand / base) | Agreeing prefix of 512 | Positionwise |
|---|---|---|---|---|---|
| tinyllama | static | 593 / 1002 | 1208 / 1042 | 24 | 33 |
| tinyllama | continuous | 2074 / 3075 | 946 / 968 | 24 | 32 |
| qwen2.5-3b | static | 1119 / 1862 | 2578 / 2488 | 0 | 4 |
| qwen2.5-3b | continuous | 2782 / 6240 | 2360 / 2404 | 0 | 16 |
| Phi-3.5-mini | static | 2579 / 4277 | 3410 / 3242 | 66 | 68 |
| Phi-3.5-mini | continuous | 4366 / 8849 | 3018 / 3114 | 66 | 70 |
| mistral-7b | static | 2340 / 3866 | 5040 / 4878 | 68 | 74 |
| mistral-7b | continuous | 4519 / 12418 | 4634 / 4744 | 99 | 101 |

Candidate checks, every cell: HTTP 200; prompt tokens within 10% of 512; 512 tokens generated with one
stream chunk each; streamed text equal to unstreamed; packed weights resident, the tiled kernel loaded,
no dequantizing fallback in the server log; a device-memory reading. 48 of 48 pass.

**Every route parts from every other early** (`determinism/per-token-oracle-prefix.json`; static; first
differing token of 512; "per-token" is `--prefill-batch 1`, where prefill runs the decode kernels):

| Model | packed vs FP16 | packed vs per-token | FP16 vs per-token |
|---|---|---|---|
| tinyllama | 24 | 24 | 63 |
| qwen2.5-3b | 0 | 30 | 0 |
| Phi-3.5-mini | 66 | 66 | 244 |
| mistral-7b | 68 | 512 | 68 |

**Determinism study** (`determinism/tinyllama-repeat-study.json`; TinyLlama, 2 fresh servers x 5
identical requests per configuration):

| Configuration | Distinct outputs of 10 |
|---|---|
| `59c53cc`, before the fix | 4 (first differences at 24, 24, 94) |
| `f02bdae`, before the fix | 3 (46, 46) |
| `59c53cc`, `--prefill-batch 1` | 1 |
| `59c53cc`, `--gpu-attention off` | 1 |
| candidate, after the fix | 1 |
| `f02bdae` plus the fix | 1 |

## How to read it

- **The race.** The GPU attention kernel reduced twice through one shared-memory array without a barrier
  between the reductions, so output depended on warp timing. The determinism table placed it: it showed
  only with GPU attention on and a batched prefill window, and both builds had it. The fix is a barrier.
  After it both builds are deterministic. Every agreement figure above is taken after the fix, so it is
  the routes' real difference.
- **Greedy agreement at 512 tokens measures where the first near-tie flips, not which route is more
  accurate.** The packed route matches the per-token route for all 512 tokens on Mistral 7B, while the
  FP16 route parts from it at token 68. On Qwen2.5-3B the FP16 route parts from per-token prefill at the
  first token, which is two equally plausible openings ("Note 1: In the year 1813..." against "In the year
  1813..."). Neither route meets ">= 99% of 512 tokens" against per-token prefill, the same rounding decode
  uses for every token.
- **TTFT** falls 1.5x to 2.7x on the candidate. These are single unpinned readings; the throughput gates
  are the pinned A/Bs.
- **Peak VRAM** includes the 512 generated tokens. It is not the prefill-over-decode VRAM threshold,
  which `20261003T235146Z-packed-kquant-reserve` reads.
