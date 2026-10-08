# GPU attention at decode width: keys read straight from global memory

**Purpose.** The closing gate (`20261008T054413Z-attention-close-gate`) found Qwen2.5-3B generation at 0.948x the
pre-tier build and attributed it to the attention kernel at decode width. These readings measure that kernel at one
query row, before and after the fix, against the pre-tier kernel, and read Qwen2.5-3B end to end on all three builds.

**Pinned: no.** Kernel timings are device time (CUDA events); the end-to-end reading is indicative. The scored
reading is the owner's pinned re-run of the gate.

**The change.** At one query row per block (decode, and `--parallel` streams) the tiled kernel staged every
32-key tile into shared memory between two barriers and then gave each of its 32 slots one key. Nothing else
shares a decode row's keys, so the staging bought nothing and serialised the loads. The decode path now has each
slot read its keys straight from global memory: the same keys, in the same order, with the same arithmetic, so the
output is **bit-identical** to the previous kernel (200 decode launches over five head shapes, contexts 1 to 4,097
and windows 0, 1, 33 and 100, compared bit for bit; a one-ulp planted fault fails the comparison at the first
launch). Prefill windows of more than one row are unchanged. It also fixes 256-wide heads at decode, where the
16-key tile gave a slot no key and the output was NaN; no supported model has 256-wide heads.

**Builds.**

| Name | Source | Shaded jar sha256 (first 16) |
|---|---|---|
| pre-tier | HEAD `05a17de` (full-materialization kernel) | `d1f50669dfc23d55` |
| tiled, before | HEAD `51fa5d4` plus test-only changes | `fefd30371caf535a` |
| decode, after | the same plus the decode path | `43b22345f1175c8f` |

## Kernel at decode width (device us per launch)

`GqaAttentionDecodeBench` (node test scope; `mvn test -pl node -Dgroups=gpu -Dtest=GqaAttentionDecodeBench
-Djuno.attentionBench=true [-Djuno.attentionBench.oldPtx=<pre-tier gqa_attention.ptx>]`): one query row, keys
written once, median of five repetitions of 100 back-to-back launches queued behind 200 long launches so the device
never waits on the host. "Pre-tier" is the pre-tier kernel's PTX loaded in the same process.

| Shape | Keys | Pre-tier | Tiled, before | Decode, after | After vs before | After vs pre-tier |
|---|---|---|---|---|---|---|
| tinyllama 32/4/64 | 64 | 14.7 | 27.8 | 7.8 | 3.6x faster | 1.9x faster |
| tinyllama | 576 | 98.9 | 193.2 | 21.9 | 8.8x | 4.5x |
| tinyllama | 2048 | 472.5 | 710.8 | 87.1 | 8.2x | 5.4x |
| qwen2.5-3b 16/2/128 | 64 | 18.1 | 36.4 | 7.1 | 5.1x | 2.5x |
| qwen2.5-3b | 576 | 102.5 | 275.3 | 19.2 | 14.4x | 5.3x |
| qwen2.5-3b | 2048 | 514.0 | 1094.8 | 60.9 | 18.0x | 8.4x |
| phi-3.5-mini 32/32/96 | 64 | 17.8 | 34.5 | 7.8 | 4.4x | 2.3x |
| phi-3.5-mini | 576 | 171.1 | 296.0 | 53.3 | 5.6x | 3.2x |
| phi-3.5-mini | 2048 | 603.9 | 1025.5 | 173.8 | 5.9x | 3.5x |
| mistral-7b 32/8/128 | 64 | 20.9 | 36.5 | 7.9 | 4.6x | 2.6x |
| mistral-7b | 576 | 181.5 | 325.0 | 33.7 | 9.6x | 5.4x |
| mistral-7b | 2048 | 635.4 | 1126.8 | 107.1 | 10.5x | 5.9x |

Every context (64, 256, 576, 1024, 2048): `bench-tiled-before.txt`, `bench-decode-after.txt`.

## Qwen2.5-3B end to end (unpinned)

`compare-llama-cpp.sh --gpu --models qwen2.5-3b-instruct-q4_k_m --n-prompt 512 --juno-warmup 2 --juno-reps 1 --reps 1
--no-tuned-lane --no-publish --juno-jar <jar>`, the three builds alternated three times (`qwen-<build>-<n>/`).

| Build | tg t/s (3 runs) | Median | Decode ms per token (median) | tg against pre-tier |
|---|---|---|---|---|
| pre-tier | 28.58 / 28.78 / 27.08 | 28.58 | 33.40 | 1.000 |
| tiled, before | 28.01 / 27.65 / 25.60 | 27.65 | 34.59 | 0.967 |
| decode, after | 29.51 / 28.83 / 26.73 | 28.83 | 33.05 | 1.009 |

The third repetition is lower on every build (the desktop shared the card). The generation lane decodes from a short
context, where attention is a small part of a step, so the end-to-end gain is small; the kernel table shows where it
grows.

## How to read it

- The earlier regression was not Qwen2.5-3B's alone: the tiled kernel was 1.5x to 2.9x slower than the pre-tier
  kernel at decode on every shape, and the decode residency region's gains hid it on the other three models.
- Local paths are replaced by `<repo>/`.
