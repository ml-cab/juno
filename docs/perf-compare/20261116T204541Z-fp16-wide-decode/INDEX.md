# FP16 decode attention on the wide decode block: kernel time and decode tg, unpinned

**Purpose.** Owner decision 5 (a) (2026-11-15): the FP16 decode attention takes the wide decode block the quantized
device KV's decode runs (256 threads, 64 slots of 4 lanes), as a change outside item 7 with its own A/B against
today's FP16 kernel and the hot-path gate. A decode launch has one block per head and row; with the 128-thread tiled
kernel each warp scheduler had one warp (Qwen2.5-3B: 16 blocks on 20 multiprocessors). This entry is the unpinned A/B
before the owner's pinned gate (`dist/fp16-wide-decode/run-gate.sh`).

**Pinned:** no. Desktop sharing the GPU and the CPU; one JVM at a time, nothing else measuring. **Host:** Xeon E5-1650 v2,
64 GiB, GTX 1080 8 GiB, sm_61. **Builds:** baseline = commit `637751b0` from `git archive` in scratch (the FP16 decode on
the tiled kernel); candidate = `637751b0` plus this change, both run from their trees' test classes through
`mvn -o test -pl node -am`.

## 1. What changed

- `gqa_decode_wide.cuh` (new): the wide decode body, moved out of `gqa_attention_kv.cu` and given an FP16 variant. The
  FP16 variant keeps the tiled kernel's interleaved lanes (quads `lane, lane + 4, ...`), so every lane of a 96- or
  80-wide head has work, and loads each key's and value's quads up front: one 8-byte read-only load a quad, no branch
  between them (a slot outside the window reads row `lo` and discards it). Same arithmetic, in the same order, as the
  tiled kernel's decode loop, at a stride of 64 keys, merged over 64 slots.
- `gqa_attention.cu`: entries `gqa_decode_d64` and `gqa_decode_d128`. `GqaAttentionKernel.launch` sends rows of one at
  head widths up to 128 (`wideDecode`) to them, through a reused `KernelParams` block (no allocation per launch, where
  the tiled launch builds an arena); 256-wide heads and prefill tiles keep the tiled entries.
- PTX: the three tiled entries of `gqa_attention.ptx` are byte for byte the committed ones (the diff only adds lines);
  `gqa_attention_kv.ptx` is byte for byte the committed one apart from the shared-variable names, which gained the new
  template argument. New entries: 128 registers (`_d128`), 78 (`_d64`), no spills (`ptxas -v`).
- A first version read each quad as two 4-byte generic loads, each behind its own branch, and read 1.04x to 1.05x the
  tiled kernel at 2,048 keys on Phi-3.5-mini and Mistral 7B. The 8-byte loads changed no output bit (a SHA-256 of 80
  decode outputs over five shapes and two windows, equal before and after) and took it to the figures below.

## 2. Kernel time at decode width (`kernel-bench.txt`)

`GqaAttentionDecodeBench`, one query row, device time per launch behind a busy prefix, median of 5 x 100 launches.
`current` is the shipped launch (the wide block); `tiled` is the same module's 128-thread tiled entry launched at one row
a block, which is what every FP16 decode row ran before (its PTX is unchanged).

| Shape | 64 keys | 256 | 576 | 1,024 | 2,048 keys: current, tiled |
|---|---|---|---|---|---|
| TinyLlama 32/4/64 | 0.938 | 0.583 | 0.476 | 0.401 | **0.295** (30.6 against 103.6 us) |
| Qwen2.5-3B 16/2/128 | 0.847 | 0.511 | 0.522 | 0.567 | **0.434** (32.4 against 74.7 us) |
| Phi-3.5-mini 32/32/96 | 0.783 | 0.690 | 0.629 | 0.715 | **0.684** (129.3 against 188.9 us) |
| Mistral 7B 32/8/128 | 0.813 | 0.583 | 0.504 | 0.492 | **0.499** (65.3 against 130.8 us) |

Every cell is under 1.0. Launches of 6 to 10 us vary by more than 10% between runs (an earlier run of the final kernel
read 1.27 at TinyLlama's 64 keys, this one 0.94). The `[f16 block-major]` lane (the quantized module's wide decode over
the same FP16 rows, each lane a contiguous chunk of the head) reads 1.19x to 2.84x the shipped launch at 2,048 keys. The
quantized lanes' first cell (TinyLlama, 64 keys, `q8_0`/`q8_0`, about 27 us) read the same in every run of this bench
today, before and after the change; not investigated here.

## 3. Decode tg at a 2,048-token context (`decode-tg-unpinned.txt`)

`DeviceKvDecodeTgLiveTest`: prefill 2,048 tokens, 128 greedy decode steps through the serving decode path, one warm-up
and three measured repetitions, a JVM's reading the median. Per model A1 B1 C1 A2 B2 (A f16 baseline, B f16 candidate,
C `q8_0` candidate); the ratio is the mean of the two JVMs per side.

| Model | FP16 before (A1, A2) | FP16 after (B1, B2) | **after over before** | `q8_0` after (C1) | `q8_0` over FP16 after |
|---|---|---|---|---|---|
| TinyLlama 1.1B | 119.7, 121.8 | 158.5, 155.9 | **1.302** | 152.2 | 0.968 |
| Qwen2.5-3B | 57.0, 55.4 | 62.5, 60.0 | **1.091** | 58.5 | 0.955 |
| Phi-3.5-mini | 43.3, 40.5 | 44.9, 44.3 | **1.065** | 43.6 | 0.977 |
| Mistral 7B | 32.9, 34.0 | 36.1, 36.0 | **1.078** | 35.2 | 0.976 |

Every B JVM is faster than every A JVM on every model.

## 4. What it does to item 7's decode bar

Item 7's decode bar (`q8_0` decode tg >= 0.95x the FP16 device copy's, exit criterion 7) was met pinned on 2026-11-15
at 1.03x to 1.23x ([`20261115T200324Z-device-kv-tg-gate-rerun`](../20261115T200324Z-device-kv-tg-gate-rerun/INDEX.md)), against the
FP16 decode on the tiled kernel. Part of that margin was the wide block itself, which FP16 now has too: against the
faster FP16 decode, `q8_0` reads 0.955x to 0.977x here, unpinned, Qwen2.5-3B nearest the bar. The owner's gate re-scores
it (lane C, bar 0.95). That gate result supersedes the 2026-11-15 re-run's ratios; the `q8_0` side's own speed is unchanged
(its PTX is the committed one).

## 5. The hot-path gate (`lora-baseline.json`, `lora-candidate.json`)

`compare-lora.sh --gpu --skip-build --no-publish --reps 3` from each tree, its jar built there (`mvn -o package
-DskipTests`), medians of three. Bars: train <= 1.25x, playback >= 0.80x.

| | Baseline | Candidate | Ratio |
|---|---|---|---|
| Train total | 41,000 ms | 41,000 ms | 1.000 |
| Train per pass | 2,733 ms | 2,733 ms | 1.000 |
| Playback | 12.915 tok/s | 12.891 tok/s | 0.998 |
| Recall | "My name is Juno" | "My name is Juno" | equal |

Pass. LoRA training and `--lora-play` attend on the CPU, so the gate does not reach the changed kernel; it holds that
nothing else moved.

## Files

`kernel-bench.txt` (section 2), `decode-tg-unpinned.txt` (section 3, every repetition of every JVM),
`lora-baseline.json` and `lora-candidate.json` (section 5; local paths replaced by placeholders).

## How to read it

On this host the FP16 decode attention is 0.30x to 0.68x of its old time at 2,048 keys and under 1.0 at every context
measured, and FP16 decode tg rises 6.5% to 30% at a 2,048-token context, most where attention is the largest share of
a token (TinyLlama). The scored A/B is the owner's pinned gate: per model A B C three times, bars FP16 after over before
>= 1.00 and `q8_0` over FP16 after >= 0.95.
