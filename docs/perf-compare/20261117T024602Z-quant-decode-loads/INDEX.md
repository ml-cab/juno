# Quantized device KV, decode loop with up-front loads: kernel time and decode tg, unpinned

**Purpose.** Owner decision 9 (a) (2026-11-16): the quantized decode loop of the wide decode block takes up-front loads
the way the FP16 decode does, inside item 7, because the owner's gate
([`20261116T224500Z-fp16-wide-decode-gate`](../20261116T224500Z-fp16-wide-decode-gate/INDEX.md)) read `q8_0` decode
at 0.938x the faster FP16 decode on Qwen2.5-3B (bar 0.95, exit criterion 7). This entry is the unpinned A/B before the
owner's pinned gate (`dist/quant-decode-loads/run-gate.sh`).

**Pinned:** no. Desktop sharing the GPU and the CPU; one JVM at a time, nothing else measuring. **Host:** Xeon E5-1650 v2,
64 GiB, GTX 1080 8 GiB, sm_61. **Builds:** before = the working tree before this change (`637751b0` plus the
uncommitted FP16 wide decode), copied to scratch (tracked and untracked files; `git archive` cannot hold uncommitted
changes); after = that tree plus this change; both run from their trees' test classes through `mvn -o test -pl node -am`.

## 1. What changed (`gqa_decode_wide.cuh`, `gqa_attention_kv.cu`)

- `gqa_decode_kv_wide` (new): the wide decode block's quantized loop. Same keys per slot, same block-major lanes and the
  same arithmetic in the same order as `gqa_decode_kv`, with each key's K and V chunks loaded before the dot product
  (`GqaChunk::load`: the block's scale and quant words, no branch between them; a slot outside the window loads row `lo`,
  a lane past the head loads the head's first block, both discarded), where the old loop loaded V after the softmax
  update.
- `gqa_quad_floats` (new): a quad's four bytes (q8_0 offset to unsigned, or q4_0 nibbles) placed under the exponent of
  2^23 by one byte permute each, less 2^23 plus the offset: the old loop's floats exactly, without the shift and the
  integer-to-float conversion per value (`cuobjdump`: 64 `I2F` and 64 `SHR` a key and lane became 80 `PRMT` and 64
  `FADD`, all full rate).
- No branch per block or quad (with a head width a multiple of 32 a lane's chunk is wholly in the head or wholly past
  it): the old loop's 15 `SSY` and 32 `SYNC` divergence regions are gone.
- `runSum = fmaf(runSum, corr, w)` written out. The old loop left the fusing to the compiler, which fused it for four
  format pairs and not for K `q4_0` with V `f16`.
- Decode entries `__launch_bounds__(256, 2)`: the up-front loads asked for 144 registers at the 128-wide entry, one block
  a multiprocessor, and a first bench read Mistral 7B and Phi-3.5-mini 1.46x and 1.35x slower at 2,048 keys (32 heads on
  20 multiprocessors ran in two waves); capped at 128, no spills (`ptxas -v`; the 64-wide entry 94).
- PTX: `gqa_attention_kv.ptx`'s three tiled entries byte for byte the committed ones; `gqa_attention.ptx` unchanged.

**Outputs.** A digest test (scratch, removed) over 50 decode configurations (5 shapes, 5 format pairs, windows 0 and 100,
four rows each): 45 bit for bit the old loop's; the five K `q4_0` / V `f16` configurations at window 0 differ by at most 2
ulp (119 of 2,048 values of a 1,000-key row), from the `runSum` fusing above.

## 2. Kernel time at decode width (`kernel-bench.txt`)

`GqaAttentionDecodeBench`, one query row, device time per launch behind a busy prefix, median of 5 x 100 launches; four
runs alternating before, after, before, after; means of each tree's two runs.

| Shape | `q8_0`/`q8_0` at 2,048 keys, before / after (us) | after over before | `q8_0`/`q4_0` after over before | `q8_0`/`q8_0` over FP16, before / after |
|---|---|---|---|---|
| TinyLlama 32/4/64 | 37.9 / 29.5 | **0.780** | 0.724 | 1.240 / 0.974 |
| Qwen2.5-3B 16/2/128 | 39.2 / 28.8 | **0.735** | 0.660 | 1.081 / 0.897 |
| Phi-3.5-mini 32/32/96 | 82.9 / 81.8 | **0.987** | 1.007 | 0.653 / 0.651 |
| Mistral 7B 32/8/128 | 56.3 / 51.6 | **0.915** | 0.832 | 1.000 / 0.900 |

At 256 to 1,024 keys after over before reads 0.82 to 0.97 except Phi-3.5-mini (1.04, 1.00, 1.10): its 96-wide heads
leave one lane of four without a chunk, which the old loop skipped by a branch and the new one runs on a cached copy of
the head's first block. The 64-key cells (6 to 10 us, and TinyLlama's `q8_0` cell reading 17 to 38 us in either tree,
noted on 2026-11-16) vary by more than the change.

## 3. Decode tg at a 2,048-token context (`decode-tg-unpinned.txt`)

`DeviceKvDecodeTgLiveTest`: prefill 2,048 tokens, 128 greedy decode steps through the serving decode path, one warm-up
and three measured repetitions, a JVM's reading the median. Per model A1 B1 F1 A2 B2 F2 (A `q8_0` before, B `q8_0`
after, F `f16`, whose kernel this change does not touch); means of two JVMs.

| Model | `q8_0` before (A) | `q8_0` after (B) | **B over A** | `f16` (F) | **B over F** (bar 0.95 when pinned) | A over F |
|---|---|---|---|---|---|---|
| TinyLlama 1.1B | 147.9, 149.9 | 155.4, 155.7 | **1.045** | 156.6, 160.0 | **0.983** | 0.941 |
| Qwen2.5-3B | 58.9, 57.9 | 60.3, 58.0 | **1.012** | 62.6, 59.3 | **0.971** | 0.959 |
| Phi-3.5-mini | 46.2, 43.8 | 43.3, 43.2 | 0.961 | 44.3, 44.4 | **0.975** | 1.014 |
| Mistral 7B | 34.2, 34.8 | 35.1, 35.4 | **1.024** | 36.4, 35.9 | **0.976** | 0.953 |

Phi-3.5-mini, three more pairs (A B x3): 46.15, 43.75, 43.65 against 43.47, 43.27, 43.56, median B over A **0.994**. Its
first JVM after an idle card reads high on both occasions (46.2): the card starts cool and throttles later (Phi-3.5-mini
runs at 93 to 94 C with software thermal slowdown on about 80% of loaded samples, per the gate). The pinned gate cools to
65 C before every JVM.

## 4. The hot-path gate (`lora-baseline.json`, `lora-candidate.json`)

`compare-lora.sh --gpu --skip-build --no-publish --reps 3` from each tree, its jar built there (`mvn -o package
-DskipTests`), medians of three. Bars: train <= 1.25x, playback >= 0.80x.

| | Baseline | Candidate | Ratio |
|---|---|---|---|
| Train total | 41,000 ms | 41,000 ms | 1.000 |
| Train per pass | 2,733 ms | 2,733 ms | 1.000 |
| Playback | 12.774 tok/s | 12.456 tok/s | 0.975 |
| Recall | "My name is Juno" | "My name is Juno" | equal |

Pass. LoRA training and `--lora-play` attend on the CPU, so the gate does not reach the changed kernel; it holds that
nothing else moved (playback's JFR token rate reads 39.17 against 37.58).

## Files

`kernel-bench.txt` (section 2, every run), `decode-tg-unpinned.txt` (section 3, both runs and the three Phi-3.5-mini
pairs), `lora-baseline.json` and `lora-candidate.json` (section 4; local paths replaced by placeholders).

## How to read it

The quantized decode launch is 0.74x to 0.99x its previous time at 2,048 keys and now under the FP16 launch on every
sweep model, and `q8_0` decode tg at a 2,048-token context reads 0.971x to 0.983x the FP16 rate unpinned, where the old
loop read 0.94x to 0.96x on the other three models (0.938x pinned on Qwen2.5-3B). The scored reading is the owner's pinned gate: per model A B
three times, `q8_0` over FP16 >= 0.95.
