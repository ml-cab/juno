# Quantized device KV: the decode path after the gate (KV write, decode attention, FFN GEMV placement)

**Purpose.** The owner's gate ([`20261113T215326Z-device-kv-tg-gate`](../20261113T215326Z-device-kv-tg-gate/INDEX.md))
read Qwen2.5-3B at 0.918x the FP16 device copy's decode rate with `q8_0`, against a bar of 0.95x, and traced the cost to
three places on the GPU. This run reads each after the work on it: the KV write, the decode attention launch, and the
FFN gate and up GEMVs.

**Pinned:** no. Every reading here is unpinned (desktop sharing the GPU) or a per-kernel device time from an `nsys` CUDA
trace; the decode-rate bar is scored again by the owner's run (`dist/device-kv-tg/run-gate.sh`).

**Host:** Xeon E5-1650 v2, 64 GiB, GTX 1080 8 GiB. **Build:** HEAD `79fa87b0` plus the uncommitted quantized device KV
work, with the two kernel changes below; the experiments in section 3 ran on scratch copies of this tree.

## 1. What changed

| Cost on Qwen2.5-3B (per token, traced) | Before | Change | After |
|---|---|---|---|
| KV write: `kv_encode_rows`, two launches a layer, one thread per 32-value block (8 threads a launch) | 340 us (FP16 cast: 125 us) | `kv_encode_kv_rows`: K and V in one launch, one warp per block, the block's largest magnitude reduced across the warp with the scalar scan's comparisons; the decode batch's `kv_append_rows_fmt` encodes the same way | **139 us** |
| Decode attention: one block of 128 threads per head and row, 16 blocks on Qwen2.5-3B's 16 heads, one warp per scheduler | 3,256 us (FP16: 3,018 us) | `gqa_decode_kv_d64/_d128`: decode rows on 256-thread blocks, 64 slots of 4 lanes, the same per-slot loop at a 64-key stride and the same merge over 64 slots, 120 registers (the shared kernel: 226) | **1,958 us** |
| FFN gate and up GEMVs | +5.6% on FP16's | none (section 3: placement, not the quantized path) | +4.4% to +4.7% |

Both kernel changes keep the bytes and the guarantees: the encoder writes the host codec's bytes exactly (tested, and two
planted defects fail the tests: `q4_0`'s tie keeping the last index, the high nibble from the wrong lane); a decoded row
depends on its own cache and length only, so a row in a decode batch is still bit for bit its single decode
(`ResidentBatchDecodeTest`); a planted merge defect in the wide decode fails three parity tests. The FP16 kernels' PTX is
byte for byte the committed one. A third change was tried and not kept: loading each key's K and V one iteration ahead
in the decode loop read 3,295 us against 3,256 (`qwen-trace-breakdown.txt`); the loop is bound by too few warps, not by
its loads.

## 2. Readings

**Decode attention alone** (`decode-kernel-bench.txt`, one decode row per launch, KV warm in L2), `q8_0` K and V over the
FP16 kernel at 2,048 keys: TinyLlama **0.337x**, Qwen2.5-3B **0.568x**, Phi-3.5-mini **0.453x**, Mistral 7B **0.504x**
(the kernel before: 0.576x, 0.955x, 0.604x, 0.825x). At 64 keys the 64-slot merge costs more than the split saves on
Phi-3.5-mini (1.19x of a 7.9 us launch).

**Qwen2.5-3B per token, traced** (`qwen-trace-breakdown.txt`, the same FP16 JVM as side A throughout):

| | FP16 | `q8_0` as gated | `q8_0` now |
|---|---|---|---|
| GPU busy | 17,330 us | 18,173 us (+843) | **16,537 us (-793)** |
| wall (traced) | 23,208 us | 23,888 us | 22,386 us |

**Decode rate, unpinned** (`decode-tg-unpinned.txt`, f16 and `q8_0` JVMs alternated):

| Model | FP16 tg | `q8_0` tg | Ratio | Gated ratio (pinned, before) |
|---|---|---|---|---|
| Qwen2.5-3B | 56.545, 57.287 | 59.034, 58.621 | **1.044, 1.023** | 0.918 |
| TinyLlama 1.1B | 124.220 | 152.122 | 1.225 | 1.085 |
| Mistral 7B | 34.582 | 35.359 | 1.022 | 0.999 |
| Phi-3.5-mini | 43.154 | 43.805 | 1.015 | 1.004 |

**Quality** (`quality-recheck.txt`): the bar's reading, teacher-forced greedy agreement at prefill width, and the NLL are
unchanged on TinyLlama (99.41%, 1.00047) and Qwen2.5-3B (99.61%, 1.00112): prefill windows run the unchanged tiled
kernel and the encoder writes the same bytes. The free-running reply, which decodes through the wide kernel, now shares
28 and 33 of 512 tokens with the FP16 reply (1 and 1 before).

## 3. The FFN GEMVs: placement, not the quantized path (`gemv-placement.txt`)

The FFN gate and up GEMVs (Q4_K, 11,008 x 2,048: unchanged code, identical launches) ran 5.6% slower in every `q8_0` JVM.
Scratch builds of this tree, one variable at a time:

- An FP16 JVM that also runs the quantized encoder and attention on a shadow `q8_0` cache: 0.990x to 0.993x. Not the
  quantized kernels.
- An FP16 JVM that loads the quantized module before its weight upload: 0.994x to 0.997x. Not the weights' placement.
- A `q8_0` JVM whose mirrors are allocated at the FP16 size: **0.989x to 0.992x**, against 1.039x to 1.042x as allocated.
  The smaller mirrors move every buffer allocated after them.
- Of those buffers, the q8_1 input vector the gate and up GEMVs read (2,304 bytes; each of the 11,008 blocks reads all of
  it): moved to a 64 KiB boundary, 1.009x to 1.013x; the GEMV output scratch moved instead, 1.068x to 1.072x.
- The vector's offset from a 64 KiB boundary: 0x800 **0.997x**; 0x0 1.011x; 0x200 1.062x; 0x400 1.035x; 0xe00 1.029x;
  0x1000 1.028x; 0x1800 1.035x. The FP16 JVM's vector sits at 0x4800, the `q8_0` JVM's at 0xbe00. No alignment rule
  covers the fast offsets: the cost follows how the vector's cache lines map onto the card's L2, which a rule written for
  this card would not carry to another.
- Weights loaded L2-only (`__ldcg`) in `q4k_gemv`: the FP16 GEMVs 3.7% faster (0.961x to 0.964x), the placement cost
  unchanged (1.059x to 1.062x).

So the FFN GEMVs' speed depends on where an allocation lands, by up to about 6% on this card, for either format: FP16
drew a fast offset on Qwen2.5-3B and `q8_0` a slow one. It is a property of the GEMV, not of the device KV, and is left
for a decision (the Tier 03 record, 2026-11-14). The `q8_0` decode passes the bar with it in place.

## How to read it

Section 2's rates are unpinned and serve as evidence; the bar is scored by the owner's re-run of
`dist/device-kv-tg/run-gate.sh` (graphics clock locked at 1,300 MHz, the card cooled to 65 C before each JVM, a CLOCK line
per JVM; the lock turned out not to apply on this card, see the re-run [`20261115T200324Z-device-kv-tg-gate-rerun`](../20261115T200324Z-device-kv-tg-gate-rerun/INDEX.md), where every sweep model passes). The `q8_0` decode now spends less GPU time per token than FP16 on every sweep model; the FP16 kernels are
unchanged, so part of that margin is the wider decode block, which an FP16 launch could also take.
