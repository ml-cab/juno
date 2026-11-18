# FFN gate and up projections in one launch: kernel time at nine input placements, decode tg A/B and the LoRA gate (unpinned)

**Purpose.** Owner decision 4 (a) (2026-11-15): the decode region's FFN gate and up GEMVs read their Q8_1 input once
for two rows. Their speed moved by up to about 6% with where that 2,304-byte vector landed in device memory
([`20261115T045828Z-device-kv-decode-path`](../20261115T045828Z-device-kv-decode-path/INDEX.md), section 3). Section 1
is the kernel half of the change's thresholds: on each sweep model's gate/up shape, at each of nine offsets of the
vector from a 64 KiB boundary, the paired launch's device time **<= 1.00x** the two single launches' at the same offset,
with each side's spread across offsets reported. Section 2 is the executor's unpinned reading of the decode-rate half
(FP16 and `q8_0` device KV each **>= 1.00x** the build before); the pinned score is the owner's gate. Section 3 is the
hot-path gate (`compare-lora.sh`).

**Pinned:** no (the executor's reading; the GTX 1080 is shared with the desktop). **Host:** Xeon E5-1650 v2, GTX 1080.
**Build:** HEAD `9c33ab93` plus the uncommitted change (`q4k_gemv_pair.cu`, `Q4KGemvPairKernel`, the tail's route).
**Baseline (sections 2 and 3):** `git archive 9c33ab93`, built in its own directory (`mvn -o package -DskipTests`). Shaded
jars: candidate 37,105,899 bytes (carries `q4k_gemv_pair.ptx`), baseline 37,098,373 bytes (does not).

**What changed.** `q4k_gemv_pair` computes gate row r and up row r in one block: a lane loads its slice of the Q8_1
vector, block sums and scales once and dots them with both rows. Where a row has at most 8 super-blocks (width up to
2,048: TinyLlama, Qwen2.5-3B), the two warps a single row leaves idle take the next row index, so a block computes two
gate and two up rows. Each row's arithmetic is the single-row kernel's, written with explicit rounding in the form that
kernel's machine code has, so the outputs are bit for bit two `q4k_gemv` launches (`Q4KGemvPairTest`).

## 1. Kernel time at nine placements of the input (`bench-run1.txt`, `bench-run2.txt`)

`Q4KGemvPairBench` (`-Djuno.gemvPairBench=true`): random Q4_K weights of each shape (fixed finite scales), the input
quantized once at each offset, CUDA events around 50 calls after 100 unmeasured ones, seven repetitions alternating the
two sides, the median per side. Two full runs.

Paired over two launches, worst offset per shape (all nine offsets are under 1.00 in both runs):

| Shape | Run 1 worst | Run 2 worst | Two launches, slowest over fastest offset (run 1 / 2) | Paired, slowest over fastest (run 1 / 2) |
|---|---|---|---|---|
| TinyLlama 5,632 x 2,048 | 0.943 | 0.856 | 1.263 / 1.284 | 1.209 / 1.171 |
| Qwen2.5-3B 11,008 x 2,048 | 0.849 | 0.884 | 1.088 / 1.158 | 1.057 / 1.094 |
| Phi-3.5-mini 8,192 x 3,072 | 0.973 | 0.960 | 1.033 / 1.038 | 1.107 / 1.069 |
| Mistral 7B 14,336 x 4,096 | 0.969 | 0.964 | 1.035 / 1.033 | 1.009 / 1.015 |

Per offset, run 1 (microseconds per gate-and-up, two launches against paired): Qwen2.5-3B 158 to 172 against 132 to 140;
Mistral 7B 350 to 362 against 336 to 339; every line is in `bench-run1.txt` and `bench-run2.txt`.

## 2. Decode tg at a 2,048-token context (`decode-tg-unpinned.txt`)

`DeviceKvDecodeTgLiveTest`: prefill 2,048 tokens, 128 greedy decode steps through the serving decode path, one warm-up
and three measured repetitions, a JVM's reading the median of the three. A = the baseline tree, B = this tree; per model
and device KV type A B A B (Mistral 7B A B four times), each JVM run from its own tree with `mvn -o test -pl node -am`,
the card cooled to 65 C before each JVM. The two JVMs of a pair ran within 13 MHz of each other's median loaded SM clock
(samples above 70 W), most at the same clock, except Mistral 7B's third `q8_0` pair (A 1,873 MHz, B 1,847 MHz).

| Model | Type | A, tok/s | B, tok/s | **B over A** (means) | B over A per pair |
|---|---|---|---|---|---|
| TinyLlama 1.1B | `f16` | 154.04, 154.59 | 162.22, 164.92 | **1.060** | 1.053, 1.067 |
| TinyLlama 1.1B | `q8_0` | 154.26, 147.19 | 160.79, 164.01 | **1.077** | 1.042, 1.114 |
| Qwen2.5-3B | `f16` | 63.11, 61.22 | 65.71, 64.17 | **1.045** | 1.041, 1.048 |
| Qwen2.5-3B | `q8_0` | 58.60, 59.82 | 59.75, 63.49 | **1.041** | 1.020, 1.061 |
| Phi-3.5-mini | `f16` | 46.67, 45.57 | 47.68, 47.16 | **1.028** | 1.022, 1.035 |
| Phi-3.5-mini | `q8_0` | 46.33, 46.30 | 46.62, 47.02 | **1.011** | 1.006, 1.015 |
| Mistral 7B | `f16` | 36.92, 37.55, 36.67, 37.76 | 38.33, 36.39, 37.43, 37.95 | **1.008** (medians 1.012) | 1.038, 0.969, 1.021, 1.005 |
| Mistral 7B | `q8_0` | 36.98, 36.75, 37.18, 36.70 | 37.47, 36.56, 37.31, 37.02 | **1.005** (medians 1.008) | 1.013, 0.995, 1.004, 1.009 |

On TinyLlama (both types), Phi-3.5-mini (both types) and Qwen2.5-3B `f16`, every B JVM is faster than every A JVM; on
Qwen2.5-3B `q8_0` the slowest B reads 0.999x the fastest A. Mistral 7B's two sides overlap: two more pairs were run after
the first two read 1.003x and 1.004x, and the four pairs read 1.008x and 1.005x, one pair of each type under 1.0.

## 3. The hot-path gate (`lora-baseline.json`, `lora-candidate.json`)

`compare-lora.sh --gpu --skip-build --no-publish --reps 3` from each tree, its jar built there, medians of three. Bars:
train <= 1.25x, playback >= 0.80x.

| | Baseline | Candidate | Ratio |
|---|---|---|---|
| Train total | 42,000 ms | 41,000 ms | 0.976 |
| Train per pass | 2,800 ms | 2,733 ms | 0.976 |
| Playback | 12.704 tok/s | 12.613 tok/s | 0.993 |
| Recall | "My name is Juno" | "My name is Juno" | equal |

Pass. LoRA training and `--lora-play` do not run the decode region, so the gate does not reach the changed kernel; it
holds that nothing else moved (playback's JFR token rate reads 39.64 against 39.09).

## Files

`bench-run1.txt`, `bench-run2.txt` (section 1), `decode-tg-unpinned.txt` (section 2, every JVM with its start
temperature, loaded clock and three repetitions), `lora-baseline.json` and `lora-candidate.json` (section 3; local paths
replaced by placeholders).

## How to read it

The kernel bar is met on every shape at every offset, in both runs. Across both runs and all offsets the paired launch
reads 0.746x to 0.884x the two launches on Qwen2.5-3B and 0.632x to 0.943x on TinyLlama, at width 2,048 where it also puts
the idle warps to work, and 0.876x to 0.973x (Phi-3.5-mini) and 0.935x to 0.969x (Mistral 7B) at 3,072 and 4,096, where
only the input reads halve. The pairing does not remove the placement sensitivity: the paired launch still moves
with the offset (1.009x to 1.209x slowest over fastest), and on Phi-3.5-mini its spread is wider than the two launches'. TinyLlama's
launches are short (66 to 107 us) and its readings scatter most on the shared card.

Decode follows the kernel: the gain is largest where the paired launch also fills the idle warps (TinyLlama 1.06x to
1.08x, Qwen2.5-3B 1.04x) and smallest at width 4,096, where the gate and up launches save 3% to 6.5% of their own time and
Mistral 7B reads 1.005x to 1.008x, inside its JVM-to-JVM spread. Unpinned, these figures do not score the decode bar; the
owner's pinned gate (A `f16` before, B `f16` after, C `q8_0` before, D `q8_0` after, three cycles) does, and Mistral 7B's
margin there is expected to be about 1%.
