# Quantized device KV, decode loop with up-front loads, pinned gate: `q8_0` decode >= 0.95x FP16 on every sweep model

**Purpose.** The owner's pinned gate for decision 9 (a) (2026-11-16): the quantized device KV's decode loop loads each
key's K and V blocks up front, unpacks quant bytes without conversion instructions and has no branch per block
([`20261117T024602Z-quant-decode-loads`](../20261117T024602Z-quant-decode-loads/INDEX.md), unpinned). One bar per sweep
model: GPU decode tg at a 2,048-token context with `q8_0` device KV over the FP16 copy **>= 0.95** (item 7's decode bar,
exit criterion 7). It was reopened on 2026-11-16 when the FP16 decode moved to the wide decode block and Qwen2.5-3B read
0.938 against it ([`20261116T224500Z-fp16-wide-decode-gate`](../20261116T224500Z-fp16-wide-decode-gate/INDEX.md)).

**Pinned:** CPU yes (governor performance, turbo off). GPU clock recorded, not fixed: the 1,300 MHz lock was accepted and
not applied (every CLOCK line reads "lock NOT applied"), as on every earlier gate on this card. The card was cooled to
65 C before every JVM. Within each model both sides ran at the same median loaded clock, except Qwen2.5-3B, where the
`q8_0` side ran 13 MHz (0.7%) below the FP16 side (section 2); that works against `q8_0`, not for it.

**Host:** Xeon E5-1650 v2, 64 GiB, GTX 1080 8 GiB (shared with the desktop). **Build:** one tree, both sides:
`637751b0` plus the uncommitted change (`tree.txt`: `head 637751b0...`, `diff c6b66b809fac35e7`), the hash `READY`
recorded before the run. That uncommitted tree was committed afterwards as `9c33ab93`: the hash rebuilt from
`git diff 637751b0 9c33ab93` (changed files as a diff, added files by name and content, the way `tree-hash.sh` reads
untracked files) is `c6b66b809fac35e7`, so the commit holds the bytes this gate measured.

**Method** (`dist/quant-decode-loads/run-gate.sh`, owner run 2026-11-17T05:31Z to 06:24Z): per model six JVMs, A B A B A B
(A `f16` device KV, B `q8_0` keys and values), each running `DeviceKvDecodeTgLiveTest`: prefill 2,048 tokens, then 128
timed greedy steps through the serving decode path, one warm-up and three measured repetitions, the JVM's reading their
median. The model's ratio is the median of the three B readings over the median of the three A readings.

## 1. Result (`results.txt`)

| Model | `f16` (A), tok/s | `q8_0` (B), tok/s | **B over A**, bar 0.95 |
|---|---|---|---|
| TinyLlama 1.1B | 160.645 | 158.281 | **0.9853** pass |
| Qwen2.5-3B | 62.688 | 60.049 | **0.9579** pass |
| Phi-3.5-mini | 46.784 | 45.566 | **0.9740** pass |
| Mistral 7B | 37.478 | 36.428 | **0.9720** pass |

JVM readings (`<model>.txt`), A1 A2 A3 / B1 B2 B3:
- TinyLlama: 161.454 148.117 160.645 / 158.470 158.281 154.752
- Qwen2.5-3B: 62.797 62.527 62.688 / 60.069 58.495 60.049
- Phi-3.5-mini: 47.381 46.734 46.784 / 43.435 45.566 45.679
- Mistral 7B: 37.505 36.876 37.478 / 36.010 36.428 36.823

Per cycle, B over the A before it: TinyLlama 0.982, 1.069, 0.963; Qwen2.5-3B 0.957, **0.936**, 0.958; Phi-3.5-mini
**0.917**, 0.975, 0.976; Mistral 7B 0.960, 0.988, 0.983. Two single cycles sit under the bar. Qwen2.5-3B's second B JVM
read 58.495 (its own repetitions 59.342, 57.752, 58.495), and the other two read 60.069 and 60.049. Phi-3.5-mini's first
B JVM read 43.435, the slowest JVM of either side, in a run where the card throttled for heat on about 80% of loaded
samples (section 2). The gate scores medians of three, as every pinned gate of this tier has, and each row's repetitions
agree within 15% of their median (largest spread 8.3%, TinyLlama A), so every row is scorable. Qwen2.5-3B's margin is
0.8%.

Against the 2026-11-16 gate, which ran the same FP16 decode code (`gqa_attention.ptx` unchanged since), FP16 reads within
1.0% on every model (TinyLlama 159.132 then, Qwen2.5-3B 62.213, Phi-3.5-mini 46.900, Mistral 7B 37.245), and `q8_0` moved:
TinyLlama 153.292 to 158.281 (+3.3%), Qwen2.5-3B 58.371 to 60.049 (+2.9%), Phi-3.5-mini 45.725 to 45.566 (-0.3%), Mistral
7B 36.368 to 36.428 (+0.2%). That is the shape the unpinned bench showed: the loop change matters most where the old
loop's launch was slowest against FP16 (TinyLlama and Qwen2.5-3B, 1.24x and 1.08x the FP16 launch before, 0.97x and 0.90x
after) and least on Phi-3.5-mini, already under the FP16 launch. Unpinned the same tree read 0.983, 0.971, 0.975 and
0.976; pinned 0.985, 0.958, 0.974 and 0.972.

## 2. The clock (`gpu-clock-summary.txt`)

Samples above 70 W (prefill and decode), all three JVMs per side:

| Model | A median | B median | Range, temperature | Throttle reasons (loaded samples) |
|---|---|---|---|---|
| TinyLlama 1.1B | 1,860 MHz | 1,860 MHz | 1,607 to 1,885 MHz, 44 to 68 C | none |
| Qwen2.5-3B | 1,835 MHz | 1,822 MHz | 1,733 to 1,860 MHz, 64 to 80 C | none |
| Phi-3.5-mini | 1,607 MHz | 1,607 MHz | 1,493 to 1,860 MHz, 66 to 94 C | software thermal slowdown (0x20) on about 80% |
| Mistral 7B | 1,809 MHz | 1,809 MHz | 1,607 to 1,835 MHz, 69 to 83 C | software thermal slowdown on 12% to 34%; software power cap (0x4) on a few |

## 3. Prefill, not a bar (`prefill-from-logs.txt`)

From the JVM logs' per-layer prefill records, a 2,048-token prefill's time, median of the three JVMs per side:

| Model | `f16` | `q8_0` | `q8_0` over `f16` |
|---|---|---|---|
| TinyLlama 1.1B | 995.9 ms | 939.0 ms | 0.943 |
| Qwen2.5-3B | 2,414.8 ms | 2,262.8 ms | 0.937 |
| Mistral 7B | 4,941.7 ms | 4,642.0 ms | 0.939 |

The `q8_0` prefill times agree with the 2026-11-16 gate's within 1.4% (949.1, 2,275.8 and 4,706.1 ms then): this change
touched only the decode loop, and the prefill tiles' PTX is the committed one. Phi-3.5-mini's handler writes no per-layer
prefill records.

## How to read it

Decision 9 (a) passes item 7's decode bar: at a 2,048-token context, decode with `q8_0` device KV reads 0.958x
(Qwen2.5-3B) to 0.985x (TinyLlama) the FP16 copy's rate, pinned, on every sweep model, against the faster FP16 decode of
2026-11-16. Exit criterion 7, reopened by that gate, is met again; its other bars (bytes per position, session capacity,
greedy agreement, the 2,048-token prefill) stand on their earlier evidence. The `q8_0` decode no longer leads the FP16
decode on any model, as it did before the FP16 decode took the wide block; it gives up 1.5% to 4.2% of the decode rate
for half the device KV.

Files: `results.txt`, `pin.txt`, `tree.txt`, per model `<model>.txt` (every JVM's DEVICE-KV-CAPACITY, DEVICE-KV-TG and
CLOCK lines), `gpu-samples/<model>-<A|B><n>.gpu.csv`, `gpu-clock-summary.txt`, `prefill-from-logs.txt`. The JVM logs stay
in `dist/quant-decode-loads/runs/20261117T053104Z/`.
