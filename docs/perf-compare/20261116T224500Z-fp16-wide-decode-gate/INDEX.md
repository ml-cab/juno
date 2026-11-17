# FP16 decode attention on the wide decode block, pinned gate: FP16 faster on every model, `q8_0` misses on Qwen2.5-3B

**Purpose.** The owner's pinned A/B for decision 5 (a) (2026-11-15): the FP16 decode attention moved from the 128-thread
tiled kernel to the 256-thread wide decode block
([`20261116T204541Z-fp16-wide-decode`](../20261116T204541Z-fp16-wide-decode/INDEX.md), unpinned). Two bars per sweep
model: FP16 decode tg after the change over before **>= 1.00** (the decision's A/B), and `q8_0` device KV decode tg over
the new FP16 decode **>= 0.95** (item 7's decode bar, exit criterion 7, re-scored against the faster FP16 decode).

**Pinned:** CPU yes (governor performance, turbo off). GPU clock recorded, not fixed: the 1,300 MHz lock was accepted and
not applied (every CLOCK line reads "lock NOT applied"), as on every earlier gate on this card. The card was cooled to
65 C before every JVM. Within each model the three sides ran at the same median loaded clock, except Qwen2.5-3B, where
the two candidate sides ran 13 MHz (0.7%) below the baseline side (section 2); that works against the candidate, not for it.

**Host:** Xeon E5-1650 v2, 64 GiB, GTX 1080 8 GiB (shared with the desktop). **Builds:** baseline = `git archive` of
`637751b0` (the FP16 decode on the tiled kernel) in `dist/fp16-wide-decode/baseline-tree`; candidate = `637751b0` plus
the uncommitted change, both from their trees' test classes. The tree hash the waiter checked (`head 637751b0...`,
`diff 6c54ee645662eb41`, `tree.txt`) still matched after the run.

**Method** (`dist/fp16-wide-decode/run-gate.sh`, owner run 2026-11-16T22:45Z to 2026-11-17T00:02Z): per model nine JVMs,
A B C A B C A B C (A `f16` baseline tree, B `f16` candidate, C `q8_0` keys and values candidate), each running
`DeviceKvDecodeTgLiveTest`: prefill 2,048 tokens, then 128 timed greedy steps through the serving decode path, one
warm-up and three measured repetitions, the JVM's reading their median. A ratio is the median of three readings over the
median of three.

## 1. Result (`results.txt`)

| Model | FP16 before (A) | FP16 after (B) | **B over A**, bar 1.00 | `q8_0` after (C) | **C over B**, bar 0.95 |
|---|---|---|---|---|---|
| TinyLlama 1.1B | 124.193 | 159.132 | **1.2813** pass | 153.292 | **0.9633** pass |
| Qwen2.5-3B | 57.104 | 62.213 | **1.0895** pass | 58.371 | **0.9382** FAIL |
| Phi-3.5-mini | 43.190 | 46.900 | **1.0859** pass | 45.725 | **0.9749** pass |
| Mistral 7B | 34.910 | 37.245 | **1.0669** pass | 36.368 | **0.9765** pass |

Per cycle, B over the A before it: TinyLlama 1.282, 1.287, 1.281; Qwen2.5-3B 1.087, 1.101, 1.089; Phi-3.5-mini 1.082,
1.082, 1.086; Mistral 7B 1.078, 1.065, 1.067. Every B JVM is faster than every A JVM (slowest B over fastest A: 1.280,
1.087, 1.074, 1.064).

Per cycle, C over the B before it: TinyLlama 0.941, 0.959, 0.963; **Qwen2.5-3B 0.938, 0.938, 0.938**; Phi-3.5-mini
0.975, 0.980, 0.981; Mistral 7B 0.961, 0.980, 0.976. Qwen2.5-3B's miss is not noise: its fastest C over its slowest B
reads 0.940. To pass it needs 59.10 tok/s from `q8_0`, 1.3% more, about 0.21 ms a token or 6 us a layer.

Both sides agree with the 2026-11-15 re-run
([`20261115T200324Z-device-kv-tg-gate-rerun`](../20261115T200324Z-device-kv-tg-gate-rerun/INDEX.md)) within 0.6%: FP16
before 124.074, 56.759, 43.369, 35.011 then; `q8_0` 153.177, 58.483, 45.810, 36.576 then. The `q8_0` decode did not move
(its PTX is the committed one); the FP16 decode it is scored against did.

## 2. The clock (`gpu-clock-summary.txt`)

Samples above 70 W (prefill and decode), per side:

| Model | A median | B median | C median | Range, temperature | Throttle reasons (loaded samples) |
|---|---|---|---|---|---|
| TinyLlama 1.1B | 1,860 MHz | 1,860 MHz | 1,860 MHz | 1,607 to 1,885 MHz, 45 to 73 C | none |
| Qwen2.5-3B | 1,835 MHz | 1,822 MHz | 1,822 MHz | 1,607 to 1,860 MHz, 62 to 79 C | none |
| Phi-3.5-mini | 1,607 MHz | 1,607 MHz | 1,607 MHz | 1,468 to 1,847 MHz, 65 to 94 C | software thermal slowdown (0x20) on about 80% |
| Mistral 7B | 1,809 MHz | 1,809 MHz | 1,809 MHz | 1,607 to 1,847 MHz, 66 to 83 C | software thermal slowdown on 17% to 28% |

## 3. Prefill, not a bar (`prefill-from-logs.txt`)

From the JVM logs' per-layer prefill records, a 2,048-token prefill's time, median of the three JVMs per side:

| Model | FP16 before | FP16 after | after over before | `q8_0` after | `q8_0` over FP16 after |
|---|---|---|---|---|---|
| TinyLlama 1.1B | 998.2 ms | 996.0 ms | 0.998 | 949.1 ms | 0.953 |
| Qwen2.5-3B | 2,425.3 ms | 2,434.0 ms | 1.004 | 2,275.8 ms | 0.935 |
| Mistral 7B | 4,957.6 ms | 4,938.5 ms | 0.996 | 4,706.1 ms | 0.953 |

Prefill tiles keep the tiled entries, and their time did not move (0.996 to 1.004). Phi-3.5-mini's handler writes no
per-layer prefill records.

## How to read it

Decision 5 (a) passes: FP16 decode tg at a 2,048-token context is 6.7% (Mistral 7B) to 28% (TinyLlama) faster on every
sweep model, pinned, with every candidate JVM faster than every baseline JVM, and prefill unchanged. Against that faster
FP16 decode, item 7's decode bar holds on three models (0.963 to 0.977) and misses on Qwen2.5-3B at 0.938, the same in
every cycle. Exit criterion 7, ticked on the 2026-11-15 re-run against the slower FP16 decode, is open again. The owner's
decision 9 (a) (2026-11-16) applies here: the quantized decode loop takes the same up-front loads, before decision 4.

Files: `results.txt`, `pin.txt`, `tree.txt`, per model `<model>.txt` (every JVM's DEVICE-KV-TG and CLOCK lines),
`gpu-samples/<model>-<A|B|C><n>.gpu.csv`, `gpu-clock-summary.txt`, `prefill-from-logs.txt`. The JVM logs stay in
`dist/fp16-wide-decode/runs/20261116T224500Z/`.
