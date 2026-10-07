# KV mirror reserved for a 512-token prompt on a partially offloaded model

**Purpose.** Measure the change that makes `--gpu-layers auto` keep one request's GPU-attention KV
mirror free at 512 positions on the layers whose weights are on the device (it kept the mirror free at
its initial 64 positions only), and makes the adaptive prefill chunk leave that memory to the mirror.
Read for: GPU-resident layer count, memory kept free, resolved prefill chunk, and whether a
512-token prompt's attention stays on the device (no device-memory fallback logged) on llama-1-30b,
the model the card does not fit. Models that fit whole are not measured here: the reserve is far
below one of their layers (Mistral 7B's 512-token mirror is 65 MiB), so it cannot change their layer
count, which the unit test `DeviceScratchBudgetTest.aModelThatFitsWholeReservesLittleForItsMirror` holds.

**Pinned: no.** Layer counts, memory kept free and fallbacks do not depend on clocks. The wall times
below are single unpinned readings, recorded for reading only and not scored against any gate.

**Builds.**

| Side | Source | Shaded jar sha256 (first 16) |
|---|---|---|
| baseline | HEAD `3b27aa3`, built from `git archive` | `1a959d18cab40409` |
| candidate | working tree on `3b27aa3` with this change | `fe69862b55a1a935` |

Host: GTX 1080 (8 GiB), driver 580.173.02, desktop session sharing the card; JDK 25.0.3. One engine at a
time, 2026-10-06 21:24Z to 22:40Z.

**Command.** `measure-reserve.sh <label> <jar> <model> 1 30g 8 <out>` (in this directory, unchanged from
`20261003T235146Z-packed-kquant-reserve` apart from the repository path): `--local --gpu --schedule
static --verbose`, one node, default `--gpu-layers auto`, default `--prefill-batch`, 30 GiB heap (the
launcher's file-size rule for this model). In one process it sends a short prompt with 8 generated
tokens, then a 512-token prompt (508 prompt tokens by this model's tokenizer) with one generated token,
sampling `nvidia-smi --query-compute-apps` for the engine's PID.

## Results

| Build | Layers on GPU | Kept free | Free after load | Resolved chunk | Prefill peak / decode peak | Device-memory fallbacks on the 508-token prompt | 508-token prefill / 8 generated tokens (wall, unpinned) |
|---|---|---|---|---|---|---|---|
| baseline | 22 | 196 MiB | 382 MiB | 448 | 1.033 (7446 / 7206 MiB) | prefill attention inside the region, to the CPU | 2030 s / 160 s |
| candidate | **21** | 453 MiB | 697 MiB | 448 | 1.066 (7318 / 6868 MiB) | **none** | 2127 s / 171 s |

Run files: `runs/<label>.json` (VRAM, timing, token usage) and `runs/<label>.log-excerpt.txt` (the
load-time and fallback log lines).

## How to read it

- **The criterion is met.** On the candidate, the 508-token prompt runs with no device-memory fallback
  of any kind (no mirror growth, region or attention-kernel warning in the log). The baseline logs the
  region's prefill attention falling back to the CPU on the same prompt, as Tier 01C's build did
  (`20261003T235146Z-packed-kquant-reserve`: mirror growth and a region window).
- **The cost is one weight layer**, 22 to 21 GPU layers, as forecast: a 512-token mirror on 22 device
  layers of this model is about 286 MiB, close to one 30B Q4_K_M layer (about 310 MiB).
- **On this model the trade does not pay in wall time.** Both readings are about 5% slower on the
  candidate (prefill 2127 against 2030 s, generation 171 against 160 s): with 39 of 60 layers on the
  CPU against 38, the extra CPU layer costs more than CPU attention over 508 positions on 22 layers.
  These are single unpinned readings with the CPU layers dominating both builds; they are recorded, not
  scored. The change buys a device path that holds at 512 positions rather than throughput on a model
  this far beyond the card.
- **Free after load is not today's figure plus a constant.** The desktop's own GPU use moves it between
  runs (Tier 01C's baseline build stopped at 23 layers with 197 MiB kept free; today's at 22 with 196).
  Compare builds within this directory only.
- **Not covered by the reserve:** growth past 512 positions, and more than one concurrent request's
  mirror (`--parallel`); both still fall back to CPU attention, announced once.

### Recorded attempt, superseded: candidate at a 22 GiB heap

`runs/heap22g-cand-llama30b-n1.*`: the same candidate at a 22 GiB heap (the residency smoke's heap rule)
(21 GPU layers, 667 MiB free after load, chunk 384) ran its short request and then ran out of Java heap
at the start of the 508-token prompt's first 384-row window, and the process had to be killed. That heap
is below the launcher's rule for this file (30 GiB). Whether the baseline, with one CPU layer fewer on
the heap, would also run out at 22 GiB was not measured. Both builds were then measured at 30 GiB as
above.
