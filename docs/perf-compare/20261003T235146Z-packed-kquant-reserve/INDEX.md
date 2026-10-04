# Forward-pass reserve sized from the prefill window

**Purpose.** Measure the change that shrinks the device memory `--gpu-layers auto` keeps free after
the weight upload (`DeviceScratchBudget`) and sizes the adaptive prefill chunk from each shard's real
prefill-window footprint (`PrefillWindowFootprint`, `PrefillChunkDefaults`). Read for: GPU-resident
layer count, memory kept free, resolved prefill chunk, device-memory fallbacks, and per-process peak
VRAM during a 512-token prefill against peak during decode (the tier's VRAM threshold, `<= 1.05x` on
mistral-7b and llama-1-30b).

**Pinned: no.** VRAM peaks and layer counts do not depend on clocks. The two 30B wall times below are
single unpinned readings, recorded for reading only, not scored against any gate.

**Builds.**

| Side | Source | Shaded jar sha256 (first 16) |
|---|---|---|
| baseline | HEAD `5a77981`, built from `git archive` | `ce4db246795fd33f` |
| candidate | working tree on `5a77981` with this change | `2d55d27b07a8a564` |

Host: GTX 1080 (8 GiB), driver 580.173.02, desktop session sharing the card; JDK 25.0.3. One engine at a
time, 2026-10-03 23:51Z to 2026-10-04 02:41Z.

**Command.** `measure-reserve.sh <label> <jar> <model> <nodes> <heap> <decode-tokens> <out>` (in this
directory): `--local --gpu --schedule static --verbose`, default `--gpu-layers auto`, default
`--prefill-batch`. In one process it sends a short prompt with 64 generated tokens (8 on the 30B), then
a 512-token prompt (the engine's own `prompt_tokens`, 464 to 512 by tokenizer) with one generated
token, sampling `nvidia-smi --query-compute-apps` for the engine's PID about every 50 ms. "Decode peak"
is the highest reading during the first request, "prefill peak" during the second.

## Results

| Model, nodes | Layers on GPU (base / cand) | Kept free (base / cand) | Resolved chunk (base / cand) | Prefill peak / decode peak (base / cand) | Device-memory fallbacks (base / cand) |
|---|---|---|---|---|---|
| tinyllama-1.1b, 1 | 22 / 22 | not binding | 52202 / 4738 | 1.098 / 1.098 | none / none |
| qwen2.5-3b, 1 | 36 / 36 | not binding | 40266 / 5213 | 1.045 / 1.045 | none / none |
| Phi-3.5-mini, 1 | 32 / 32 | not binding | 38848 / 3659 | 1.125 / 1.126 | none / none |
| **mistral-7b, 1** | 32 / 32 | not binding | 23880 / 2601 | **1.043 / 1.043** | none / none |
| **llama-1-30b, 1** | 22 / **23** | 416 / 197 MiB | 3471 / 320 | **1.034 / 1.012** | region attention / KV mirror growth, one region window |
| llama-1-30b, 3 | 23 / 23 (20+2+1 / 20+3+0) | 351 / 132 MiB | 2480 / 32 | 1.007 / 1.001 | region, matmul, KV mirror / same kinds |

30B wall times, single unpinned readings (not a gate): 508-token prefill 2467 s / 2118 s (1 node) and
2450 s / 1967 s (3 nodes); 8 generated tokens 189 s / 168 s and 167 s / 196 s. About 37 of the 60
layers run on the CPU in every configuration, which dominates both figures.

## How to read it

- **Models that fit whole are unchanged.** Same layers, same peaks, no fallbacks. The resolved chunk
  falls about tenfold but stays at 2601 rows or more, so every 128-, 512- and 2048-token prompt the
  sweeps and smokes send is still one window.
- **VRAM threshold: met on both models it names, on both builds.** Mistral 7B 1.043, 30B 1.012 (1
  node). The baseline already met it: removing the weight-shaped dequant scratch from the prefill path
  happened in the switch to the packed kernel, before this change. What a 512-token prefill adds over
  decode now is the region window and its attention scores. TinyLlama (1.098) and Phi-3.5-mini (1.125)
  are above 1.05 on both builds, from the same window, and are not in the threshold's scope.
- **The 30B on one node gains one GPU layer** (22 to 23) from the smaller reserve. On three nodes the
  total stays 23: the third shard's first layer, uploaded before its per-layer cost is known, hits the
  allocator's refusal (handled, warned) instead of being stopped by the rule.
- **Fallbacks remain on the 30B on both builds.** The KV mirror for a 508-token prompt on 23 layers
  needs about 311 MiB, which no reserve on this card holds; it grows at run time and was never reserved
  for. Every fallback lands on a correct path (CPU attention or the host path for a window's layer).
  The baseline's 3471-row chunk was the per-token figure's over-ask; the candidate's 320 rows is the
  footprint's answer for the memory left.
- `host-specific`: the 44 to 54 MiB the free-memory query reports with the card full, which the
  reserve now carries as a 64 MiB allowance, was measured on this card and driver with a desktop
  session.
