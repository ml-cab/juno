# Tier 02D: GPU host overhead and decode-region coverage

Status: not started
Gap analysis refs: none — see "Why this tier, why now"

**Added 2026-10-08 (owner decision, Tier 02 decision 15).** Tier 02's closing sweeps met every GPU end-of-plan
target with room (GPU tg 0.858x on Phi-3.5-mini and 1.030x on Mistral 7B against 0.70x; GPU pp 0.594x at 512 and
0.540x at 2048 against 0.40x and 0.25x). By this plan's rule a target met before the work meant to reach it measures
nothing, so the owner raised them, and placed the work that has to reach them in this tier, after Tier 02C and
before Tier 03.

## Objective

Bring the GPU path to the raised end-of-plan targets by removing what is left between the device kernels and the
wall clock: Qwen2's decode outside the device region, the host work around Phi-3's prefill window, and Phi-3's
remaining per-token decode cost.

## Why this tier, why now

**Three levers had no owner.** At Tier 02's close (reference column 2026-10-08, `docs/perf-compare/20261008T173153Z/`
and following):

- **Qwen2.5-3B decodes at 0.416x**, the lowest GPU tg of the four sweep models, because the device-resident decode
  region declines Qwen2: `GpuResidencyOptions` and `ResidentQkvPath` give "uses the split-half RoPE layout (and Q/K/V
  biases), which the device region does not implement". Both operations already exist as device kernels:
  `rope_split_half` (`rope.cu`) runs in Qwen3's decode region, and `bias_add` runs in the prefill window
  (`PrefillWindowRegion`). The LLaMA-family decode region uses neither. On the three models the region covers, it
  lifted decode 1.56x to 1.99x (Tier 02 item 6 sweeps).
- **Phi-3.5-mini binds both prefill rows** (0.594x at 512, 0.540x at 2048), and its 512-token window spends only
  45% of its time in GEMM, against 71% on Mistral 7B. Of the rest, device attention and elementwise kernels are 19%,
  and 35.9% is host-side or unattributed: unattributed residue 12.0%, the region's host side 9.9%, the host KV write
  8.2%, matmul staging 5.8%
  (`docs/perf-compare/20261008T054413Z-attention-close-gate/spans512/`). Mistral 7B's same terms total about 10%.
- **Phi-3.5-mini's decode** reads 0.858x. No item in a later tier moves default-flag decode: Tier 06's speculation is
  opt-in, and the reference engine does not speculate, so it does not score these rows.

It runs after Tier 02C because CPU is still the largest gap in the program (about 0.1x), and before Tier 03 because
Tier 03 changes the KV layout the host KV write sits on (Tier 03 item 6 owns the host KV storage; this tier owns the
time the prefill window spends writing it).

## Scope

### In scope

1. **Decompose before fixing.** On the current build, publish:
   (a) a Phi-3.5-mini and a Qwen2.5-3B decode step: device kernel time, host time between launches, and wait time
   per layer, from `--device-spans` at decode width plus `jdk.ExecutionSample`, at context 64 and 512;
   (b) the Phi-3.5-mini 512- and 2048-token prefill window with `prefill-breakdown.sh`, with the 12% residue
   attributed (add the span that covers it, or name the code it runs);
   (c) for each raised row, the ratio the decomposition says this tier's items can reach with everything else held.
   If a raised row is out of reach by that arithmetic, write the numbers here and restate the row with the owner
   before implementing, as Tier 02's step 2 did.
2. **Qwen2 in the decode region.** Run the LLaMA-family decode region on split-half RoPE layouts with
   `rope_split_half`, and add the Q, K and V biases on the device with the prefill window's bias kernel, so
   Qwen2/Qwen2.5 decode runs the whole layer in the region. Same scope and fallbacks as the region today
   (single-sequence decode, CUDA, K-quant device weights), the same notice rules (`auto` silent where declined).
   Bit-identity against the op-at-a-time path by test, as for every region handler.
3. **Phi-3's prefill window host work.** Remove or overlap the terms the decomposition ranks: the region's host side
   (`projection_and_region_host`), the host KV write (`kv_host_write`, the time only; its storage is Tier 03 item 6),
   matmul staging, and whatever the attributed residue turns out to be. Apply the same change to the LLaMA family and
   Qwen3 where the term exists there.
4. **Phi-3's decode step.** From item 1 (a), remove the largest term of a Phi-3.5-mini decode step that is not device
   kernel time; if the step is already bound by device kernels, say so with the number and name the kernel.

### Out of scope

- CPU work (Tier 02C), the KV layout and host KV storage (Tier 03), new quantization formats (Tier 04), speculative
  decoding (Tier 06), ROCm (Tier 10).
- Phi-2 and Qwen3-MoE device paths (Tier 08 item 6).

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | N/A for the region and the prefill window (CUDA only); unchanged, held by the standing CPU gate |
| 2 | CUDA GPU inference | primary target |
| 3 | ROCm GPU inference | N/A or NEEDS-AMD-HARDWARE: the region is CUDA-only; `auto` resolves off on ROCm |
| 4 | Static schedule | prefill windows (item 3) at every `--parallel` width the tests cover |
| 5 | Continuous schedule | prefill chunks reach item 3's changes; the decode region stays single-sequence |
| 6 | Single-node local mode | primary dev surface |
| 7 | Pipeline-parallel cluster | forked nodes run the region on their layers; greedy output matches local mode |
| 8 | Tensor-parallel cluster | same |
| 9 | LoRA training | declines the region, as today; unchanged |
| 10 | LoRA playback | declines the region, as today; unchanged |
| 11 | Vision | moondream2 (Phi-2 text half) does not reach these paths; `compare-vision.sh` confirms unchanged |
| 12 | OpenAI REST surface | verified through the smoke runs |
| 13 | Native REST surface | same |
| 14 | CLI | no new flag; `--gpu-residency` help and `docs/howto.md` name Qwen2 as covered |
| 15 | JVM embedding facade | N/A: no new embedder-invocable capability |

## Implementation steps

1. Run `scripts/performance-tests/check-plan-thresholds.sh` first.
2. Item 1, the decomposition, published; restate any row it shows out of reach, with the owner.
3. Items 2, 3 and 4 in the order the decomposition ranks them, re-measuring after each.
4. Run the full cross-surface smoke matrix.

## Tests to write/upgrade before implementation

- **Plan check, first**: `scripts/performance-tests/check-plan-thresholds.sh` passes before any other test or code
  in this tier (README execution rule 7).
- **Qwen2 in the region (item 2)**: a `ResidentQkvPath` case per new operation (split-half rotation, Q/K/V bias)
  bit-identical to the op-at-a-time path at several positions and across concurrent threads; a handler test that a
  Qwen2.5-3B decoded token makes 2 uploads and layers + 1 downloads (`juno.DeviceStaging`), as the other region
  handlers' tests do; greedy parity on against off. `smoke-gpu-residency.sh` (unmodified) then reports Qwen2.5-3B as
  active rather than declined.
- **Prefill window (item 3)**: logits bit-identical, or within the existing `PrefillRegionHandlerParityTest` bounds,
  region on against off, on every changed handler; a byte or span assertion for each removed term.
- **`ModelLiveRunnerIT`**: check 9 at 512 and 2048 on Qwen2.5-3B and Phi-3.5-mini, unchanged contract.
- **No new smoke script**: `smoke-gpu-residency.sh` and `smoke-long-prompt-prefill.sh` already exercise both paths;
  re-run them unmodified.
- **Standing CPU and allocation gate** (README, "Test infrastructure"): run against the pre-tier jar and score it
  before closing this tier.
- **Perf gate (required)**: hot-path change (forward pass, GPU residency). `compare-llama-cpp.sh --gpu` closing
  sweeps at 128, 512 and 2048 (Phi-3.5-mini at 2048 at `COMPARE_HEAP=8g`), `compare-lora.sh` and `compare-vision.sh`,
  published under `docs/perf-compare/`.

  **Threshold** (the README's raised end-of-plan rows, scored on this tier's pinned closing sweeps):
  - GPU tg **>= 0.70x** on every sweep model (reference 0.416x, Qwen2.5-3B binding);
  - GPU tg **>= 0.90x** on Phi-3.5-mini (reference 0.858x);
  - GPU pp **>= 0.70x** at `n_prompt=512` (reference 0.594x) and **>= 0.60x** at `n_prompt=2048` (reference 0.540x),
    every sweep model, Phi-3.5-mini binding.

  Throughput must not regress: Juno tg and pp t/s **>= 0.95x** the pre-tier build on every sweep model, from a
  same-hour interleaved A/B with pinned clocks (README, "No-regression gates tighter than the floor are
  Juno-against-Juno"). Vision latency **<= 1.25x** and decode **>= 0.80x**; LoRA playback **>= 0.80x**.

## Models needed

The four sweep models and Qwen3-1.7B (the other split-half region handler, held unchanged). All present.

## Exit criteria

- [ ] Item 1's decomposition published before any fix, with each raised row's reachable ratio; any row it puts out
      of reach restated with the owner first.
- [ ] Qwen2 decodes in the region: bit-identity by test, 2 uploads and layers + 1 downloads per decoded token,
      greedy identical on against off, `smoke-gpu-residency.sh` reports it active.
- [ ] Phi-3's prefill window host terms reduced as item 3 states, each by a span or byte assertion; logits within
      the parity bounds.
- [ ] Phi-3's decode step: the largest non-kernel term removed, or the step shown kernel-bound with the number.
- [ ] Raised end-of-plan rows (GPU tg >= 0.70x every sweep model, >= 0.90x Phi-3.5-mini; GPU pp >= 0.70x at 512,
      >= 0.60x at 2048) each reported met or missed with its number from the pinned closing sweeps.
- [ ] Perf gate published: no regression (>= 0.95x A/B, pinned), vision and LoRA gates, and the standing CPU and
      allocation gate met.
- [ ] Cross-surface checklist fully resolved.
- [ ] Docs (`docs/howto.md` `--gpu-residency`, `docs/agent-arch.txt`, `docs/performance.md`) updated, Juno-native
      language only.
- [ ] `CHANGELOG.md` entry added.
