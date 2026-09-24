# Tier 01B: Prefill throughput

Status: not started
Gap analysis refs: none directly — this tier exists because the gap analysis has no prefill section
at all, while the published measurements under `docs/perf-compare/` show prompt processing to be the
single largest gap Juno has. See "Why this tier, why now".

## Objective

Raise batched-prefill (prompt-processing) throughput on the GPU path, which is the largest measured
gap against llama.cpp anywhere in this repository and which no other tier in this plan owns. Find
where prefill time actually goes using the JFR spans that already exist (`juno.PrefillBatch`,
`juno.ForwardPass`, `juno.MatVec`, `juno.Attention`), attack the dominant term, and move the
Juno/llama.cpp pp ratio by a stated multiple rather than incidentally.

## Why this tier, why now

The plan as originally written had no tier whose objective was prefill throughput. Tier 02 touches
attention but targets peak memory at long context; Tier 04 adds fused kernels for formats that are
not Q4_K; Tier 07 targets multi-session throughput. Meanwhile the numbers say this:

| Model (GPU, Q4_K_M) | llama.cpp pp t/s | Juno pp t/s | Juno/llama pp | Juno/llama tg | Juno tokens actually prefilled |
|---|---|---|---|---|---|
| tinyllama-1.1b | 3654.97 | 59.25 | 0.016 | 0.239 | **30** |
| qwen2.5-3b | 1482.11 | 28.82 | 0.019 | 0.284 | **20** |
| Phi-3.5-mini | 1199.81 | 28.79 | 0.024 | 0.330 | **21** |
| mistral-7b | 668.57 | 20.89 | 0.031 | 0.513 | **20** |

(`docs/perf-compare/20260918T204809Z/INDEX.md`, `n_gen=64`, GTX 1080.)

**Read that last column before drawing a conclusion from the pp ratios.** That run has
`raw_prompt: 0`, so llama-bench prefilled the 128 tokens it was asked for while Juno prefilled the
20-to-30-token sentence `compare-llama-cpp.sh` hard-codes. A 20-token prefill on a cold JVM is the
single worst shape Juno has: the smallest possible batch, executed once, inside C2 warmup. The
0.016-to-0.031x figures are a real gap plus a batch-width mismatch plus a cold-start penalty, and
the three are not separable from this run. They are not a valid before-measurement for this tier and
must not be quoted as one.

The parity-corrected numbers are fewer, and they are worse rather than better:

| Measurement | Juno pp t/s | llama.cpp pp t/s | Ratio |
|---|---|---|---|
| mistral-7b, real 520-token prompt (`20260915T043143Z`, `raw_prompt: 1`) | 11.79 | 709.73 | **0.0166x** |
| tinyllama, real 530-token prompt (same run) | 31.83 | 4225.02 | **0.0075x** |
| tinyllama, real 512-token prompt, `--gpu-attention on`, `--prefill-batch 32` (`20260916T040113Z-prefill/`) | **119.56** | 4225.02 | **0.028x** |

That third row is the best GPU prefill figure anywhere in this repository. Everything this tier does
is measured against it, not against 0.0166x.

Note also that the two runs previously cited as proof that "pp gets worse with prompt length"
differ in **two** variables besides prompt length: the 128 run has `raw_prompt: 0` (20 Juno tokens)
while the 512 run has `raw_prompt: 1` (520 Juno tokens), and the 128 run used `--vector 1` while the
512 run used `--vector 0`. The claim may well be true — a fixed per-chunk cost paid sixteen times
for a 512-token prompt predicts exactly that shape — but it is **not established by those two
runs**, and step 1 below is what actually establishes it.

Decode is 2 to 4 times slower than llama.cpp. Prefill, measured like-for-like, is 35 to 130 times
slower depending on configuration. A serving engine whose time-to-first-token scales that badly with
prompt length has a product problem, not only a benchmark problem, and every tier after this one
that measures a llama.cpp-relative pp ratio will keep reporting a number nobody is working on.

**One prefill lever exists on part of the model set, and what it is worth elsewhere is unmeasured.**
The GPU-resident attention kernel behind `--gpu-attention` was measured taking pp from **31.04 to
119.56 t/s (3.85x)**, with attention's share of prefill wall time dropping from **78.7% to 11.0%**,
at `--prefill-batch 32` on a real 512-token prompt — on **`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`**
(`docs/perf-compare/20260916T035952Z-prefill/INDEX.md` off versus
`20260916T040113Z-prefill/INDEX.md` on; `docs/performance.md` labels the same table "TinyLlama
Q4_K_M"). An earlier draft of this tier attributed that result to Phi-3.5. It did not come from
Phi-3.5, and three things follow that change what item 0 is worth:

- **TinyLlama is Llama-family, so it already runs with the kernel on by default.** The 3.85x is the
  performance of the *current default*, not a lever waiting to be switched on. There is no
  measurement, anywhere in this repository, of what Phi-2, Phi-3, Qwen3 or Qwen3-MoE would gain.
- **The derived ratio was wrong by roughly 3x.** "Roughly 120 t/s against llama.cpp's ~1200 t/s is
  about 0.10x" used Phi-3.5's llama.cpp figure against TinyLlama's Juno figure. Against the correct
  denominator (4225 t/s on the same run shape) 119.56 t/s is **0.028x**.
- **Item 0 is therefore a bet, not a banked win**, and this tier's threshold cannot lean on it.

The flag's default is `auto`, not `off`, and `auto` resolves to on whenever CUDA is present —
`GpuAttentionOptions.preferGpuAttention()` is literally `CudaAvailability.isAvailable()` for `AUTO`.
**There is no per-architecture resolver.** The class has no architecture awareness at all; the
reason the kernel is inactive for Phi-2, Phi-3, Qwen3 and Qwen3-MoE is simpler and more expensive to
fix than a resolver would be: those handlers never read `GpuAttentionOptions`. Only
`LlamaTransformerHandler` and `LoraTrainableHandler` reference it, and each of the four uncovered
handlers owns its own KV map and attention math. ROCm is inactive for a third reason again — `AUTO`
keys off CUDA availability specifically.

So item 0 is **four kernel integrations plus a capability-reporting mechanism that does not exist
yet**, not "existing kernel code reaching more architectures." Size it accordingly. It still goes
first, because every other measurement in this tier would otherwise be taken against a baseline that
is fast on four architectures and slow on four others — but it is no longer the cheap win this tier
was originally written around.

This tier is sequenced immediately after Tier 01 **and depends on it.** An earlier draft claimed
independence from Tier 01's go/no-go outcome on the grounds that "prefill is dominated by large-batch
GEMM and host-device transfer, not by the per-op dispatch overhead Tier 01 targets." That reasoning
defeats itself: host-device transfer *is* what Tier 01's activation-residency primitive removes, and
scope item 2 below — now the largest item in this tier — is built directly on it. If Tier 01 comes
back negative and downgrades to partial-complete, item 2 cannot proceed as written and this tier
escalates rather than silently substituting a smaller scope. Items 0, 1, 3 and 4 remain executable
on today's op-at-a-time GPU path either way.

## Scope

### In scope

0. **Make `--gpu-attention` genuinely default.** Close the architecture gap so the GPU-resident
   attention kernel is on by default everywhere it is correct, rather than resolving to the scalar
   CPU path for half the supported model set:
   - Add GPU-attention support to `Phi2TransformerHandler`, `Phi3TransformerHandler`,
     `Qwen3TransformerHandler` and `Qwen3MoeTransformerHandler`, which today keep the scalar CPU
     path unconditionally.
   - Decide and document the ROCm answer. If the kernel can be ported, port it (subject to this
     plan's `NEEDS-AMD-HARDWARE` rule, since there is no AMD device here). If it cannot land this
     tier, `auto` must say so — a startup notice naming the backend and the resulting path, not a
     silent resolution to scalar.
   - Once coverage is complete, change the default from `auto` to `on` and keep `auto` as an
     explicit opt-in for anyone who wants per-architecture resolution. Where a path genuinely cannot
     support the kernel, it fails loudly to the documented fallback rather than resolving quietly.
   - Keep the two existing, deliberate exemptions and re-verify them rather than assuming: LoRA
     **training** and `--lora-play` ignore `--gpu-attention` (attention stays scalar CPU, the train
     REPL warns), and that stays true unless this tier explicitly changes it.
   - Carry the documented consequence honestly: the FP16 KV mirror can produce occasional
     multi-token greedy-decode divergence from the scalar path. That is acceptable as a default only
     if it is stated plainly in `docs/howto.md` and `--help`, and `off` remains available as the
     bit-identical CPU-parity baseline. Re-verify the divergence characteristics on each newly
     covered architecture before turning its default on — do not inherit the Llama-family result.
   - **Fix the benchmark's stale description in the same change.** `compare-llama-cpp.sh` documents
     its default lane as `--gpu-attention off` (its `--help` text and the comment above the
     default-flags lane), but it only passes the flag when the variable is non-empty, so the default
     lane has actually been running whatever Juno's own default resolves to. Every published default
     lane number is therefore labelled wrong. Correct the labelling, and state in the first
     re-baselined run which lanes were affected — this sits alongside the benchmark-parity
     preconditions in [`README.md`](README.md) and is the same class of defect.
1. **Measure before changing anything.** Produce a per-term breakdown of prefill wall time for all
   four sweep models at `n_prompt` of 128 and 512, on GPU, from the JFR spans that already exist:
   `juno.PrefillBatch` (window size, start position), `juno.ForwardPass` (prefill total), `juno.MatVec`
   (call count and time), `juno.Attention` (prefill share). Attribute the remainder — host-to-device
   staging, dequantization, layout packing, per-chunk fixed cost — explicitly rather than leaving it as
   an unlabelled residue. This breakdown is the tier's primary artifact and decides what items 2-5
   are actually worth doing; publish it under `docs/perf-compare/` before writing any kernel code.
2. **Stop staging the activation batch to host between every matmul.** An earlier draft of this item
   proposed extending batched `sgemm` to "every other quantized residency type." That set is empty:
   `DeviceFloatMatrix`, `DeviceHalfMatrix` and `DeviceQ4KMatrix` are the *only* device matrix types
   in the module, `CudaMatVec` already overrides `sgemm` for all three, and it already dispatches to
   a real tiled `cublasGemmEx` path above `HALF_SGEMM_BATCH_MAX = 8` — which a prefill window always
   exceeds. Every other quant format is dequantized on the host at load and uploaded as FP16
   (`uploadFp16Layer`). There is no dtype falling through to a serial GEMV loop on CUDA.

   The real cost is the data movement around those GEMMs. `MatVec.sgemm` takes `float[][] X` and
   returns a **new** `float[][] Y`, so every projection stages the entire activation window
   host-to-device and the entire result device-to-host. At a 512-token window on a 4096-dim model
   that is roughly 8 MB moved each way per matmul, seven matmuls per layer, every layer, every
   chunk. `LlamaTransformerHandler.sgemmLayerInto` then allocates the returned `float[][]` and
   `System.arraycopy`s it into the output buffer it was handed, adding a full-batch allocation and
   copy per matmul on top.

   Extend Tier 01's residency primitive to the prefill window: upload the activation batch once per
   layer (once per forward pass where the boundary permits), keep the GEMM outputs on-device across
   the projections, and materialize to host only where something that is not GPU-resident needs the
   data — the attention boundary until Tier 02 lands its kernel, the gRPC boundary in cluster mode,
   or the LM head. Give `MatVec` a non-allocating batched form so `sgemmLayerInto` can stop
   allocating and copying; coordinate the contract with Tier 10 item 4, which adds the same
   output-parameter shape for the CPU path, so the two do not land two different spellings of it.

   **This item is the one that depends on Tier 01.** If Tier 01's residency primitive did not ship,
   escalate rather than proceeding — see "Why this tier, why now."
3. **Chunk sizing and staging cost.** `--prefill-batch` defaults to adaptive whole-prompt sizing only
   on GPU + `static` + local single-shard, and to a fixed `32` everywhere else — including CPU, the
   `continuous` schedule, cluster, and `juno lora` (`docs/howto.md:63`). A fixed 32-token chunk pays
   the per-chunk fixed cost 16 times for a 512-token prompt. Extend adaptive sizing to the surfaces
   that are still pinned at 32 where the VRAM query that drives it is meaningful, and make the
   per-chunk fixed cost itself smaller (the pinned host-staging work already published at
   `docs/perf-compare/20260918T153900Z-prefill-adaptive/` measured -30% prefill and -27% request wall
   on a real 488-token mistral-7b prompt — continue that line rather than restarting it).
4. **Residual prefill attention.** On TinyLlama, turning the kernel on dropped attention from 78.7%
   to 11.0% of prefill wall time, so on Llama-family models item 0 has already taken most of this —
   but that is the one architecture where the kernel was already the default, so nothing here is
   banked for the four this tier newly covers. Re-read item 1's post-item-0 breakdown and confirm,
   **per architecture**, whether attention is still a material share of prefill at 512 tokens. If it
   is, state plainly that the remainder belongs to Tier 02's tiled kernel and record the number so
   Tier 02 can be held to it; if item 0 already reduced it to noise, record that too, so Tier 02 is
   not later credited with a prefill win that this tier already banked. Do not extrapolate the
   78.7%-to-11.0% figure onto any architecture that was not measured — it is one model's result.

### Out of scope

- CPU prefill throughput. The CPU pp ratio is 0.050 to 0.099 (`docs/perf-compare/20260918T031702Z/`),
  a serious gap, but its causes are the allocation, threading and kernel issues Tier 10 owns after
  its scope was widened. Re-measure CPU pp at the end of Tier 10, not here.
- ROCm prefill. `RocmMatVec` has no `sgemm` override at all, so ROCm prefill is a serial GEMV loop
  today; closing that is Tier 10 item 1 (ROCm tiled-GEMM), which is where the hardware-gated work is
  already concentrated. This tier must not leave the ROCm path worse than it found it.
- New quantization formats (Tier 04) and new architectures (Tier 08).

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | must not regress; CPU prefill throughput itself is Tier 10's, but the correctness oracle and the fixed-`32` chunking default both live here |
| 2 | CUDA GPU inference | primary target |
| 3 | ROCm GPU inference | N/A for the new batched paths (CUDA-only this tier); must verify ROCm's existing serial-GEMV prefill still works and is not regressed by any shared dispatch change. **Not N/A for item 0**: ROCm resolves `--gpu-attention` to scalar today, and this tier must either port the kernel (`NEEDS-AMD-HARDWARE`) or make the fallback explicit and announced rather than silent |
| 4 | Static schedule | primary target — this is where adaptive whole-prompt prefill already runs |
| 5 | Continuous schedule | chunked prefill mixed with decode is the continuous engine's core loop; any per-chunk cost reduction must be verified there too, and the fixed-`32` default re-examined |
| 6 | Single-node local mode | primary dev/test surface |
| 7 | Pipeline-parallel cluster | prefill chunks cross the gRPC boundary per shard — confirm the per-chunk fixed cost being reduced here is not simply relocated into serialization |
| 8 | Tensor-parallel cluster | same |
| 9 | LoRA training | training prefills its own microbatches; confirm unaffected, or improved, but not silently changed in numerics. Training's existing `--gpu-attention` exemption (attention stays scalar CPU, REPL warns) is **re-verified and kept** by item 0, not quietly swept into the new default |
| 10 | LoRA playback | playback prefills a LoRA-modified prefix; confirm the delta-add still composes correctly against a wider batched path. Same `--gpu-attention` exemption as row 9 — re-verified, kept, and still warned about |
| 11 | Vision | `VisionEncoder` runs the widest batches in the system (B around 741) and reuses the same `MatVec` primitives — a batched-dispatch change is exactly the kind that regressed vision once before; run `compare-vision.sh` as a required gate, not an optional one. Vision already inherits `--gpu-attention` by delegating to `LlamaTransformerHandler`, so item 0 changes nothing for it directly — verify that stays true rather than assuming it |
| 12 | OpenAI REST surface | time-to-first-token is the user-visible form of this tier's metric; measure TTFT, not only aggregate pp |
| 13 | Native REST surface | same |
| 14 | CLI | `--prefill-batch` semantics change on any surface where the default moves off `32`; this is user-visible and must be documented |

## Implementation steps

1. Re-baseline first. Run `compare-llama-cpp.sh --gpu` at `n_prompt` 128 and 512 on all four sweep
   models on current HEAD (post-Tier-01), under the parity-corrected harness required by this plan's
   "Benchmark parity preconditions" (README), and with each lane's actual resolved
   `--gpu-attention` value recorded rather than assumed — the historical numbers quoted above were
   taken before those corrections, with at least one lane mislabelled, and are not a valid
   before-measurement for this tier's gate.
2. **Land item 0 next, before anything else in this tier.** Not because it is cheap — it is four
   kernel integrations against four handlers that each own their own KV map and attention math, plus
   a capability-reporting mechanism that does not exist yet — but because leaving it until later
   would mean every subsequent measurement in this tier is taken against a baseline that is fast on
   four architectures and slow on four others. Re-measure immediately after, so the default change
   has its own attributable number, and record that number per architecture: the Llama-family 3.85x
   says nothing about what these four will do.
3. Produce the per-term prefill breakdown (scope item 1) on the post-item-0 build and publish it.
   Decide which of items 2-4 the breakdown actually justifies, and record the decision in this file
   — item 0 may well have moved which term dominates, which is the point of sequencing it here. The
   expected ranking going in is that host-device staging (item 2) dominates once attention is on the
   GPU, since a 512-token window moves roughly 8 MB each way per matmul; if the breakdown says
   otherwise, follow the breakdown.
4. Implement in the order the breakdown ranks, largest term first.
5. Re-measure after each change rather than only at the end, so a negative result is attributable to
   one change instead of the batch.
6. Run the full cross-surface smoke matrix, including the vision gate.

## Tests to write/upgrade before implementation

- **New GPU-attention parity tests, one per newly covered architecture** (Phi-2, Phi-3, Qwen3,
  Qwen3-MoE): kernel output against that handler's scalar CPU attention within stated tolerance, and
  a greedy-decode divergence characterisation on a real model — how often, and by how many tokens,
  the FP16 KV mirror diverges. This is the evidence that decides whether that architecture's default
  flips on; do not inherit the Llama-family answer for it.
- **New GPU-attention capability tests.** Note what these can and cannot be: `GpuAttentionOptions`
  has no architecture awareness today (`AUTO` is just `CudaAvailability.isAvailable()`), so there is
  no resolver to test until item 0 builds one. The tests are therefore (a) each covered handler
  reports its GPU-attention capability truthfully through the new mechanism, (b) an uncovered
  handler or backend produces the documented explicit notice rather than a silent scalar fallback,
  (c) `off` still gives the bit-identical CPU-parity baseline, and (d) the LoRA training and
  `--lora-play` exemptions still hold and still warn.
- **New residency/staging tests for item 2**: a prefill window produces output equal within float
  tolerance to today's stage-per-matmul path, at window sizes spanning the `HALF_SGEMM_BATCH_MAX = 8`
  dispatch threshold (1, 2, 8, 9, 32, 512) and at the vision batch width (B around 741); device
  memory is not leaked across repeated windows (reuse the accounting assertion Tier 01 adds); and
  the non-allocating batched form produces bit-identical results to the allocating one for every
  backend implementation.
- **A staged-bytes assertion**: instrument H2D/D2H bytes per prefill window and assert the
  post-item-2 figure against the pre-item-2 baseline, so item 2's win is measured in bytes moved and
  not only in wall time.
- **New `PrefillBatchOptions` tests** for any surface whose chunk-sizing default changes: assert the
  new default per surface, and assert that an explicit `--prefill-batch N` still overrides on every
  surface.
- **`ModelLiveRunnerIT`**: add a long-prompt (512-token) prefill check asserting correct output, for
  both schedules.
- **New bash smoke script**: `scripts/performance-tests/smoke-tier01b-prefill.sh` — drives
  `/v1/chat/completions` with 128-, 512- and 2048-token prompts against `tinyllama` and `mistral-7b`,
  on both schedules, asserting correct output and recording TTFT; and asserts greedy-decode output is
  identical to the pre-tier build for the same prompt and seed.
- **Perf gate (required)**: this is by definition a forward-pass change. `compare-lora.sh`,
  `compare-vision.sh` (mandatory here, not optional — vision runs the widest batches in the system),
  `compare-prefill-batch.sh`, and `compare-llama-cpp.sh` at both `n_prompt` 128 and 512; publish under
  `docs/perf-compare/<timestamp>-tier01b-prefill/`.

  Every number below is a median of at least three runs with min/max published, per the README's
  noise-floor rule — this host resolves to about ±15%, and two of these thresholds sit inside that.

  **Threshold, item 0 on its own.** Every architecture newly covered by the default change (Phi-2,
  Phi-3, Qwen3, Qwen3-MoE) must show a measured prefill gain against its own pre-item-0 baseline on
  the same host and prompt length — a default flipped on that buys nothing on a given architecture is
  a finding to report and investigate, not a checkbox. **There is no reference point for these four.**
  The 3.85x figure is TinyLlama's, on the one architecture family where the kernel was already the
  default, so it predicts nothing here; record what each of the four actually does. Decode is also
  expected to move on these architectures, since attention has been measured at 64.2% of decode wall
  time at ctx around 512; a flat decode result here is a signal that the kernel is not actually being
  taken, not a pass.

  **Threshold, item 2 on its own.** Bytes staged host-to-device and device-to-host per 512-token
  prefill window must drop by **>= 70%** against the step-1 baseline. This is the item's primary
  number because it is the one that is not confounded by clock state or noise: a residency change
  either stops moving the bytes or it does not.

  **Threshold, the tier overall.** mistral-7b Q4_K_M on GPU must reach **pp ratio >= 0.10x**
  llama.cpp at `n_prompt=512` **and** pp must not fall with prompt length — the 512-token ratio must
  be greater than or equal to the 128-token ratio for every sweep model. Both are measured under the
  parity-corrected harness (`RAW_PROMPT=1`, warmup, median of three), and **both are re-derived from
  step 1's re-baseline before implementation starts**, because neither has ever been measured
  like-for-like. Understand the size of the 0.10x ask honestly: the best parity-corrected prefill
  figure in this repository is TinyLlama at 0.028x with the attention kernel on, so 0.10x is roughly
  a three- to four-fold improvement over the best result this project has produced, not a six-fold
  improvement over a 0.0166x reading that was partly an artefact. If step 1's re-baseline puts
  mistral-7b materially below 0.028x, restate this threshold against the re-baselined number and say
  so here rather than carrying a target that was set against the wrong denominator. Decode must not
  regress: tg ratio within 0.95x of the step-1 baseline for every sweep model. Vision gate per the
  existing rule: `latency_ms` <= 1.25x baseline, decode tps >= 0.80x baseline.

  **Contingency, in the same spirit as Tier 01's.** If the breakdown in step 3 shows prefill time is
  dominated by a term this tier cannot move without work owned by a later tier (for example: residual
  attention at long context, which is Tier 02; or per-format kernels, which is Tier 04), do not
  iterate indefinitely. Publish the breakdown, ship whatever items the breakdown does justify, state
  the measured ratio honestly, and mark the tier **partial-complete** with a named successor tier for
  the dominant term — then escalate to the user, since that re-scopes another tier. Item 0 is
  excluded from this contingency: it ships regardless, because removing a silent per-architecture
  degrade is worth doing whatever the measurement says. Item 2 has its own escalation path instead
  of this one — if Tier 01's residency primitive did not ship, item 2 does not proceed on a
  substitute design; escalate.

## Models needed

The four standing sweep models (`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`,
`qwen2.5-3b-instruct-q4_k_m.gguf`, `Phi-3.5-mini-instruct-Q4_K_M.gguf`,
`mistral-7b-instruct-v0.1-q4_k_m.gguf`) plus `moondream2-q5_k.llamafile` for the vision gate cover
items 1 through 4. All present.

**Item 0 has a real model gap, and it blocks part of its own threshold.** Per
[`INVENTORY.md`](INVENTORY.md), of the four architectures whose default this tier flips on:

| Architecture | Testable today | With what |
|---|---|---|
| Phi-3 | yes | `Phi-3.5-mini-instruct-Q4_K_M.gguf` |
| Phi-2 | partly | only via `moondream2-q5_k.llamafile`'s phi2 backbone, which exercises it through the vision path rather than plain text generation |
| Qwen3 | **no** | only `qwen35` is on disk, and Tier 00 made that one fail closed by name |
| Qwen3-MoE | **no** | no `qwen3moe` file on disk at all |

So item 0 can ship kernel support for all four, but can only produce the required per-architecture
measured gain for Phi-3, and partially for Phi-2. Ask the user for a plain Phi-2 GGUF, a plain
`qwen3` (non-3.5, non-MoE) GGUF, and a working `qwen3moe` GGUF **at the point this tier starts** —
these are the same three files [`INVENTORY.md`](INVENTORY.md) already lists as missing for Tiers 06
and 08, so obtaining them here pays for those tiers too. If they are not available, Qwen3 and
Qwen3-MoE ship the kernel path with unit-test coverage only and their **defaults stay resolved off**
with the explicit notice from item 0 rather than being flipped on unmeasured — turning on an
unvalidated default is exactly the silent-degrade this item exists to remove.

## Exit criteria

- [ ] `--gpu-attention` defaults to on for every architecture whose gain was actually measured on a
      real model (Llama-family, Mistral, Qwen2, Phi-3 at minimum), and any architecture that could not
      be measured for want of a model file keeps its default resolved off behind the explicit notice
      rather than being flipped on unvalidated — with the missing file named. The ROCm answer decided
      and either implemented (`NEEDS-AMD-HARDWARE`) or made an explicit announced fallback. No path
      resolves to scalar silently, whichever way each one landed.
- [ ] Each newly covered architecture shows a measured prefill gain against its own pre-item-0
      baseline, or is documented as unmeasured with the reason and the missing model named.
- [ ] Greedy-decode divergence from the FP16 KV mirror characterised per newly covered architecture
      and documented in `docs/howto.md` and `--help`, with `off` retained as the bit-identical
      CPU-parity baseline.
- [ ] LoRA training and `--lora-play` exemptions re-verified as still holding and still warning.
- [ ] `compare-llama-cpp.sh`'s stale default-lane `--gpu-attention` labelling corrected, and the
      first re-baselined run states which previously published lanes were mislabelled.
      `docs/performance.md`'s GPU-resident-attention section is corrected in the same pass: it states
      the default is **off** while `GpuAttentionOptions.fromEnv()` defaults to `auto`.
- [ ] Item 0's own attributable measurement published separately from the rest of the tier's, so the
      default change's effect is visible on its own.
- [ ] Per-term prefill breakdown published for all four sweep models at `n_prompt` 128 and 512, with
      no unattributed residue — every term named, including host-device staging and dequantization.
- [ ] Prefill activations stay device-resident across a layer's projections, with the materialization
      boundary documented; bytes staged per 512-token window down >= 70% against the step-1 baseline;
      `sgemmLayerInto`'s per-matmul allocate-and-copy removed via the non-allocating batched form,
      whose contract matches the one Tier 10 item 4 adds for CPU.
- [ ] Chunk-sizing defaults reviewed per surface; any surface still pinned at `32` has a measured
      reason, not an inherited one.
- [ ] Prefill throughput no longer degrades with prompt length: the `n_prompt=512` pp ratio is greater
      than or equal to the `n_prompt=128` pp ratio for every sweep model — **both measured under
      `RAW_PROMPT=1` with the same `--vector` setting**, which no published pair of runs has ever
      been. Step 1 establishes whether the degradation is real before this criterion can be scored.
- [ ] Threshold above met, or the tier is explicitly marked partial-complete with the dominant term
      named and assigned to a successor tier (not silently marked complete).
- [ ] Decode (tg) and vision both verified not regressed, with published numbers.
- [ ] Cross-surface checklist fully resolved.
- [ ] Docs (`docs/howto.md` for any `--prefill-batch` default change, `docs/performance.md`,
      `docs/agent-arch.txt`) updated, Juno-native language only.
- [ ] `CHANGELOG.md` entry added.
