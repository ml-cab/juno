# Tier 01: GPU activation-residency redesign

Status: not started
Gap analysis refs: §1.6, §2.8

## Objective

Fix the single architectural root cause behind three independently-measured regressions
(GPU-resident RMSNorm/RoPE/SwiGLU, draft-model speculative decoding, and the CPU SIMD hot path
rejection): every GPU op in `MatVec`/`GpuBindings` today is host-`float[]`-in/host-`float[]`-out,
so chaining GPU ops within a single transformer layer pays a host round trip between every op
regardless of whether each individual op is cheap on-device. This tier builds the minimum
activation-residency primitive needed to keep an activation vector/batch on-device across a
*chain* of ops (norm → projection → attention → norm → FFN) within one layer, and re-measures the
previously-shelved GPU-resident RMSNorm using it, as the tier's proof point.

## Why this tier, why now

This is the highest-leverage architectural investment identified in the gap analysis (§1.6): it is
a prerequisite for GPU-resident RMSNorm/RoPE/SwiGLU (§2.8, currently dormant scaffolding), for
draft-model speculative decoding to stop regressing (§1.5, currently 0.52x), and for any future
fused-attention kernel (Tier 02 depends on being able to keep QK^T and softmax intermediate state
on-device). Doing it once now, immediately after Tier 00's correctness pass, means Tiers 02-13 can
build GPU-side features without independently re-discovering and re-shelving the same dispatch-cost
wall three more times.

## Scope

### In scope

1. A new on-device activation-residency abstraction (name TBD during design —
   e.g. `DeviceActivation`/`ResidentTensor`) that represents a batch of activation vectors living in
   device memory across multiple consecutive GPU ops, only materializing back to host `float[]`
   when a downstream consumer that isn't yet GPU-resident needs it (e.g. handing off to a CPU-side
   sampler, or to a node that will forward the activation to a different process over gRPC).
2. Wiring this through the one existing, already-tested-but-dormant op: `CudaRmsNorm`/
   `CudaGraphSession`. Re-measure RMSNorm decode latency with residency in place, on the same
   GTX 1080/TinyLlama/Mistral-7B workloads `docs/performance.md`'s Phase B analysis used, to confirm
   the 6.9x-slower-than-CPU-scalar finding is actually fixed by residency (not just "somewhat
   better") before calling this tier done.
3. Extending residency to at least one additional op in the same layer (RoPE is the natural next
   op after the Q/K/V projection) to prove the chain — not just a single isolated op — is what
   fixes the round-trip cost. `RopeKernel`, mentioned as "never built" in the gap analysis, gets
   built here, using the residency primitive from day one rather than the old ad-hoc per-op
   pattern.
4. A clear, tested boundary: where does residency start and end within one decode step? (e.g.
   resident from post-embedding-lookup through the FFN output, materializing to host only at the
   point the residual stream needs to leave the GPU — for the LM head, for gRPC handoff in cluster
   mode, or for CPU-side sampling.) Document this boundary since every later tier that adds a
   GPU-side op needs to know whether it should participate in residency or not.

### Out of scope (explicitly deferred)

- Full-layer GPU residency for every op (attention, FFN, residual-add) — this tier proves the
  pattern on norm+RoPE; extending it to the full layer for every transformer handler
  (`LlamaTransformerHandler`, `Phi2TransformerHandler`, `Phi3TransformerHandler`,
  `Qwen3TransformerHandler`, `Qwen3MoeTransformerHandler`) is real follow-on work sized into later
  tiers as they touch each op (Tier 02 for attention, Tier 06 for speculative decoding's
  draft-model forward passes).
- Re-fixing draft-model speculative decoding's 0.52x regression — that's Tier 06's job, but Tier 06
  depends on this tier's primitive existing first.
- ROCm-side residency — CUDA only for this tier (ROCm has no CUDA-graph equivalent readily
  available; a ROCm residency design is a separate, later decision, tracked in Tier 10).
- Multi-GPU/cross-process residency — out of scope; this is single-device, single-process only.

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | must be provably unaffected — residency is a CUDA-only code path; scalar CPU path is the correctness oracle for comparison |
| 2 | CUDA GPU inference | primary target; must show a measured, not just theoretical, improvement over the Phase B baseline |
| 3 | ROCm GPU inference | N/A this tier — explicitly out of scope, must not regress ROCm's existing (already GEMV-only) behavior |
| 4 | Static schedule | residency must work for the existing static micro-batch path (batch size ≥ 1) |
| 5 | Continuous schedule | residency must work across mixed prefill/decode steps — verify separately, since `ContinuousBatchEngine` calls `forwardBatch`/`prefillBatch` differently than the static path |
| 6 | Single-node local mode | primary dev/test surface |
| 7 | Pipeline-parallel cluster | activation must still correctly materialize to host at the point it's serialized over gRPC to the next node — this is the clearest test that the residency boundary is correctly drawn |
| 8 | Tensor-parallel cluster | same materialization requirement, at the AllReduce boundary |
| 9 | LoRA training | training's frozen-weight forward pass currently stays FP16/FP32 host-resident by design (`LoraMmqPolicy`) — confirm residency doesn't change training numerics; training keeps ignoring `--mmq`/GPU-attention as before, unaffected |
| 10 | LoRA playback | LoRA delta add-on (`applyLoraInPlace`) happens after the base projection — confirm it correctly reads a materialized (or residency-aware) activation, not stale host data |
| 11 | Vision | `VisionEncoder` reuses the same `MatVec` primitives — confirm residency doesn't silently apply to vision's very different batch shapes (vision batches are wide, ~741, vs. decode's batch=1) without being re-validated there first; if it's not extended to vision this tier, it must be explicitly bypassed, not accidentally partially applied |
| 12 | OpenAI REST surface | end-to-end chat completion latency/output must be byte-identical to pre-tier for CPU, and correct (not necessarily identical, since GPU float ops can have different rounding) for GPU |
| 13 | Native REST surface | same |
| 14 | CLI | `./juno local`/`cluster` unaffected in flags/UX; this is an internal change |

## Implementation steps

1. Write the correctness/perf test harness first (see Tests below) so the "regresses decode" claim
   can be re-measured objectively before any residency code exists — this gives a before/after
   baseline.
2. Design and implement the minimal residency primitive, scoped to RMSNorm + RoPE only.
3. Wire `CudaRmsNorm` and the new `RopeKernel` through it inside `LlamaTransformerHandler`'s decode
   path (the only handler with a fused GQA attention kernel today), gated behind the existing
   `--gpu-attention`-style opt-in flag pattern so it can be toggled off if something regresses.
4. Re-run the Phase B-style microbenchmark (batch=1, dim=2048, N iterations) and the full
   `compare-lora.sh`/model-sweep perf gate; publish results under `docs/perf-compare/`.
5. Only after the measured win is confirmed, wire the flag on by default and update
   `docs/howto.md`/`docs/performance.md` accordingly (Juno-native language, no competitor names).

## Tests to write/upgrade before implementation

- **Unit tests**: extend `CudaGraphSessionTest`/add `CudaRmsNormTest` cases that assert
  bit-for-bit-or-within-tolerance correctness of the residency-backed path against the existing CPU
  scalar RMSNorm and the existing ad-hoc (non-resident) `CudaRmsNorm`, on real GPU hardware.
- **New unit test** for the residency primitive itself: allocate, chain two ops, materialize,
  confirm no leaked device memory (add a device-memory-accounting assertion — Juno already tracks
  VRAM pressure per `GpuBindings.memGetInfo`, reuse that).
- **`LlamaTransformerHandlerVerifyParityTest`** (already exists for speculative decoding verify
  parity) or a new sibling test: confirm decode output is unchanged with residency on vs. off for a
  fixed seed/greedy run.
- **`ModelLiveRunnerIT`**: add a GPU-path check (if not already gated to GPU-only hosts) asserting
  correct output with the new flag on, using `tinyllama` and `mistral-7b`.
- **New bash smoke script**: `scripts/performance-tests/smoke-tier01-gpu-residency.sh` — runs
  `./juno local` with the residency flag on/off against `tinyllama`, `mistral-7b`, and
  `llama-1-30b` (the large model, to catch any VRAM-accounting regression under memory pressure),
  diffing greedy-decode output between the two flag states (must be identical or within documented
  floating-point tolerance) and asserting the on-state doesn't crash or leak VRAM across repeated
  requests.
- **Perf gate (required — this is a hot-path change)**: `compare-lora.sh` plus a
  targeted RMSNorm/RoPE microbenchmark rerun of the existing Phase B methodology; publish under
  `docs/perf-compare/<timestamp>-tier01-residency/`.

## Models needed

`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`, `mistral-7b-instruct-v0.1-q4_k_m.gguf`, and
`llama-1-30b.Q4_K_M.gguf` are already present and sufficient. No downloads needed.

## Exit criteria

- [ ] Residency primitive implemented, unit-tested, and documented (what it is, where the
      materialization boundary is, which ops participate).
- [ ] RMSNorm + RoPE measured *faster* than CPU scalar (or at minimum, no longer the ~7x-slower
      finding from Phase B) with residency, on real GTX 1080 hardware, published in
      `docs/perf-compare/`.
- [ ] No correctness regression: greedy decode output identical (CPU) or within tolerance (GPU)
      with the new path enabled vs. disabled, across all three cross-surface-listed models.
- [ ] Cluster (pipeline- and tensor-parallel) smoke tests confirm activations still correctly
      materialize at the process/AllReduce boundary — no stale or device-resident data crossing a
      gRPC call.
- [ ] LoRA train + playback smoke tests unaffected.
- [ ] `docs/agent-arch.txt`/`docs/performance.md`/`docs/howto.md` updated (Juno-native language).
- [ ] `CudaGraphSession`/`CudaRmsNorm` are no longer "dormant scaffolding" — either wired live
      (preferred, if the measurement confirms the fix) or the tier is not marked complete.
- [ ] Full `mvn test`/`mvn verify -pl juno-master` pass with zero regressions.
- [ ] `CHANGELOG.md` entry added.
