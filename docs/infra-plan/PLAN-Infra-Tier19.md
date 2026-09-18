# Tier 19: GPU-Resident Elementwise Ops + Launch-Fusion (Decode Host/FFI Overhead)

**Status: In progress — Phase A step 1 (JFR baseline) done; step 2 (`RmsNormKernel`) built,
parity-tested, and measured on real hardware, but deliberately left dormant (not constructed by
`LlamaTransformerHandler`) after a live A/B showed the per-call GPU round trip costs ~11x more than
the CPU path it would replace. See "Measured finding" below before continuing to Rope/ResidualAdd/
SwiGlu with the same per-op pattern.**

## Measured finding (2026-09-18) — per-op GPU dispatch regresses without residency

`RmsNormKernel`/`CudaRmsNorm` were implemented exactly per this doc's original plan (new PTX kernel,
Java binder mirroring `GqaAttentionKernel`, `CudaRmsNorm` handler-facing wrapper mirroring
`CudaGqaAttention`, parity-tested against `LlamaTransformerHandler.rmsNorm` on real GTX 1080
hardware — see "Progress log" below for full detail) and wired into all three of
`LlamaTransformerHandler`'s forward paths behind a `rmsNormGpu != null` dispatch, exactly as this
doc's "Chosen design" section specified: no CLI flag, auto-activating whenever the handler is
already on the CUDA-resident weight path.

A live end-to-end run (`./juno local --model-path models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
--temperature 0`, greedy, real GPU) confirmed **correctness** — output text byte-identical to the
CPU-only path — but a controlled A/B (same prompt, same build, only `CudaRmsNorm.tryCreate(...)`
toggled) measured `juno.RmsNorm` decode p95 at **0.168ms with the GPU path active vs 0.015ms
CPU-only — an ~11x per-call regression**, and total `juno.RmsNorm` decode cost across the run rose
from 10.6ms to 122.7ms. This is not noise (the two figures are ~11x apart, controlled by nothing but
the one flag); it is the direct, physically-expected cost of an independent
H2D-upload/kernel-launch/D2H-download round trip (a few tens of microseconds of fixed CUDA dispatch
+ copy latency, per `PLAN-Infra-PERF-ANALYSIS.md`'s own Nsight finding this tier formalizes) applied
to an operation whose CPU cost is sub-microsecond for a ~2048-4096-element vector.

**Root cause, and why it invalidates this doc's original "no CLI flag, auto-activate" design
decision:** every `CudaMatVec.sgemv`/`sgemm` entry point takes and returns **host** `float[]`
arrays — there is no device-pointer-accepting overload today (confirmed by this session's own
research pass before writing the kernel). So `rmsNorm`'s GPU output must round-trip to host before
the very next GEMV call re-uploads it anyway. Moving **one** op to the GPU while its neighbors on
both sides still do independent host round trips does not remove a round trip — it **adds** one,
for an operation that was nearly free on the CPU to begin with. The Nsight finding that motivated
this tier (GPU idle 73-82% of decode wall time) is real, but this increment's "one kernel at a time,
same round-trip-per-call shape as `Q4KMmqKernel`/`GqaAttentionKernel`" approach does not address the
idle time's actual cause for a **cheap** op — it would only pay off for a chain of GPU ops that stay
device-resident across the chain (no intermediate host round trip), which does not exist yet and is
a materially bigger change (new device-pointer-accepting `sgemv`/`sgemm` overloads, or an
activation-residency abstraction) than "port each op independently."

**Decision:** `CudaRmsNorm` is **not constructed** by `LlamaTransformerHandler` (the field stays
`null` unconditionally, with the reasoning inlined as a code comment at the call site) — the default
path is scalar CPU `rmsNorm`/`rmsNormInto`, unchanged from before this session. The kernel, binder,
and wrapper are kept (correct, tested, real infrastructure) for whenever the residency gap is
closed, rather than deleted — but this session does **not** proceed to implement `RopeKernel`/
`ResidualAddKernel`/`SwiGluKernel` with the same per-op-independent-round-trip pattern, since doing
so would predictably compound the same regression three more times for no measured benefit. This
needs a design decision (see "Recommended next step" below) before Phase A continues.

**Recommended next step for whoever picks this up:** before writing more elementwise kernels, either
(a) add device-pointer-accepting `sgemv`/`sgemm` overloads (or an equivalent activation-residency
abstraction) so a chain of ops — norm → QKV projection → rope → attention → out-proj → residual →
norm → gate/up projection → swiglu → down-proj → residual — can stay on-device end to end within one
layer, only round-tripping at layer boundaries (if at all), or (b) treat Phase A as not viable as
originally scoped and re-prioritize Phase B (CUDA graph capture/replay) or a different P0 lever,
since Phase B's launch-fusion premise has the same "must not add a round trip" constraint. Either
way, do **not** re-run this increment's "one kernel, host round trip either side" pattern for the
remaining three ops without first fixing this.

## Phase B tried early, on the single op — confirms and sharpens the finding (2026-09-18)

Rather than guess which of (a)/(b) above to pursue, this session tested (b) directly and cheaply: a
new `CudaGraphSession` (CUDA driver graph capture/replay — `cuStreamBeginCapture`/`cuStreamEndCapture`/
`cuGraphInstantiate`/`cuGraphLaunch`, new bindings in `CudaDriverBindings`) captured `RmsNormKernel`'s
exact H2D/kernel/D2H sequence once, then replayed it in a tight loop with fresh input data written
into the same fixed host staging buffers each time (correctness verified: `CudaGraphSessionTest`,
3 cases, including replaying the same captured graph across three different random inputs and
checking each against `LlamaTransformerHandler.rmsNorm` within `1e-4f`).

A standalone microbenchmark (batch=1, dim=2048, 2000 iterations after warmup, isolated from the full
pipeline so the comparison is clean) measured:

| Path | us/call | vs CPU |
|---|---|---|
| CPU scalar (`LlamaTransformerHandler.rmsNorm`) | 15.6 | 1.0x |
| GPU ad-hoc per-call (`CudaRmsNorm`, today's dormant path) | 107.0 | 6.9x slower |
| GPU CUDA-graph replay (`CudaGraphSession`) | 33.4 | 2.2x slower |

Graph replay recovers **~3.2x** of the ad-hoc overhead (107us → 33us) — meaningful, and confirms the
CUDA driver dispatch/marshalling cost this tier's Nsight finding pointed at is real and reducible.
But it does **not** close the gap to CPU: **33us is still 2.2x the 15.6us CPU cost** for this one op
in isolation. The remaining ~33us is believed to be the irreducible cost of `cudaStreamSynchronize`
itself (the CPU blocking until the GPU finishes) — graph replay collapses the *dispatch* of multiple
driver calls into one `cuGraphLaunch`, but a single graph containing one op still needs exactly one
sync point before its result can be read back and used, same as the ad-hoc path's final `D2H` sync.

**Sharpened conclusion:** launch fusion (Phase B) only pays for itself when it amortizes that one
sync-point cost across a **chain of many** operations (a full layer, or more), not when applied to a
single op read back immediately. This is the same requirement as recommendation (a) above (chain
multiple GPU-resident ops without an intermediate host round trip) — Phase B does not offer a
shortcut around it; it *adds value once that chain exists*, by replacing that chain's many potential
sync points with one. `CudaGraphSession` is kept as verified, reusable infrastructure for that future
work (it correctly captures/replays an arbitrary async op sequence on fixed buffers, is not
`RmsNormKernel`-specific), but — consistent with the decision above — is **not** wired into
`LlamaTransformerHandler`; there is currently nothing resembling a same-op-count-reducing chain to
apply it to.

## Progress log

**2026-09-18 — Phase A step 1 (JFR baseline instrumentation).** Added four new JFR events —
`juno.RmsNorm`, `juno.Rope`, `juno.ResidualAdd`, `juno.SwiGlu` (new classes in `node`:
`RmsNormEvent`/`RopeEvent`/`ResidualAddEvent`/`SwiGluEvent`, mirroring `AttentionEvent`'s shape:
`windowSize`/`startPosition`/`dimension` fields, `@Category({"Juno","Inference"})`) — wrapping the
existing scalar CPU call sites in `LlamaTransformerHandler`'s three forward paths
(`transformerLayer` single-token decode, `transformerLayerBatch` batched prefill,
`transformerLayerMultiDecode` `--parallel` multi-stream decode). One event per call site covering
the whole batch/window (matching `AttentionEvent`'s per-layer granularity, not per-row), except
`ffn()`'s residual-add-adjacent split: two `RmsNormEvent`/`ResidualAddEvent` calls per layer
(attention sub-layer, FFN sub-layer) and one `RopeEvent`/`SwiGluEvent` call per layer. No behavior
change — measurement only. `JfrMetricsExtractor` aggregates all four into the same
`{prefill,decode}.{count,p95_ms,total_ms}` shape as `juno.Attention`, via a new shared
`DurationBucket`/`isWindowPrefill` helper pair that the existing `ATTENTION` case was also
refactored onto (identical output, less duplication). Tests: `JfrMetricsExtractorElementwiseOpsTest`
(16 cases, parameterized over the four event names, mirroring
`JfrMetricsExtractorAttentionTest`'s scenario matrix) and
`LlamaTransformerHandlerElementwiseOpsJfrTest` (2 cases: batched prefill and single-token decode,
asserting exact per-layer event counts and field values). Full `node`+`metrics` suites green (534 +
38 tests, 0 failures) after the change.

Scope note: instrumented `LlamaTransformerHandler` only (Llama-family, Mistral, Qwen2 — the same
architecture scope Phase A's GPU kernels will target first). `Phi3TransformerHandler` and
`Qwen3TransformerHandler`/`Qwen3MoeTransformerHandler` have their own call sites that delegate to
the same shared static methods (`LlamaTransformerHandler.rmsNorm`/`rope`/`silu`/`add`) but were not
separately wrapped with events this pass — named follow-up, not silently implied covered, consistent
with this tier's own Phi-3/Qwen3 GPU-kernel scoping. `Phi2TransformerHandler` uses entirely different
primitives (`layerNorm`+bias, `Phi2Rope.ropePartial`, GELU, no gate tensor — no RMSNorm/SwiGLU family
at all) and is out of scope for this tier regardless.

Pre-existing, unrelated finding: `JfrMetricsExtractorAttentionTest.mixedPrefillAndDecode_aggregatesIndependently`
has a latent floating-point exact-equality flake (`isEqualTo` on a `total_ms` computed two different
summation orders over real, non-fixed JFR-measured nanosecond durations) — reproduced on the
unmodified base branch before this session's changes, so not a regression introduced here. Not fixed
as part of this tier (out of scope); the equivalent new assertion in
`JfrMetricsExtractorElementwiseOpsTest` uses `isCloseTo` with a small offset instead, so the new test
doesn't inherit the same flake.

**2026-09-18 — Phase A step 2 (`RmsNormKernel`), built and measured, left dormant.** New files:
`node/src/main/cuda/rms_norm.cu` (block-per-row sum-of-squares reduction + normalize, mirroring
`gqa_attention.cu`'s `block_reduce` pattern, `RMSNORM_THREADS=128`), hand-compiled to
`node/src/main/resources/cab/ml/juno/node/rms_norm.ptx` via `nvcc -ptx -arch=compute_61 -O3`
(toolchain confirmed present in this environment: CUDA 12.0, real GTX 1080); `RmsNormKernel.java`
(PTX load/launch, mirrors `GqaAttentionKernel.java` exactly) and `CudaRmsNorm.java` (handler-facing
batch wrapper with grow-on-demand device scratch, mirrors `CudaGqaAttention.java`). Tests, all green
on real hardware: `RmsNormKernelParityTest` (3 cases: single-row, batched, non-power-of-two dim, all
matching `LlamaTransformerHandler.rmsNorm` within `1e-4f`) and `CudaRmsNormTest` (3 cases: null on
non-CUDA context, output-array reuse, scratch growth across calls). Wired into all three forward
paths via new `rmsNormGpuOrCpu`/`rmsNormIntoGpuOrCpu` dispatch helpers — **then measured live and
found to regress decode by ~11x per call** (see "Measured finding" above) and **deliberately left
uncreated** (`rmsNormGpu` stays null; the dispatch helpers and call-site wiring are kept as
correctness-verified scaffolding, but `CudaRmsNorm.tryCreate(...)` is never invoked in production
today). Default behavior is therefore unchanged from before step 2: confirmed via the same live
`./juno local` run showing byte-identical output and `juno.RmsNorm` decode p95 back at 0.014ms.

**2026-09-18 — Phase B tried early on the single op, before implementing more Phase A kernels.**
New `CudaGraphSession` (`node`): CUDA driver graph capture/replay wrapper, new bindings added to
`CudaDriverBindings` (`cuStreamBeginCapture`/`cuStreamEndCapture`/`cuGraphInstantiate`/
`cuGraphLaunch`/`cuGraphDestroy`/`cuGraphExecDestroy`, verified against the real `libcuda.so.1`
exported symbols in this environment — CUDA 12.0 driver). 3 new tests (`CudaGraphSessionTest`, all
green on real hardware) proving capture-once/replay-with-fresh-data-many-times correctness. A
standalone microbenchmark (not a checked-in perf-gate test — see "Phase B tried early" section above
for the full table) found graph replay cuts `RmsNormKernel`'s ad-hoc per-call overhead by ~3.2x
(107us -> 33us) but still leaves it 2.2x slower than the 15.6us CPU baseline, because a single op
still pays one unavoidable `cudaStreamSynchronize` round trip — the same conclusion as the
"Measured finding" section reached from a different angle (via the full live pipeline rather than an
isolated kernel loop). `CudaGraphSession` is generic (not `RmsNormKernel`-specific) and kept as
verified infrastructure; not wired into the handler, since there is no multi-op chain yet for it to
amortize a sync point across.

Not yet done: `RopeKernel`/`ResidualAddKernel`/`SwiGluKernel` don't exist. Per both the "Measured
finding" and "Phase B tried early" sections above, implementing them with the same per-op-independent-
round-trip pattern (with or without individual graph capture) is expected to reproduce the same
regression and is not recommended without first addressing GPU activation residency across chained
ops. The todo list's step 4 (Nsight re-run + bake-off) has not been reached and should not be, on the
current design, until that's resolved. Next: a design decision on activation residency (device-
pointer-accepting `sgemv`/`sgemm` overloads, enabling a real multi-op chained graph), not
implementation todo 3 as originally sequenced.

## Agent handoff

Read and follow `models/CLAUDE.md` before implementing:

1. Unit tests first, only for valuable business logic.
2. Implementation details designed with performance in mind.
3. Follow KISS. Prefer adding new Java classes over extending an existing one.
4. Update `docs/agent-arch.txt`, `docs/howto.md`, `README.md` when applicable.
5. No emojis; be strict and precise.
6. Output: list changed files for preview; never zip files back.

Also read, in this order:

- [`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) — Execution rules §1-§6 (one tier in flight,
  perf-compare gate, no tier labels shipped, all-architectures rule, interaction-matrix rule); P0
  step 8 detail; the "Nsight profiling pass done, finding reframes the lever" note this tier resumes
  from directly.
- [`PROMPT-P0-Gate.md`](PROMPT-P0-Gate.md) — the prompt this tier formalizes into a full plan. Its
  2026-09-18 addendum ("New evidence: the kernel is not the remaining bottleneck") is this tier's
  entire root-cause justification; do not re-derive it, and do not restart the tile-kernel-tuning work
  its own text downgrades.
- [`PLAN-Infra-PERF-ANALYSIS.md`](PLAN-Infra-PERF-ANALYSIS.md) → "Post-MMQ GPU idle-time finding
  (2026-09-18)" — full Nsight methodology and numbers.
- [`PLAN-Infra-Tier13.md`](PLAN-Infra-Tier13.md) — existing Q4 MMQ kernel/binder pattern
  (`q4k_gemv.ptx`, `Q4KMmqKernel`, `DeviceQ4KMatrix`, `CudaMatVec.sgemv`/`sgemvSameX`) this tier's new
  kernels should mirror, and 13C's `--gpu-attention` interaction matrix (Phi-3/Qwen3/ROCm gaps this
  tier does not close).
- `.cursor/rules/juno-no-infra-tier-labels.mdc`, `juno-docs-no-competitors.mdc`,
  `juno-infra-lora-perf.mdc`

### Code already landed (extend — do not rip out)

| Piece | Location |
|-------|----------|
| Q4 MMQ PTX / load pattern to mirror | `q4k_gemv.ptx`, `Q4KMmqKernel`, `DeviceQ4KMatrix` |
| CUDA dispatch | `CudaMatVec` (`sgemv`, `sgemvSameX`, `supportsQ4KMmq`) |
| Vendor-neutral dispatch | `GpuBindings.createMatVec()`, `GpuContext.selectBindings()` |
| Handlers with the scalar ops this tier targets | `LlamaTransformerHandler`, `Phi3TransformerHandler`, `Qwen3TransformerHandler` |
| `--gpu-attention` GPU-resident precedent (same pattern, different op) | Tier 13 Phase C |
| Bake-off harness | `scripts/performance-tests/compare-llama-cpp.sh` |

## Execution placement

| Field | Value |
|-------|-------|
| **Phase** | P0 step 8 (resumes the work `PROMPT-P0-Gate.md` diagnosed but did not implement; supersedes that prompt's "profile the kernel" framing per its own 2026-09-18 addendum) |
| **Depends on** | Tier 13 Phase B (Q4 MMQ, feature complete), Tier 13 Phase C (`--gpu-attention`, feature complete) — this tier targets the scalar ops those tiers left untouched, plus the launch/FFI overhead surrounding all of them |
| **Blocks** | Any further P0 tile-kernel tuning should stay parked (downgraded to ~27% ceiling per the Nsight finding) unless this tier's own re-measurement says otherwise |
| **Parallel with** | None — ROADMAP §1 allows only one Infra tier in flight. Vision/LoRA/Vector-SIMD parallel tracks are unaffected unless this tier changes shared forward-pass code, in which case they still run their own §2 compare independently |

## Overview

`PROMPT-P0-Gate.md`'s 2026-09-18 Nsight pass found that post-MMQ, the GPU sits **idle 73-82%** of
decode wall time; the `q4k_gemv`/`q5k_gemv`/`q6k_gemv` kernels themselves run in a reasonable
55-140 microseconds per launch and are not the bottleneck. A hypothetically zero-cost kernel could
only improve Phi-3.5 decode by ~27% at most — not enough alone to close 0.326x to the 0.5x P0 gate.
The idle time is believed to be host-side: `rmsNorm`, RoPE, residual-add, and SwiGLU
(`silu(gate)*up`) are confirmed by code inspection to still be plain scalar Java loops on the CPU —
no JDK Vector API, no GPU kernel, and (per the same finding) not even inside a JFR span today. Each
of the ~7 per-layer projections a decode token issues round-trips through one of these CPU ops
between GPU kernel launches, and each launch itself pays `Arena.ofConfined()` setup and
`MemorySegment` marshalling cost through Panama FFI. Neither cost shows up in the existing
`juno.MatVec`/`juno.Attention` JFR spans, which is why the historical "93-96% MatVec" read on the
idle time incorrectly.

This tier proposes exactly the two levers `PROMPT-P0-Gate.md` names and neither implements: (A) move
the elementwise ops onto the GPU so activations stay device-resident between projections, and (B)
reduce the number of discrete host round-trips per decode token via launch fusion / CUDA graph
capture-replay. **Explicitly not in scope:** restarting tile/`mul_mat_vec`-class kernel tuning on the
existing GEMV kernels — that lever is downgraded, not deprioritized-but-still-open.

## Scope and compatibility

Two additive, independently-gated phases. Phase B depends on Phase A within this tier (a captured
CUDA graph cannot contain a host-side scalar loop), so they are not parallel work — implement and
measure Phase A first, then decide whether Phase B is still needed.

### Phase A — GPU-resident elementwise ops

1. Instrument `rmsNorm`, RoPE, residual-add, and SwiGLU with JFR spans *first*, on the CPU path,
   before writing any GPU kernel. This is prerequisite measurement, not optional busywork — the
   Nsight finding inferred these ops' cost from idle time, not from a direct measurement, and Tier
   19's own before/after claim needs a real number to compare against.
2. Add CUDA kernels for each op (new PTX + Java binder classes, mirroring `Q4KMmqKernel`'s pattern —
   do not fold these into `CudaMatVec`, which is the GEMV/GEMM dispatch class, not an elementwise-op
   class). Land and parity-test one op at a time, in call-frequency order: `rmsNorm` (called at least
   twice per layer plus once at output), then RoPE, then residual-add, then SwiGLU.
3. Wire through `GpuBindings` so the dispatch stays vendor-neutral by name (CLAUDE.md: "New GPU
   functionality should go through `GpuBindings`, not a vendor-specific class"). ROCm has no
   accelerated path for any of these today (`RocmMatVec` has zero `sgemm` overrides at any batch
   size) — this tier does not change that; the GPU-resident op path activates only when the CUDA
   backend is selected, and the existing scalar CPU path remains the fallback everywhere else,
   matching ROADMAP §6 (documented fallback, not a silent no-op).
4. Scope to the architectures Tier 13C's GPU-resident attention already covers first: Llama-family,
   Mistral, Qwen2. Phi-3/Qwen3/Qwen3-MoE stay on the scalar path pending their own `--gpu-attention`
   follow-up (named in Tiers 9/12/13C already) — do not silently imply they are covered.
5. Correctness: new parity tests comparing GPU-op output against the existing scalar CPU
   implementation within FP tolerance, following the shape of the existing Q4 MMQ parity tests
   (`CudaSgemmBatchedPrefillParityTest`, `Q4KDequantParityTest`).
6. Checkpoint: re-run the Nsight/`nsys` pass and re-publish a bake-off after Phase A alone. Report
   the actual Phi-3.5 and TinyLlama ratio movement honestly — do not assume Phase B is needed before
   measuring whether Phase A closes enough of the gap by itself.

### Phase B — launch fusion / CUDA graph capture-replay

1. Only start once Phase A's ops exist for at least one architecture and step 6's checkpoint shows
   the gap is not yet closed.
2. Capture a decode layer's (or a full forward pass's) fixed launch sequence via the CUDA graph API
   (`cudaGraphCreate`/`cudaGraphLaunch` equivalents through Panama FFI) and replay it per decode
   token instead of re-issuing each kernel launch plus its `Arena.ofConfined()` marshalling
   individually.
3. CUDA graphs are a CUDA-only API with no ROCm equivalent scoped here — this is a **genuinely
   CUDA-only feature** per the exception CLAUDE.md already carves out for the packed-Q4 GEMV path.
   Document the ROCm gap as a named follow-up rather than attempting a HIP-graph port in this tier.
4. Handle shape changes between decode steps (varying batch under `--parallel`, KV length growth)
   by re-capturing rather than reusing a stale graph; measure and document re-capture cost — if it is
   comparable to the per-token launch overhead it replaces, this lever does not pay off for
   single-token decode and that must be reported honestly, not hidden behind an average.

### Non-goals

- Restarting tile/`mul_mat_vec`-class kernel tuning on `q4k_gemv`/`q5k_gemv`/`q6k_gemv` (downgraded
  lever per the Nsight finding — do not reopen without new evidence).
- FlashAttn-class kernels (P5, separate go/no-go gated on Tier 8 baselines).
- A ROCm/HIP-graph equivalent of Phase B.
- Any change to `--mmq`/`--gpu-attention`/`--gpu-layers` default values, CLI surface, or semantics —
  this tier operates entirely inside the already-wired GPU-resident code path those flags select.
- Extending GPU-resident ops to Phi-3/Qwen3/Qwen3-MoE (separate named follow-up, not this tier's
  exit gate).

## Chosen design

Reuse Tier 13's established shape: new PTX kernel(s) + new Java binder class(es), dispatched through
`GpuBindings`/`CudaMatVec` alongside the existing GEMV path, gated behind the CUDA backend being
selected (no new CLI flag — this activates automatically whenever `--mmq`/`--gpu-attention`/
`--gpu-layers` already put the model on the CUDA-resident path for a covered architecture, the same
way `sgemvSameX`'s QKV fusion does today). Land Phase A as four independently-gated, individually
parity-tested increments rather than one large change, so a partial win (e.g. `rmsNorm` alone) is
still measurable and publishable if later increments run out of session time.

## New/modified classes

- **New**: GPU-resident elementwise kernel classes (one per op or a single grouping class — leave the
  exact boundary to the implementer, but prefer new classes over extending `CudaMatVec`, which is a
  GEMV/GEMM dispatch class with a different responsibility). Suggested names: `RmsNormKernel`,
  `RopeKernel`, `ResidualAddKernel`, `SwiGluKernel`.
- **New** (Phase B only): a CUDA graph capture/replay session class (e.g. `CudaGraphSession`)
  owning the capture lifecycle and re-capture-on-shape-change logic.
- **Modified**: `GpuBindings` (new dispatch entries), `LlamaTransformerHandler`/
  `Phi3TransformerHandler`/`Qwen3TransformerHandler` (call sites — Phi-3/Qwen3 keep calling the
  scalar path per the architecture-scoping note above), JFR instrumentation for the four ops
  (new event classes or spans on the existing `ForwardPass`/`MatVec` JFR infrastructure).

## Cross-feature smoke (before feature complete)

- GPU-resident op output matches scalar CPU output within FP tolerance for TinyLlama/Qwen2.5-3B/
  Mistral-7B (covered architectures); Phi-3.5/Qwen3 continue producing scalar-path output unchanged
  (regression check, not a new capability for them).
- JFR spans for the four ops fire and show GPU-backend dispatch on covered architectures, CPU-backend
  dispatch (unchanged) elsewhere — proof the path actually ran, not just that the process exited 0.
- `compare-lora.sh` run (this tier touches the forward pass) — gate: train/ms-per-pass ≤1.25x,
  playback tps ≥0.80x vs last published baseline or `release-0.1.2`.
- `compare-vision.sh` only if the CLIP/Phi-2 vision path shares any of the four touched ops (check
  `ForwardPassHandler`/`VisionAwareForwardPassHandler` call sites first — Tier 17's GEMM work found
  vision's batched prefill uses its own CPU weight-stationary kernels and never shared the touched
  code; if the same holds here, record that as the reasoning for skipping the vision compare rather
  than silently omitting it, per ROADMAP §2 item 3's own precedent).

## Exit checklist (compatibility)

- [ ] Interaction matrix section present, listing which architectures get GPU-resident ops (Llama-
      family/Mistral/Qwen2) and which stay on scalar path (Phi-3/Qwen3/Qwen3-MoE, ROCm) — mirroring
      Tier 13C's own matrix shape, not left blank.
- [ ] No silent no-op: architectures without the new path visibly fall back to the existing scalar
      code, not a broken or skipped call.
- [ ] `docs/performance.md`, `docs/perf-compare/README.md`, and this tier's ROADMAP catalog row
      updated with the honest before/after numbers, whichever way they land.

## Verification and exit gate

Global rules (`PLAN-Infra-ROADMAP.md` → Execution rules): one Infra tier in flight; publish a
`docs/perf-compare/` bake-off (inference + LoRA; vision if applicable per the smoke section above)
before marking feature complete.

**Feature complete** when:

1. Phase A: all four ops have GPU kernels for the covered architectures, parity tests green, JFR
   spans present and firing on GPU for a live TinyLlama/Mistral run.
2. A fresh Nsight/`nsys` idle-time measurement and a published `compare-llama-cpp.sh --gpu` bake-off
   exist for the post-Phase-A state, reporting the actual Phi-3.5 and TinyLlama tg ratio movement.
3. A decision is recorded, with numbers, on whether Phase B was needed and (if attempted) its
   measured effect, including the re-capture-cost caveat if graphs did not pay off for single-token
   decode.
4. `compare-lora.sh` gate ok; vision compare run or explicitly reasoned as not applicable.

**Gate met** (P0 program gate, separate status per ROADMAP's feature-vs-gate vocabulary) requires
Phi-3.5 GPU decode tg ≥0.5x llama.cpp on the published bake-off. Per `PROMPT-P0-Gate.md`'s own
standing instruction: **if the gate still fails after this tier's honest engineering effort, report
fail with numbers — do not amend the ROADMAP gate downward** unless the user explicitly asks. Feature
complete does not require gate met; both statuses must be recorded distinctly in the ROADMAP catalog
row, as every other tier in this set already does.

## Implementation todos

1. JFR-instrument `rmsNorm`/RoPE/residual-add/SwiGLU on the current scalar CPU path (baseline
   measurement, no behavior change).
2. `RmsNormKernel` (PTX + binder) for Llama-family/Mistral/Qwen2; parity test; JFR span; wire into
   `LlamaTransformerHandler`.
3. `RopeKernel`, `ResidualAddKernel`, `SwiGluKernel` in the same pattern.
4. Re-run Nsight/`nsys` + publish `compare-llama-cpp.sh --gpu` bake-off; record the Phase-A-alone
   ratio movement in `docs/performance.md` and the ROADMAP catalog row.
5. If still short of 0.5x: implement Phase B (`CudaGraphSession` capture/replay for the decode hot
   loop); measure and document re-capture cost; re-publish bake-off.
6. `compare-lora.sh`; vision compare or reasoned skip; update `docs/perf-compare/README.md`,
   `docs/performance.md`, ROADMAP status (feature complete and gate status, recorded separately).

## Preview files (expected)

Modified: `CudaMatVec` or a new elementwise binder module, `GpuBindings`,
`LlamaTransformerHandler`/`Phi3TransformerHandler`/`Qwen3TransformerHandler` (call sites only),
`docs/performance.md`, `docs/perf-compare/README.md`, `docs/infra-plan/PLAN-Infra-ROADMAP.md`.

New: elementwise PTX kernel sources + Java binder classes, `CudaGraphSession` (Phase B only), parity
tests, `docs/perf-compare/<timestamp>/` (+ `-lora/`) bake-off artifacts.
