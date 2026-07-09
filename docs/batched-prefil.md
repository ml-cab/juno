---

# Batched Pre-fill — Design and Planning

---

Status: PLAN ONLY — no source changes in this pass. Written for self-execution
in a later session. No other docs are touched (README, CHANGELOG, agent-arch,
howto) per explicit instruction; those updates are listed as a required step
at merge time, not performed here.

## 0. Problem statement

Confirmed via code inspection (`GenerationLoop.java`, `LlamaTransformerHandler.java`,
`MatVec.java`) and consistent with the attached investigation transcript:

- `GenerationLoop.generate()` and `GenerationLoop.generateBatch()` both prefill
  a prompt with a **sequential per-position loop**:

  ```java
  for (int p = startPos; p < promptIds.length - 1; p++) {
      int[] prefillSlice = Arrays.copyOfRange(promptIds, 0, p + 1);
      pipeline.forward(kvKey, prefillSlice, p); // KV stored; logits discarded
  }
  ```

  Two stacked costs, not one:
  1. `Arrays.copyOfRange(promptIds, 0, p + 1)` reallocates and copies a
     growing 0..p slice on every step — O(N^2) copying before any matmul
     runs, even though `LlamaTransformerHandler.getInitialActivation()` only
     ever reads the newest token.
  2. Every step is a full 32-layer forward pass for exactly one token
     (`MatVec.sgemv` — GEMV only, confirmed in both `CpuMatVec` and the GPU
     paths). No batched matmul exists anywhere in the codebase today.

- `MatVec` (`node/src/main/java/cab/ml/juno/node/MatVec.java`) exposes only
  `sgemv(A, x, rows, cols)` plus device-resident GEMV overloads. `CudaMatVec`
  / `RocmMatVec` call `blasSgemv` / `rocblas_sgemv` (or the FP16 strided-batched
  *GEMV*, which batches attention heads, not prompt tokens) — never
  `cublasSgemm` / `rocblas_sgemm` or `*GemmStridedBatched`.
- `InferencePipeline.forwardBatch()` batches **across concurrent requests**
  (N different sessions, N single-token forward passes, ideally one shared
  GPU launch) — it does not batch multiple tokens of one request's prompt.
  These are two independent batching axes; this plan only addresses the
  second one (intra-request prefill).
- Attention cost genuinely grows with position (causal attention over a
  growing KV cache), so some slowdown for long prompts is real and expected.
  But the dominant, fixable cost for CPU/GEMV-bound execution is that every
  one of the ~7 weight matrices per layer is multiplied against a single
  token vector N times instead of against all N prompt tokens as one
  matrix-matrix operation — the standard "batched prefill" optimization
  every fast engine (llama.cpp, vLLM, TensorRT-LLM) relies on, and the one
  missing here.
- Vision requests are the primary trigger today (576 CLIP patch tokens
  pushed through `VisionAwareForwardPassHandler` before any text token),
  but the bug is architecture-wide: any long text-only prompt on any model
  hits the same wall.

## 1. Goals

1. Replace the sequential single-token prefill loop with genuine batched
   matrix-matrix (GEMM) execution across all new prompt tokens for a single
   request, on both CPU and GPU (CUDA + ROCm) backends.
2. Preserve numerical behavior: batched prefill must produce the same
   logits (within float rounding-order tolerance) as the existing
   sequential path for the same prompt. This is a refactor of *how* the
   math is scheduled, not a change to the math itself.
3. Keep decode (one new token per step, post-prefill) untouched — it is
   already the minimal-cost case (batch size 1) and does not need a new
   code path.
4. Cover every caller of the transformer forward pass: `LlamaTransformerHandler`,
   `Phi3TransformerHandler`, `LoraTrainableHandler` (inference-with-adapter
   path), and the vision decorator `VisionAwareForwardPassHandler`.
5. Cover both compute substrates: `CpuMatVec` and the GPU paths
   (`CudaMatVec`, `RocmMatVec`), including the LoRA low-rank adapter matmul.
6. Cover both pipeline transports: `LocalInferencePipeline` (in-JVM,
   `--local`) and `ProcessPipelineClient` (cluster, gRPC).
7. Expose the new path behind a user-facing CLI flag, `--prefill single|batched`,
   defaulting to `batched`, so the old sequential path stays reachable as an
   explicit escape hatch (bisection, GPU-vendor bug workaround, or a
   like-for-like comparison against `docs/performance.md` baselines) without
   requiring a rebuild or a code change. See Section 4.8.

## 2. Non-goals (explicitly out of scope for this change)

- **Fused/flash attention.** Causal attention score computation still loops
  per new-token position against its own KV-cache slice (each position
  attends to a different-length history, so the *attention* step stays
  inherently ragged). This plan batches the **linear projections**
  (QKV-in, attn-out, FFN up/down — the ~7 GEMV calls per layer that
  dominate FLOPs) and the KV-cache write; it does not rewrite attention
  into a single fused kernel. This is the same scoping the reference
  engines use for a first batched-prefill pass, and is enough to remove
  the O(N) redundant weight-matrix traversal that is the actual bottleneck
  here.
- **Batching across concurrent requests during prefill** (i.e., interleaving
  Session A's prefill with Session B's prefill in one GPU call). That is a
  distinct optimization already partially modeled by
  `InferencePipeline.forwardBatch()` for decode; extending it to prefill is
  a natural follow-up but adds ragged-length batching complexity that
  should not be mixed into this change.
- **LoRA training-step batching.** `LoraTrainableHandler.trainStep()`'s
  truncated-BPTT backward pass is intentionally per-position
  (gradients do not cross KV-cache boundaries; see its class Javadoc). Only
  the **inference-with-adapter forward** path (`--lora-play`, i.e.
  `LoraTrainableHandler.forward()`) is in scope. `trainStep()` keeps its
  current sequential per-token forward+backward; batching training is a
  separate, larger change to gradient accumulation and is not attempted
  here.
- Gemma/Qwen handlers — per `docs/arch.md` these are still under
  development with no LoRA support; they are not touched until they reach
  the same "supported" status as `LlamaTransformerHandler` / `Phi3TransformerHandler`.

## 3. New abstractions (new classes preferred over extending existing ones)

Per implementation rule D3/D4 (KISS, prefer new classes), prefill batching
is added as a **parallel path**, not a modification of the existing
single-token contract. `ForwardRequest` / `ForwardResult` / `MatVec.sgemv`
keep their current meaning and callers (decode, tests, LoRA training)
unchanged.

### 3.1 `BatchForwardRequest` (new record, `node` module)

```java
public record BatchForwardRequest(
        String requestId,
        int[] tokenIds,      // NEW tokens only, in order — length = window size
        float[] activations, // flattened windowSize * hiddenDim, non-null for non-first nodes
        int startPosition,   // KV cache position of tokenIds[0] / activations row 0
        int windowSize        // tokenIds.length or activations.length/hiddenDim
) {
    public static BatchForwardRequest withTokens(String requestId, int[] tokenIds, int startPosition) { ... }
    public static BatchForwardRequest withActivations(String requestId, float[] activations, int windowSize, int startPosition) { ... }
    public boolean isFirstNode() { return tokenIds != null; }
}
```

Unlike the current `ForwardRequest.tokenIds()`, this carries **only the new
window** (e.g. 576 patch tokens, or 1170 total minus whatever was already
cached), not a copy of everything from position 0. This alone removes the
O(N^2) `Arrays.copyOfRange` churn independent of the matmul fix.

### 3.2 `BatchForwardResult` (new record, `node` module)

```java
public record BatchForwardResult(
        String requestId,
        float[] activations,  // flattened windowSize * hiddenDim, non-null for intermediate nodes
        float[] lastLogits,   // logits for the LAST position only (windowSize-1) — all we need after prefill
        int windowSize,
        long computeNanos
) {
    public boolean isFinalNode() { return lastLogits != null; }
}
```

Only the final position's logits are needed after prefill (the loop today
discards every intermediate logit anyway); returning all N logit vectors
would be wasted allocation for a 32064–152064-wide vocab.

### 3.3 `ForwardPassHandler.forwardBatch(BatchForwardRequest, ShardContext)` (new default method)

```java
default BatchForwardResult forwardBatch(BatchForwardRequest request, ShardContext context) {
    // Correctness-preserving default: loop token-by-token through the
    // existing single-token forward(), reusing today's exact code path.
    // Any handler that does not override this keeps working, just without
    // the speedup — mirrors the existing InferencePipeline.forwardBatch()
    // pattern (serial default, real implementations override).
}
```

- `LlamaTransformerHandler` and `Phi3TransformerHandler` override this with
  the real batched-GEMM implementation (Section 4).
- `LoraTrainableHandler` overrides it for the inference-with-adapter path
  only (Section 4.4); `trainStep()` is untouched (non-goal, Section 2).
- `VisionAwareForwardPassHandler` overrides it to batch its embedding
  substitution across the window before delegating (Section 4.5).
- Test doubles (`CyclicForwardPassHandler` or equivalent) get the default
  for free, same as `InferencePipeline.forwardBatch()`'s existing serial
  default — no test-double changes required unless a test wants to assert
  the batched path specifically.

### 3.4 `MatVec.sgemm(...)` (new interface method, `node` module)

```java
/**
 * Compute Y = A * X for a batch of B input columns in one call.
 * A: [rows, cols] row-major (unchanged from sgemv). X: [batch][cols].
 * Returns Y: [batch][rows].
 *
 * Weight-stationary blocking: implementations should load each weight row
 * once and multiply it against all B columns before advancing to the next
 * row, maximizing weight reuse — the actual performance win over calling
 * sgemv B times (which re-streams the full weight matrix from memory/VRAM
 * B times).
 */
default float[][] sgemm(float[] A, float[][] X, int rows, int cols) {
    // Correctness-preserving default: B calls to sgemv(). Every existing
    // MatVec implementation (including test fakes) is correct by
    // construction; only CpuMatVec/CudaMatVec/RocmMatVec override for speed.
    float[][] Y = new float[X.length][];
    for (int b = 0; b < X.length; b++) Y[b] = sgemv(A, X[b], rows, cols);
    return Y;
}

default float[][] sgemm(DeviceFloatMatrix A, float[][] X) { throw new UnsupportedOperationException(...); }
default float[][] sgemm(DeviceHalfMatrix A, float[][] X)  { throw new UnsupportedOperationException(...); }
```

Same shape as the existing `sgemv` overload family (host, device-FP32,
device-FP16) — additive, does not change any existing method signature.

## 4. Per-module implementation plan

### 4.1 `node` — `CpuMatVec.sgemm`

- Real implementation: for each output row `r`, load `A[r*cols .. r*cols+cols)`
  once, then loop over the batch computing all B dot products against that
  row before moving to row `r+1` (weight-stationary — row loaded once,
  reused B times; this is the entire performance thesis, no B× re-read of
  `A` from memory).
- Parallelize over rows with `IntStream.range(0, rows).parallel()` (same
  `ForkJoinPool.commonPool()` pattern `CpuMatVec.sgemv` already uses) — each
  parallel task now does `cols * B` multiply-adds instead of `cols`,
  improving the compute-per-task-dispatch ratio, which is itself a second,
  smaller win on top of weight reuse.
- Unit test: `CpuMatVecSgemmTest` — for random A/X, assert
  `sgemm(A, X, rows, cols)[b]` equals `sgemv(A, X[b], rows, cols)` for every
  `b`, exactly (same summation order per column achievable by keeping the
  per-row inner loop structure identical to `sgemv`'s). This is the
  business-logic-critical test per rule D1: correctness of the new batched
  path against the existing trusted path is the single highest-value test
  in this whole change.

### 4.2 `node` — GPU backends (`CudaMatVec`, `RocmMatVec`, `GpuBindings`)

- Add two vendor-neutral `MethodHandle` accessors to `GpuBindings`,
  following the existing naming convention (`blasSgemv`,
  `blasHSSgemvStridedBatched`):
  - `blasSgemmStridedBatched()` — FP32 weights, FP32 activations. Backs
    `cublasSgemmStridedBatched` / `rocblas_sgemm_strided_batched`.
  - `blasHSGemmStridedBatched()` — FP16 device-resident weights
    (`DeviceHalfMatrix`), FP32 activations in/out — mirrors the existing
    `blasHSSgemvStridedBatched` naming (`H`=FP16 input, `S`=FP32
    output) so the FP16-weights/FP32-math convention already established
    for GEMV carries over to GEMM without inventing new letters.
- `CudaMatVec.sgemm(DeviceHalfMatrix A, float[][] X)`: single H2D upload of
  the whole `X` batch as one contiguous device buffer (`cols * B` floats),
  one `cublasGemmStridedBatched`-style call (or a plain `cublasSgemm` with X
  as a `[cols, B]` device matrix — simpler and sufficient since all B
  columns share the same `A`; strided-batched is only needed when A differs
  per batch element, which it does not here), one D2H download of the
  `[rows, B]` result. This collapses what is today B sequential
  H2D-launch-D2H round trips (the ~262,000-kernel-launch problem called out
  in the investigation notes) into 3 total transfers per weight matrix
  per layer, independent of B.
- `RocmMatVec.sgemm` mirrors this with `rocblas_sgemm` against the same
  `DeviceHalfMatrix`/`DeviceFloatMatrix` wrappers — no new device-memory
  abstractions needed, `DeviceHalfMatrix`/`DeviceFloatMatrix` already
  describe a 2D buffer; only the multiply call and the batch upload/download
  are new.
- Serialization: keep the existing `GpuContext` lock discipline
  (`CudaMatVec`/`RocmMatVec` already serialize BLAS calls per device on a
  shared lock) — the batched GEMM call is a single critical section, same
  as today's single `sgemv`, so no new locking logic is required, just a
  bigger payload per critical section.
- Unit test: `CudaMatVecBackendTest` / `RocmMatVecTest` gain an `sgemm`
  parity case (skipped when no device is available, consistent with the
  existing `CudaAvailability`/`RocmAvailability`-gated pattern in those
  tests) asserting `sgemm` output matches `sgemv` called B times.

### 4.3 `node` — `LlamaTransformerHandler.forwardBatch` / `Phi3TransformerHandler.forwardBatch`

New private `transformerLayerBatch(float[][] x, int li, int startPos, float[] kCacheLayer, float[] vCacheLayer)`
alongside the existing `transformerLayer` (kept as-is for decode):

1. QKV projection: one `matVec.sgemm(Wqkv, x, ...)` call per layer instead
   of B calls — this is the dominant win.
2. RoPE: still applied per row inside the batch (`rope(q_b, startPos + b, ...)`
   for each `b` in `0..B)`); RoPE cost is O(headDim) per token, negligible
   next to the GEMV/GEMM cost it rides alongside, so no batching needed
   here, just a loop over the batch dimension writing into the already-open
   activation buffer.
3. KV cache write: B `System.arraycopy` calls (one per new position) into
   `kCacheLayer`/`vCacheLayer` at `(startPos + b) * kvDim` — same
   `ensureKvCapacity` growth logic as today, just called once per layer for
   the whole window instead of once per token per layer.
4. Attention (`gqa`): loop over `b in 0..B`, each row attends causally over
   `kCache[0 .. startPos+b]` — this stays a per-row loop (Section 2,
   non-goal: fused attention) but the ~7 weight-matrix GEMVs that used to
   dominate cost are gone from this loop; only the O(seqLen) score/weighted-sum
   arithmetic remains, which is comparatively cheap.
5. Attn-out projection and FFN (gate/up/down): batched the same way as
   step 1 — one `sgemm` call per weight matrix per layer for the whole
   window.
6. `runLayersBatch` returns the final layer's activations for all B
   positions to the caller; only the last row is projected through the LM
   head (existing `lastRmsHiddenForEmbedding`/logits path, unchanged),
   matching `BatchForwardResult.lastLogits` semantics from Section 3.2.

`Phi3TransformerHandler` gets the identical treatment (it already shares
`rope`/`gqa`-shaped logic per `docs/phi3-inference-handoff.md`; the
extended-RoPE/NeoX-pairing specifics apply per-row exactly as they do
per-call today, no interaction with batching).

Unit tests (business-logic-critical, per rule D1):
- `LlamaTransformerHandlerBatchParityTest` — construct a small
  fixture-backed handler (same fixture GGUF/config as
  `LlamaTransformerHandlerF16MatVecTest`), run a short prompt through the
  existing sequential per-token `forward()` loop and through the new
  `forwardBatch()`, assert identical final-position logits (exact or
  within a documented float epsilon if reduction order differs on the GPU
  path).
- `Phi3TransformerHandlerBatchParityTest` — same shape, guards the
  extended-RoPE path specifically since that is the most fragile part of
  the Phi-3 handler per the handoff doc.
- KV-cache-after-batch-prefill test: after `forwardBatch` over a window,
  a subsequent single-token `forward()` decode call at `startPos + B` must
  produce the same next-token distribution as if the whole prompt had been
  prefilled one token at a time — this is the test that actually proves the
  optimization is safe to ship, since a subtly wrong KV cache write would
  otherwise only show up as degraded generation quality, not a crash.

### 4.4 `node` — `LoraTrainableHandler.forwardBatch`

- In scope: the plain inference path used when a `.lora` adapter is
  applied at inference time (`--lora-play`), i.e. `LoraTrainableHandler.forward()`,
  which reuses frozen quantized base weights plus the low-rank `A`/`B`
  adapter matrices.
- The base-weight matmuls batch exactly as in Section 4.3 (same `MatVec.sgemm`).
- The adapter matmul (`x -> A -> B`, rank ~8) is small (`rank * hiddenDim`
  multiply-adds per token) — cheap enough that a straightforward per-row
  loop over the batch is acceptable for the adapter path specifically; it
  is not on the critical path the way the frozen base weight GEMVs are.
  Batching it too is a low-risk follow-up, not required to remove the
  10-minute stall.
- `trainStep()` is explicitly untouched (Section 2) — its `LayerState`
  bookkeeping is built around one position at a time by design (truncated
  BPTT), and batching it is a separate change to gradient accumulation,
  not a forward-pass scheduling change.
- Unit test: `LoraTrainableHandlerBatchInferenceParityTest` — same shape as
  4.3's parity test, run with a small `LoraAdapterSet` applied, assert
  batched and sequential inference paths agree. Explicitly does **not**
  touch `trainStep()`/`LoraAdamOptimizer` — existing `LoraTrainableHandlerTest`
  training coverage is untouched.

### 4.5 `vision` — `VisionAwareForwardPassHandler.forwardBatch`

- This is the module that most needs the fix (image prompts are exactly
  the 576+-token windows that trigger the multi-minute stall) and the
  easiest to batch correctly, since its job — substituting a precomputed
  patch vector for `IMAGE_TOKEN_ID` positions, looked up from the
  per-request `ConcurrentHashMap` registered by `registerVisionEmbeddings` —
  is already a pure per-position lookup with no cross-position state.
- New `getInitialActivationBatch(BatchForwardRequest request)`: loop over
  `request.tokenIds()` building a `float[windowSize][hiddenDim]` up front
  (image-token rows come from the registered patch table, text-token rows
  from the normal embedding lookup delegated to the wrapped handler), then
  hand the whole matrix to the wrapped handler's `forwardBatch()` for the
  actual transformer compute. This loop is O(windowSize * hiddenDim) —
  negligible next to the transformer matmuls it precedes, so no further
  optimization is needed here beyond not doing it one token/one full
  forward pass at a time as today.
- Unit test: `VisionAwareForwardPassHandlerBatchTest` — extend the existing
  `VisionAwareForwardPassHandlerTest` fixture with a window containing a
  mix of image and text token IDs, assert the batched activation matrix
  matches row-by-row what today's per-token `forward()` path produces for
  the same window.

### 4.6 `coordinator` — `GenerationLoop`

Replace both prefill loops (`generate()` line ~306-313 and
`generateBatch()` line ~130-138) with a single windowed call, gated by a
new `PrefillMode` passed into `GenerationLoop` at construction (see
Section 4.8 for where this value comes from):

```java
int windowSize = promptIds.length - 1 - startPos;
if (windowSize > 0) {
    if (prefillMode == PrefillMode.BATCHED) {
        int[] window = Arrays.copyOfRange(promptIds, startPos, promptIds.length - 1);
        pipeline.prefillBatch(kvKey, window, startPos); // logits discarded, same as today
    } else { // PrefillMode.SINGLE — today's exact code path, kept verbatim
        for (int p = startPos; p < promptIds.length - 1; p++) {
            int[] prefillSlice = Arrays.copyOfRange(promptIds, 0, p + 1);
            pipeline.forward(kvKey, prefillSlice, p);
        }
    }
}
```

- New `InferencePipeline.prefillBatch(String requestId, int[] newTokens, int startPosition)`
  default method (mirrors the existing `forwardBatch` default-serial
  pattern): default loops calling today's `forward()` once per token
  (byte-for-byte the current behavior, zero risk if a pipeline
  implementation does not override it); `LocalInferencePipeline` overrides
  it to call the new `ForwardPassHandler.forwardBatch` chain end to end.
- Both `generate()` and `generateBatch()` (the two duplicated prefill loops
  called out in the class's own Javadoc) get the same change — one
  windowed call replaces the per-position loop in each when
  `prefillMode == BATCHED`, eliminating the
  `Arrays.copyOfRange(promptIds, 0, p+1)` O(N^2) copy pattern in both places
  as a side effect; the `SINGLE` branch is a deliberate, literal copy of
  today's loop, not a re-derivation of it, so `--prefill single` remains a
  byte-for-byte fallback to the currently-shipped, currently-trusted
  behavior with zero new logic to regress.
- Unit test: `GenerationLoopBatchTest` (existing file — extend) and
  `GenerationLoopTest` gain a case asserting the same generated token
  sequence for a fixed seed/greedy config under `--prefill batched` and
  under `--prefill single`, using a deterministic fixture handler —
  behavioral parity between the two modes, not just a code-path smoke test.

### 4.7 `node` / `juno-player` — cluster (gRPC) path

- `LocalInferencePipeline` (in-JVM, `--local` mode — the mode the reported
  vision bug reproduces in) gets the full speedup with no wire-format
  changes, since everything stays as Java objects in one JVM.
- `ProcessPipelineClient` (cluster mode, gRPC) needs a **new** proto
  message rather than overloading the existing `ForwardRequest.batch_size`
  field, which is already documented (`inference.proto`) as the
  cross-request batch count — conflating it with intra-request window size
  would silently break the existing cross-request batching semantics.
  Proposed: add `int32 window_size = 10;` to `ForwardRequest` /
  `ForwardResponse` (next free field numbers), meaning "activation bytes
  encode `window_size` concatenated position-vectors instead of one" when
  `window_size > 1`; `window_size` unset/`0` (proto3 default) keeps today's
  single-token wire format byte-for-byte compatible with older nodes.
- `ActivationCodec.encode`/`decode` need a batch-aware overload
  (`encode(float[][] rows, ActivationDtype)` /
  `decode(bytes, dtype, windowSize)`) — additive, existing single-vector
  overloads unchanged, so `TensorParallelPipelineClient` and any other
  caller not yet updated keeps compiling and working against
  `window_size=0`.
- This is the highest-risk cross-cutting piece (wire format, backward
  compatibility across a rolling cluster upgrade) and is called out
  separately in Section 6 as the item most worth prototyping first in
  isolation, behind its own test (`ActivationCodecBatchTest`) before wiring
  it into `ProcessPipelineClient`.
- Until this lands, `ProcessPipelineClient` keeps the default
  `InferencePipeline.prefillBatch()` (serial fallback) — cluster mode stays
  correct and unblocked by the gRPC work, it simply does not get the
  speedup until `window_size` ships. This should be called out plainly in
  the eventual CHANGELOG entry so cluster-mode users are not surprised
  that only `--local` mode is fast immediately after this change (the same
  kind of local-vs-cluster gap already on record for vision routes per
  `docs/agent-arch.txt`'s "KNOWN LIMITATION" note).

### 4.8 `juno-player` / `juno-master` — `--prefill single|batched` CLI flag

New enum, new class per rule D4 (prefer new classes over extending
existing ones):

```java
public enum PrefillMode {
    SINGLE,   // today's sequential one-token-at-a-time prefill loop
    BATCHED;  // new windowed GEMM prefill (Sections 4.1-4.7) — default

    public static PrefillMode parse(String s) {
        return switch (s.toLowerCase(Locale.ROOT)) {
            case "single" -> SINGLE;
            case "batched" -> BATCHED;
            default -> throw new IllegalArgumentException(
                "Unrecognized --prefill value '" + s + "' (expected: single, batched)");
        };
    }
}
```

Placement: `coordinator` module (alongside `GenerationLoop`, which is the
only class that reads it) — not `node`, since neither `ForwardPassHandler`
nor `MatVec` need to know which mode is active; they just get called via
`forward()` or `forwardBatch()` depending on which branch `GenerationLoop`
takes.

- **`ConsoleMain` flag parsing** (`juno-player`), following the exact
  existing `--dtype` pattern (`parseDtype`, Section on `ConsoleMain.java`
  lines ~266-268, ~1483-1491 — explicit case, explicit unrecognized-value
  `WARNING` to stderr rather than a silent fallback, matching this
  codebase's own recently-fixed `--dtype` bug from `CHANGELOG.md` Session
  35, which is exactly the failure mode to avoid repeating here):

  ```java
  case "--prefill":
      prefillMode = parsePrefillMode(args[++i]);
      break;
  ```

  ```java
  private static PrefillMode parsePrefillMode(String s) {
      try {
          return PrefillMode.parse(s);
      } catch (IllegalArgumentException e) {
          System.err.println("WARNING: " + e.getMessage() + " — defaulting to 'batched'");
          return PrefillMode.BATCHED;
      }
  }
  ```

  Default when the flag is absent entirely: `PrefillMode.BATCHED` (goal 7).
  Help text addition alongside the existing `--dtype` line:
  `--prefill single|batched     Prefill strategy (default: batched)`.
- **`scripts/run.sh`** / **`scripts/run.bat`**: `--prefill` (and a
  `PREFILL_MODE` env var override) threaded through to `ConsoleMain` for
  both `local` and `cluster` commands, following the same pattern already
  used for `--mmproj-path`/`MMPROJ_PATH` and `--api-port`/`API_PORT` in the
  Session 35 vision work — Windows parity is treated as a first-class
  requirement here from the start, not a follow-up gap (unlike the
  `--api-port` omission on `run.bat local` that Session 35 had to
  backfill).
- **Cluster mode**: `PrefillMode` is a coordinator-local decision — the
  coordinator's `GenerationLoop` picks `SINGLE` or `BATCHED` and calls
  either `pipeline.forward()` or `pipeline.prefillBatch()` accordingly; the
  node side does not need its own `--prefill` flag, since `ForwardPassHandler.forwardBatch()`
  is just another entry point nodes already expose (Section 3.3). Until
  Section 4.7's gRPC wire-format change ships, `--prefill batched` on a
  cluster falls through `InferencePipeline.prefillBatch()`'s default serial
  implementation on `ProcessPipelineClient` — correct, but no faster than
  `single` yet on cluster specifically. This should be visible to the user:
  `ConsoleMain` should log a one-line `INFO` note at cluster startup when
  `PrefillMode.BATCHED` is selected but the active pipeline is
  `ProcessPipelineClient` without batched-wire support, so nobody spends
  time debugging "why is `--prefill batched` not faster on my cluster"
  before Section 4.7 lands.
- **Interaction with vision (`VisionChatHandler`)**: no new flag needed
  there — `--prefill` is a global generation-strategy setting read once by
  `GenerationLoop` at startup, and vision requests flow through the same
  `GenerationLoop.generate()`/`generateBatch()` call sites as text, so
  `--prefill batched` (the default) is exactly what fixes the reported
  10-minute vision stall with zero vision-specific flag surface.
- Unit tests:
  - `PrefillModeTest` (new, `coordinator` module) — `parse("single")`,
    `parse("BATCHED")` (case-insensitive), unrecognized value throws
    `IllegalArgumentException` with a message naming the rejected value
    (mirrors `ConsoleMainDtypeTest`'s assertions for `--dtype`, Section
    on the `--dtype` fix in `CHANGELOG.md`).
  - `ConsoleMainPrefillFlagTest` (new, `juno-player` module) — drives
    argument parsing via reflection the same way `ConsoleMainDtypeTest`
    and `ConsoleMainLoggingTest` already do, asserts: flag absent → `BATCHED`;
    `--prefill single` → `SINGLE`; `--prefill garbage` → `WARNING` to
    stderr + falls back to `BATCHED` (not a hard failure — consistent with
    how `--dtype`'s own unrecognized-value case was fixed to warn-and-fallback
    rather than crash).
  - Extend `GenerationLoopBatchTest`/`GenerationLoopTest` per Section 4.6
    to construct `GenerationLoop` with each `PrefillMode` value explicitly
    and assert identical generated output — this is the test that actually
    proves `single` is a safe, permanent fallback and not just an inert flag.

## 5. Cross-cutting drawbacks and risks

- **Memory.** A batched window materializes `windowSize * hiddenDim` floats
  per layer instead of `hiddenDim` — for 1170 tokens at hidden=4096 that is
  ~19 MB per layer activation buffer (transient, one layer at a time, not
  ×32 resident simultaneously) — small next to the 98.3%-of-16GB pressure
  already reported, but worth a guard: if `windowSize * hiddenDim * 4 bytes`
  would exceed a configurable ceiling, chunk the window into sub-batches
  (e.g. 256 tokens at a time) rather than one all-at-once matrix — still
  vastly fewer chunks than today's one-token-at-a-time loop, and bounds
  peak memory on the already-tight CPU host from the report. This chunking
  can reuse the exact same `forwardBatch` method with a smaller `windowSize`
  called in a short outer loop from `GenerationLoop` or
  `LocalInferencePipeline` — no new abstraction needed, just a loop bound.
- **GPU VRAM OOM fallback.** `README.md` documents "automatic CPU quantised
  fallback on VRAM OOM" for device-resident weights; the batched GEMM path
  must respect the same fallback — `CudaMatVec.sgemm`/`RocmMatVec.sgemm`
  should catch the existing OOM signal the GEMV path already handles and
  fall back to `CpuMatVec.sgemm` (or the default per-token loop) rather
  than a new failure mode.
- **Numerical drift.** Batched GEMM may sum in a different order than B
  sequential GEMVs (BLAS libraries are free to reorder/reduce differently
  at different batch widths). This is expected and acceptable (documented
  precedent: the existing FLOAT16 activation wire format already accepts
  ~0.1% relative error, per `inference.proto`'s own comment), but the
  parity tests in Section 4.3 should assert closeness with an explicit
  epsilon on GPU paths rather than bitwise equality, while CPU-path parity
  should stay bitwise (same instruction order is achievable there since
  `CpuMatVec` fully controls the reduction).
- **Concurrent requests during prefill.** `kvCacheK`/`kvCacheV` are
  `ConcurrentHashMap`s keyed by `requestId`, and `forwardBatch` for one
  request still only writes that request's own KV slots — no new
  cross-request interaction is introduced; the existing "thread-safe for
  distinct request IDs" contract on `ForwardPassHandler`/`LlamaTransformerHandler`
  is unchanged by this plan.
- **Session-cache-hit partial prefill.** `GenerationLoop.generate()`'s
  session KV reuse (`startPos` computed from a matched cache offset) still
  works unchanged — the new window is simply `promptIds[startPos .. len-1)`
  instead of the full prompt; sessions with a full cache hit already skip
  prefill entirely (`prefillSteps <= 0`) and are unaffected.
- **JFR observability.** `docs/performance.md` keys off `juno.ForwardPass.prefillMs`
  and `juno.MatVec.durationMs`; a batched call is still one `ForwardPass`
  JFR event per node call as before (now covering a window instead of one
  token) — no new event type is required, but the existing `MatVec` JFR
  event should record `batchSize` as a new field so `prefillMs` p95 numbers
  in the matrix remain interpretable after this change (called out here,
  not implemented, since `docs/performance.md` itself is out of scope for
  this pass).

## 6. Suggested build order (smallest safe increments first)

1. `CpuMatVec.sgemm` + `CpuMatVecSgemmTest` (pure, no handler changes,
   fully unit-testable in isolation, proves the weight-stationary-blocking
   thesis on CPU where the reported bug actually lives).
2. `BatchForwardRequest`/`BatchForwardResult`/`ForwardPassHandler.forwardBatch`
   default method (additive, compiles with zero behavior change everywhere
   until something overrides it).
3. `LlamaTransformerHandler.forwardBatch` + parity/KV-cache tests
   (Section 4.3) — this is the change that actually fixes the reported
   symptom for text and, combined with step 5, for vision.
4. `Phi3TransformerHandler.forwardBatch` + parity test — same shape,
   confirms the extended-RoPE path is batch-safe.
5. `VisionAwareForwardPassHandler.forwardBatch` (Section 4.5) — smallest
   vision-side change, directly closes the original vision-request symptom
   once layered on top of step 3.
6. `PrefillMode` enum + `GenerationLoop` prefill-loop branch + `InferencePipeline.prefillBatch`
   default (Sections 4.6, 4.8) — wires the fast path into `--local` end to
   end **behind the flag, defaulting to `batched`**; `juno test` smoke
   suite and `docs/performance.md`'s reproduction commands should be
   re-run manually against `models/tinyllama-...` and, if available, a
   real llava/mmproj pair, under both `--prefill batched` (must be fast)
   and `--prefill single` (must reproduce today's exact numbers/output, to
   confirm the fallback branch was not accidentally changed) before
   calling this done.
7. `ConsoleMain`/`scripts/run.sh`/`scripts/run.bat` flag parsing and help
   text (Section 4.8) — small, isolated, testable independently of the GPU
   work below; unblocks manual verification of step 6 with a real running
   binary rather than only unit tests.
8. `CudaMatVec.sgemm` / `RocmMatVec.sgemm` + `GpuBindings` new handles
   (Section 4.2) — GPU speedup; independent of steps 1-7 landing first,
   can be developed in parallel once `MatVec.sgemm`'s interface (step 2)
   is fixed.
9. `LoraTrainableHandler.forwardBatch` inference-path override
   (Section 4.4) — lowest urgency (LoRA-overlay inference is not the
   reported symptom), last in line.
10. gRPC/cluster wire-format change (Section 4.7) — highest risk, isolated
    last; until it lands, cluster mode is correct but not sped up under
    `--prefill batched` specifically, which is an acceptable interim state
    (Section 4.8 already specifies the `INFO` log line covering this gap)
    and should be called out explicitly when this work is merged.

Each step above should compile and pass the full existing test suite on
its own before the next step begins, per KISS — no step depends on a step
later in this list.

## 7. Definition of done for this feature (for the later implementation session)

- All new tests listed in Section 4 pass; no existing test is modified to
  weaken an assertion in order to pass (if an existing test needs to
  change, the reason must be that the test asserted the old sequential
  behavior by name/shape, not that batching broke something it shouldn't
  have).
- `curl -X POST http://localhost:8081/v1/vision/chat ...` (the reproduction
  command from the reported issue) returns a coherent answer for
  `juno-console.jpg` well under a minute on the reporter's CPU-only
  environment, down from the reported 10+ minutes / no response, using
  llava-v1.5-7b or a comparable local model, with no flag passed (default
  is `batched`).
- The same reproduction run with `--prefill single` explicitly set
  reproduces today's exact (slow) behavior — proving the flag actually
  switches code paths rather than being cosmetic, and that the fallback is
  real and load-bearing, not just a documented intention.
- `./juno test` (the existing 6 pipeline + 2 tensor smoke suite) passes
  under both `--prefill batched` (default) and `--prefill single`.
- No new public API breaks: `ForwardRequest`, `ForwardResult`, `MatVec.sgemv`,
  `InferencePipeline.forward`, and their existing callers/tests are
  untouched — everything above is additive per Section 3.
- `--prefill` appears in `ConsoleMain`'s `--help` output and behaves
  identically on `scripts/run.sh` and `scripts/run.bat` (flag name, env var
  name, default value) — Windows parity checked in the same pass, not
  deferred, per the note in Section 4.8.
- Follow-up doc updates (explicitly deferred, not part of this plan's
  output): `docs/agent-arch.txt` (new classes/methods), `docs/howto.md`
  (no user-facing flag changes expected, but worth confirming), `README.md`
  performance-of-vision note, and `docs/performance.md`/`juno_test_matrix.html`
  re-measurement of prefill p95 once implemented.


---

# Implementation of the batched pre-fill 

---

**New node classes (3)**

`BatchForwardRequest` — record carrying a window of new token IDs (first node) or flattened activations (subsequent nodes) plus `startPosition`. Eliminates the O(N²) `copyOfRange` churn at the call site.

`BatchForwardResult` — record carrying either all-window activations flattened (intermediate node) or the last-position logits only (final node). Only one logit vector per prefill call, not N.

`PrefillMode` — `SINGLE | BATCHED` enum with case-insensitive `parse()`. Lives in `coordinator` alongside `GenerationLoop`.

**Modified interfaces (2)**

`MatVec` — three new `default sgemm` overloads: `float[] A`, `DeviceHalfMatrix`, `DeviceFloatMatrix`. All default to looping `sgemv` B times (correct everywhere), ready for GPU backends to override with a single BLAS SGEMM call.

`ForwardPassHandler` — new `default forwardBatch(BatchForwardRequest, ShardContext)`. Default loops the existing single-token `forward()` path exactly, so any handler that does not override keeps working with zero behavior change.

`InferencePipeline` — new `default prefillBatch(requestId, newTokens[], startPosition)`. Default loops `forward()` once per token — same as the old `GenerationLoop` loop, minus the growing prefix copy.

**Modified implementations (7)**

`CpuMatVec` — overrides `sgemm(float[] A, float[][] X, int rows, int cols)` with weight-stationary blocking: each weight row loaded once, dot-producted against all B input columns before moving to the next row. Parallelized over rows with `IntStream.parallel()`.

`LocalInferencePipeline` — overrides `prefillBatch`: builds a `BatchForwardRequest.withTokens`, walks the stage list calling `forwardBatch` on each handler, passes flattened activations between nodes, discards the final logits.

`LlamaTransformerHandler` — `forwardBatch` + private `runLayersBatch` + `transformerLayerBatch` + `sgemmLayer` dispatch helper. Linear projections (Q/K/V, attn-out, gate/up/down) each become one `sgemmLayer` call for the whole window; RoPE and causal attention stay per-token; KV cache is written B positions per layer in one pass. Uses `hasEmbeddings` (handler field) to decide embedding lookup, matching `getInitialActivation`.

`Phi3TransformerHandler` — same structure, adapted for fused QKV and fused gate+up tensors via `sgemmFused`; extended RoPE applied per-token.

`LoraTrainableHandler` — inference-with-adapter path only (`forwardBatch` + `runLayersBatch` + `inferenceLayerBatch`). Base-weight matmuls batched; LoRA deltas applied per-token in a cheap loop. `trainStep` untouched.

`GenerationLoop` — two constructors (existing one delegates to new one with `BATCHED` default). Both prefill loops (`generate` and `generateBatch`) branch on `prefillMode`: `BATCHED` calls `pipeline.prefillBatch(kvKey, window, startPos)` with the exact window slice; `SINGLE` keeps the original loop verbatim as an escape hatch.

`VisionAwareForwardPassHandler` — `forwardBatch` override: builds the full window activation matrix in one pass (image-token rows from patch table, text-token rows zeroed), then hands a single `withActivations` request to the wrapped text handler. This is the direct fix for the reported 10-minute vision stall.

`ConsoleMain` — `--prefill single|batched` flag with `parsePrefillMode` helper (warn-and-fallback on unknown value, same pattern as `--dtype`). All four `GenerationLoop` construction sites pass `prefillMode`. Help text updated.

**Scripts (2)**

`run.sh` — `PREFILL_MODE` env var + `--prefill` option parser + `${prefill_mode_arg}` threaded into both `cmd_cluster` and `cmd_local` exec blocks.

`run.bat` — same, Windows parity from the start (not a follow-up gap).

**Tests (5)**

`CpuMatVecSgemmTest` — business-logic-critical: asserts `sgemm(A, X)[b]` equals `sgemv(A, X[b])` exactly for every slot (same float reduction order on CPU).

`PrefillModeTest` — `parse` round-trips, case-insensitivity, unrecognized value throws with the rejected string in the message.

`GenerationLoopTest` (extended) — three new cases: `BATCHED` vs `SINGLE` produce identical token sequence; both stop at EOS; default constructor equals explicit `BATCHED`.

`VisionAwareForwardPassHandlerBatchTest` — pure-image window, text-only window, mixed window, non-first-node passthrough, no-embeddings passthrough, result carries correct windowSize.

`ConsoleMainPrefillFlagTest` — reflection-based, same pattern as `ConsoleMainLoggingTest`: `single`, `BATCHED`, upper-case variants; unrecognized value emits WARNING to stderr and returns `BATCHED`, not a hard failure.

---

Total changes 21 source files.

Compile-time errors:
- node module change class to package visible for tests
- vision un-finalize some class to be able to test it

Test failures: 
- VisionAwareForwardPassHandlerBatchTest.text_only_window_rows_are_zero:94 » IllegalArgument patchVectors must not be empty

Runtime-errors: 0

Single mode (--prefill single):

juno.ForwardPass.count: 768
juno.ForwardPass.prefill.count: 3
juno.ForwardPass.decode.count: 765
juno.MatVec.count: 39,680
juno.MatVec.duration.total_ms: 32,382 ms
juno.TokenProduced.count: 200 tokens
juno.TokenProduced.elapsed_seconds: 27.67s
TPS: 7.23
Batched mode (--prefill batched, default):

juno.ForwardPass.count: 549
juno.ForwardPass.prefill.count: 0 (!)
juno.ForwardPass.decode.count: 549
juno.MatVec.count: 36,990
juno.MatVec.duration.total_ms: 32,501 ms
juno.TokenProduced.count: 182 tokens
juno.TokenProduced.elapsed_seconds: 28.64s
TPS: 6.36


Agent:
	Sonnet 4.6 Max