# Vector API (SIMD) Adoption — Design and Implementation Plan

Status: PLAN ONLY — no source changes in this pass. Assumes the batched-prefill
plan (`batched-prefill-plan.md`) is already implemented and merged: `MatVec.sgemm`
exists, `CpuMatVec.sgemm` already does weight-stationary blocking (one row of
`A` loaded once, multiplied against all `B` batch columns), and prefill calls
`sgemm` instead of `B` calls to `sgemv`. This plan is scoped to the arithmetic
inside each row's dot product — replacing the scalar inner loops with
`jdk.incubator.vector` lanes — and treats batching (across tokens) and SIMD
(across elements within one row) as orthogonal, stacking optimizations.

No other docs are touched in this pass, per the same constraint as the
batched-prefill plan.

## 0. Why this is the right next step, and why it is independent of batching

Batched prefill fixes *how many times* each weight row is streamed from
memory (once per layer per window, not once per layer per token). It does
not change *how* each row's dot product against the batch is computed —
that is still a scalar `for` loop, one `float` multiply-add at a time, in
every hot path traced in `LlamaTransformerHandler`:

- `matVec(float[] A, float[] x, int rows, int cols)` — the plain F32 path
  (lines ~1162-1176), scalar `acc += A[base + c] * x[c]`.
- `matVecQ4Kraw` / `matVecQ5Kraw` / `matVecQ6Kraw` / `matVecQ3Kraw` /
  `matVecQ2Kraw` / `matVecQ8_0raw` / `matVecF16raw` — every quantized
  raw-bytes path, each with the same shape: unpack a block's packed bits
  into per-element weights, then scalar-accumulate against `x`. Concretely,
  in `matVecQ4Kraw`, the innermost loop unpacks one nibble at a time from a
  packed byte and multiplies it against one element of `x`:

  ```java
  for (int i = 0; i < 32; i++)
      acc += (scale0 * (raw[qsBase + qi + i] & 0x0F) - min0) * x[xBase + g + i];
  ```

  This is exactly the shape SIMD gather/unpack + FMA lanes are built for.
- `gqa()` — the attention score dot product (`dot += q[...] * kCache[...]`)
  and the weighted-value accumulation (`out[...] += w * vCache[...]`), both
  scalar loops over `headDim`.
- `rope()` — per-element trig-based rotation; smaller win (transcendental
  `cos`/`sin` dominate, not the multiply-add), addressed separately in
  Section 5 as a lower-priority item.

None of this depends on whether the caller is `sgemv` (batch=1) or `sgemm`
(batch=B, after the prefill plan): the *row* dot product is the same
operation either way, just called across more columns of `x` when batched.
SIMD-izing the row loop benefits `sgemv`, `sgemm`, and `gqa` uniformly —
this is additive to batching, not a substitute for it, and is the next
highest-leverage CPU change once batching removes the O(N) redundant
weight-matrix traversal.

## 1. Goals

1. Replace scalar inner-loop dot products in `matVec` (F32), all seven
   quantized raw-bytes paths, and `gqa`'s score/weighted-sum loops with
   `jdk.incubator.vector` (`FloatVector`, `ByteVector`) lane operations.
2. Preserve numerical behavior within float-reduction-order tolerance —
   SIMD lane-sum reduction can reorder additions relative to the scalar
   loop; this is the same class of acceptable drift already documented for
   the batched-prefill plan's GPU parity tests (Section 4.3 there), not a
   new risk category.
3. Preserve every existing `MatVec`/`ForwardPassHandler` public contract —
   this is an internal-implementation change to the bodies of `matVec*`
   static methods and `gqa()`, not a signature or architecture change.
4. Fall back cleanly to the existing scalar path on any JVM/CPU where the
   Vector API's preferred species is not meaningfully wider than scalar
   (e.g. no AVX2/AVX-512 equivalent, or a non-x86/non-ARM target the
   incubator module doesn't have a good species for) — the scalar loop
   already in the codebase remains, unconditionally correct, as the
   fallback body.
5. Cover both the plain-F32 path and quantized dequantization paths — the
   quantized paths are the ones every real Q4_K_M/Q5_K/Q6_K deployment
   actually uses (per `README.md`'s example models), so this is not
   optional scope; skipping it would SIMD-accelerate only the untypical
   F32 case.
6. Gate the whole feature behind a flag with a safe default, mirroring the
   `--prefill single|batched` precedent from the batched-prefill plan
   (`--cpuLoops scalar|vector`, default `vector`), so there is a
   byte-for-byte scalar escape hatch without a rebuild.

## 2. Non-goals

- **GPU paths.** `CudaMatVec`/`RocmMatVec` already run on vendor SIMD
  (CUDA cores / Tensor Cores via cuBLAS/rocBLAS) — `jdk.incubator.vector`
  is a CPU-only concern. No GPU code is touched here.
- **Auto-vectorization reliance.** The JIT's C2 auto-vectorizer can
  sometimes vectorize simple scalar loops, but the loops here are
  bit-unpacking-heavy (nibble/6-bit-packed reads, branchy scale/min
  lookups) — exactly the shape auto-vectorization reliably fails on. This
  plan uses the explicit Vector API rather than hoping the JIT finds it,
  which is also why it is worth doing as deliberate work rather than
  leaving to chance.
- **Changing the quantization formats themselves** (Q4_K/Q5_K/Q6_K block
  layout, GGUF tensor structure) — this plan only changes how an existing,
  already-correct block layout is unpacked and multiplied, not the layout.
- **Rewriting RoPE for SIMD** as a first-class goal — included only as a
  low-priority Section 5 item, since its cost is dominated by
  `Math.cos`/`Math.sin` transcendentals, which the Vector API's
  `SIN`/`COS` lane operations can address but with smaller relative payoff
  than the matmul paths; sequenced last.
- **Multi-JVM-vendor guarantees.** `jdk.incubator.vector` is part of
  OpenJDK proper (not vendor-specific), but is still an incubator module
  under `JEP 338`/`438`/`460`/`489`-style evolution — this plan assumes
  whatever JDK 25 distribution the project already targets
  (`maven.compiler.release=25` in the root `pom.xml`) includes it, and
  notes the fallback (Section 4.5) precisely because incubator module
  availability across minor JDK 25 update releases is not guaranteed to be
  identical to a GA API.

## 3. New abstractions (new classes preferred over extending existing ones)

Per rule D4 (prefer new classes over extending existing ones), the SIMD
implementations live in **new** classes, not inline rewrites of
`LlamaTransformerHandler`'s existing static methods. The existing scalar
methods (`matVec`, `matVecQ4Kraw`, etc.) are left exactly as they are today
and become the fallback body, called from a new dispatch point.

### 3.1 `VectorMatVecOps` (new final class, `node` module)

```java
/**
 * SIMD (jdk.incubator.vector) implementations of the dot-product-heavy
 * inner loops used by CpuMatVec and the quantized matVec paths.
 *
 * Every method here has a scalar twin already in LlamaTransformerHandler;
 * this class never changes numerical intent, only how the reduction is
 * carried out across CPU vector lanes. Falls back to the scalar body
 * automatically when isAvailable() is false (see VectorSupport).
 */
final class VectorMatVecOps {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;

    private VectorMatVecOps() {}

    static float[] matVecF32(float[] A, float[] x, int rows, int cols) { ... }

    static float dotF32(float[] a, int aOffset, float[] b, int bOffset, int length) { ... }

    // one per quantized format — Section 4.2
    static void accumulateQ4K(byte[] raw, int rowByteOffset, float[] x, int xBase, /* out */ float[] accHolder) { ... }
    ...
}
```

- `dotF32` is the single shared primitive: SIMD dot product over a
  contiguous `float[]` range with scalar tail handling
  (`F_SPECIES.loopBound(length)` for the vectorizable prefix, plain scalar
  loop for the remainder — standard Vector API idiom). `matVecF32` and
  `gqa`'s score loop both become thin callers of this one primitive,
  keeping the SIMD logic in exactly one place per rule D3 (KISS).
- Quantized-format methods stay one-per-format (mirroring the existing
  `matVecQ4Kraw`/`matVecQ5Kraw`/... structure) because each format's bit
  layout (4-bit/5-bit/6-bit/2-bit/3-bit nibbles, per-sub-block scale/min
  packing) is genuinely different unpacking logic, not a single shared
  loop — forcing them into one generic method would violate KISS in the
  other direction (one confusing branchy method instead of seven clear
  ones).

### 3.2 `VectorSupport` (new final class, `node` module)

```java
/**
 * Runtime capability check for jdk.incubator.vector, cached once at class
 * load (zero per-call overhead), following the same isAvailable() pattern
 * already used by CudaAvailability / RocmAvailability.
 */
final class VectorSupport {
    private static final boolean AVAILABLE = probe();
    static boolean isAvailable() { return AVAILABLE; }
    static int preferredFloatLanes() { return FloatVector.SPECIES_PREFERRED.length(); }
    private static boolean probe() {
        try {
            // touch FloatVector.SPECIES_PREFERRED and run a 1-lane sanity op;
            // false on any LinkageError/UnsupportedOperationException
            ...
            return FloatVector.SPECIES_PREFERRED.length() > 1;
        } catch (Throwable t) {
            return false;
        }
    }
}
```

- Deliberately mirrors `CudaAvailability`/`RocmAvailability`'s existing
  "detection via try/probe, zero heap allocation, cached boolean" shape
  (`docs/agent-arch.txt` describes both) — this codebase already has a
  house style for hardware-capability detection, and this plan reuses it
  rather than inventing a new one.
- `preferredFloatLanes() > 1` is the deliberate bar: on a target where the
  preferred species is 1 lane wide (no real SIMD unit reachable), SIMD
  code would just add dispatch overhead for no gain, so `isAvailable()`
  returns `false` and every call site falls back to the existing scalar
  method transparently.

### 3.3 Dispatch point: `CpuMatVec` / `LlamaTransformerHandler` call sites

No new interface method is needed on `MatVec` — `CpuMatVec.sgemv`/`sgemm`
keep their exact signatures (Section 2: no architecture change). Only the
*body* of `LlamaTransformerHandler.matVec(float[], ...)` and each
`matVecQ*Kraw`/`matVecF16raw` method gains a one-line dispatch at the top:

```java
static float[] matVec(float[] A, float[] x, int rows, int cols) {
    if (VectorSupport.isAvailable() && cpuLoopMode == CpuLoopMode.VECTOR)
        return VectorMatVecOps.matVecF32(A, x, rows, cols);
    // existing scalar body, byte-for-byte unchanged, unreachable only when SIMD is used
    float[] y = new float[rows];
    ...
}
```

This keeps the existing scalar method as the literal, unmodified fallback
body (same "kept verbatim" discipline as `--prefill single` in the
batched-prefill plan), rather than deleting it and reconstructing it from
memory — the fallback's trustworthiness comes from it being the exact code
already shipping today, not a re-derivation of it.

## 4. Per-module implementation plan

### 4.1 `node` — `VectorMatVecOps.dotF32` / `matVecF32`

```java
static float dotF32(float[] a, int aOffset, float[] b, int bOffset, int length) {
    var acc = FloatVector.zero(F_SPECIES);
    int i = 0;
    int bound = F_SPECIES.loopBound(length);
    for (; i < bound; i += F_SPECIES.length()) {
        var va = FloatVector.fromArray(F_SPECIES, a, aOffset + i);
        var vb = FloatVector.fromArray(F_SPECIES, b, bOffset + i);
        acc = va.fma(vb, acc);
    }
    float sum = acc.reduceLanes(VectorOperators.ADD);
    for (; i < length; i++) // scalar tail — length not a multiple of lane width
        sum += a[aOffset + i] * b[bOffset + i];
    return sum;
}
```

- `matVecF32(A, x, rows, cols)` becomes: same `rows >= 256` parallel/serial
  split already in the existing method (kept as-is — that threshold logic
  is orthogonal to SIMD and stays put), inner body calls
  `dotF32(A, r * cols, x, 0, cols)` instead of the scalar accumulation
  loop.
- Unit test: `VectorMatVecOpsF32Test` — for random `A`/`x` at several
  `cols` values including non-multiples of the preferred lane width (to
  exercise the scalar tail path), assert `VectorMatVecOps.matVecF32(...)`
  equals `LlamaTransformerHandler`'s existing scalar `matVec(...)` within
  a documented epsilon (not bitwise — FMA and lane-sum reduction order
  differ from the scalar left-to-right sum). This is the business-logic
  test per rule D1: proving the SIMD path agrees with the trusted scalar
  path is the single highest-value test in this change, exactly as the
  batched-prefill plan's own parity tests were for that change.

### 4.2 `node` — quantized paths (`matVecQ4Kraw` et al.)

Each format gets its own `VectorMatVecOps` method following the same
shape: unpack a `BLOCK_SIZE`-wide (256 for K-quants, 32 for Q8_0) chunk of
weights into a lane-width-aligned scratch buffer using `ByteVector` masked
loads and shift/AND lane ops, then FMA against the corresponding slice of
`x`, accumulating into a running `FloatVector` sum.

Concrete example — Q4_K (the format actually used by the reporter's model
per `docs/performance.md`'s Q4_K_M example configs), replacing this
existing scalar tail:

```java
for (int i = 0; i < 32; i++)
    acc += (scale0 * (raw[qsBase + qi + i] & 0x0F) - min0) * x[xBase + g + i];
```

with a lane-batched unpack: load 32 packed bytes as a `ByteVector`, mask
`& 0x0F` across all lanes at once, widen to `FloatVector`, multiply by the
broadcast `scale0` and subtract broadcast `min0` in one FMA-shaped
expression, multiply against the corresponding `FloatVector` loaded from
`x[xBase + g .. xBase + g + 32)`, accumulate. The high-nibble half
(`>> 4`) is the same shape with a different mask/shift.

- This is the highest-payoff single item in this whole plan: Q4_K is the
  default quantization for every example model in `README.md`
  (`tinyllama-1.1b-chat-v1.0.Q4_K_M`, `Phi-3.5-mini-instruct-Q4_K_M`), so
  this is the format actually exercised by the reported vision stall and
  by the majority of real usage — not a hypothetical F32 improvement.
- Q5_K/Q6_K/Q3_K/Q2_K follow the identical pattern with their own
  bit-widths; Q8_0 is the simplest (already byte-aligned, no sub-byte
  packing — closest to a plain `dotF32` after a widen-byte-to-float step,
  so it doubles as the easiest correctness check to land first).
- F16 (`matVecF16raw`) — widen each `short` (read via `readLE16`) to
  `float` via `FloatVector`'s half-precision conversion lanes if
  available in this JDK's incubator surface, else keep the scalar
  half-to-float conversion per element and SIMD only the subsequent FMA —
  a smaller win than the K-quant formats but still additive.
- Unit tests, one per format, same shape as 4.1:
  `VectorMatVecOpsQ4KTest`, `VectorMatVecOpsQ5KTest`, `VectorMatVecOpsQ6KTest`,
  `VectorMatVecOpsQ3KTest`, `VectorMatVecOpsQ2KTest`, `VectorMatVecOpsQ8_0Test`,
  `VectorMatVecOpsF16Test` — each constructs a small GGUF-shaped raw block
  fixture (reusing whatever fixture-building helpers
  `PhiQuantizedMatVecTest`/`MatVecBackendContractTest` already use, per
  `docs/phi3-inference-handoff.md`'s test list, rather than duplicating
  fixture logic — new test classes, shared fixture helpers), and asserts
  agreement with the existing scalar `matVecQ*Kraw` within epsilon.

### 4.3 `node` — `gqa()` score and weighted-sum loops

```java
// score computation — was: for (d) dot += q[...] * kCache[...]
scores[t] = VectorMatVecOps.dotF32(q, qBase, kCache, kOffset, Hd) * scale;

// weighted-value accumulation — was: for (d) out[...] += w * vCache[...]
VectorMatVecOps.fmaAccumulate(out, outBase, vCache, vOffset, w, Hd);
```

- New `VectorMatVecOps.fmaAccumulate(float[] out, int outOffset, float[] v,
  int vOffset, float scalarWeight, int length)` — broadcast `scalarWeight`
  into a `FloatVector`, FMA against loaded `v` lanes, add into loaded `out`
  lanes, store back; scalar tail for the remainder, same idiom as 4.1.
- This is **unaffected by whether attention itself is batched** — per the
  batched-prefill plan's Section 2 non-goal, attention stays a per-position
  loop even after batching; SIMD-izing its inner `Hd`-wide (headDim, e.g.
  128) dot products and accumulations is a clean, independent win on top,
  and one of the few remaining hot loops the batched-prefill plan
  explicitly left untouched (batched-prefill-plan.md Section 2: "this
  plan batches the linear projections... it does not rewrite attention").
- Unit test: `GqaVectorParityTest` — fixed `q`/`kCache`/`vCache` fixture,
  assert SIMD-path `gqa()` output matches today's scalar output within
  epsilon, across a few `headDim` values including one not a multiple of
  the preferred lane width (e.g. headDim=80, seen on some real GGUF
  configs, versus headDim=128) to force the scalar-tail path.

### 4.4 Feature flag: `--cpuLoops scalar|vector`

Per rule D4 and the same reasoning as `--prefill single|batched`
(user-facing capability toggle, safe default, escape hatch without a
rebuild):

```java
public enum CpuLoopMode {
    SCALAR, // force the existing scalar loops — bisection / correctness-comparison /
            // unsupported-CPU escape hatch
    VECTOR; // default — prefer jdk.incubator.vector lanes; transparently falls back
            // to the scalar body per-call, per VectorSupport.isAvailable(), on any
            // host where SIMD is not actually available — "vector" is a preference,
            // not a hard requirement, so this mode is never less safe than SCALAR

    public static CpuLoopMode parse(String s) { ... } // same case-insensitive + WARNING-on-garbage
                                                        // pattern as PrefillMode.parse and parseDtype
}
```

- Placement: `node` module (unlike `PrefillMode`, which lives in
  `coordinator` because only `GenerationLoop` reads it — `CpuLoopMode` is
  read by `LlamaTransformerHandler`'s static dispatch, which lives in
  `node`).
- Only two values, deliberately — no separate "force SIMD and fail fast if
  unavailable" third state. `VECTOR` already means "prefer SIMD, silently
  step down to scalar per call when `VectorSupport.isAvailable()` is
  false" (Section 3.3's dispatch snippet), so a would-be `ON` value would
  either duplicate `VECTOR` exactly or need to crash on an unsupported
  host — and a flag whose default value can crash the process on some
  hardware is not an acceptable default. Keeping it binary
  (`scalar`/`vector`) means the default is unconditionally safe everywhere
  `--prefill batched`'s default already is, which is the same bar this
  flag is held to.
- `ConsoleMain` flag parsing follows the exact same pattern established for
  `--prefill` in the batched-prefill plan (Section 4.8 there): explicit
  case, `parseCpuLoopMode` helper, `WARNING` + fallback to `VECTOR` on
  garbage input, help text line
  (`--cpuLoops scalar|vector     CPU matmul inner-loop strategy (default: vector)`),
  `scripts/run.sh`/`scripts/run.bat` parity (`CPU_LOOPS_MODE` env var) from
  the start rather than as a follow-up.
- Default: `VECTOR`. This is safe to default on (unlike a hypothetical
  always-crash-if-unsupported mode) precisely because `VECTOR` already
  encodes the fallback: on a host where `VectorSupport.isAvailable()` is
  `false`, `--cpuLoops vector` (the default, or the flag omitted entirely)
  behaves identically to `--cpuLoops scalar` — same reasoning that made
  `--prefill batched` a safe default in the batched-prefill plan, applied
  here to a hardware-dependent capability instead of an always-available
  one.
- Unit tests: `CpuLoopModeTest` (parse cases, mirrors `PrefillModeTest`),
  `ConsoleMainCpuLoopsFlagTest` (mirrors `ConsoleMainPrefillFlagTest`/
  `ConsoleMainDtypeTest` reflection-driven parsing test: flag absent →
  `VECTOR`; `--cpuLoops scalar` → `SCALAR`; `--cpuLoops garbage` →
  `WARNING` to stderr + falls back to `VECTOR`), and a `VectorSupportTest`
  asserting `isAvailable()` never throws (only ever returns `true`/`false`)
  regardless of the actual host CPU.

### 4.5 Fallback correctness on unsupported hosts

- `VectorSupport.probe()`'s try/catch must catch `Throwable`, not just a
  named exception — incubator module absence, `--add-modules` not passed,
  or a species query failing on an exotic CPU could surface as
  `NoClassDefFoundError`, `UnsupportedOperationException`, or an
  `IllegalArgumentException` from a degenerate species, depending on JDK
  build and platform. A narrower catch risks a hard crash on first call
  instead of a clean scalar fallback, which would make `VECTOR` mode
  unsafe — the entire point of `VECTOR` being the default is that it must
  never be worse than `SCALAR` on any host.
- Build/runtime requirement: `--add-modules jdk.incubator.vector` must be
  added to compilation (`node/pom.xml`'s `maven-compiler-plugin`
  `compilerArgs`, alongside the existing `--enable-native-access=ALL-UNNAMED`
  seen in `node/pom.xml`/`juno-master/pom.xml`) and to every runtime launch
  path: `scripts/run.sh`/`run.bat` (both `local` and `cluster` commands),
  the shaded `juno-node`/`juno-master` jar's manifest or launch wrapper,
  and `juno-master/pom.xml`'s existing surefire `argLine` (which already
  carries `--enable-preview --enable-native-access=ALL-UNNAMED` — this is
  the correct, single place to add the new flag for that module's test
  run, following existing precedent rather than inventing a second
  mechanism). Missing this on any one launch path would make
  `VectorSupport.isAvailable()` throw at class-init time on that path
  specifically — caught by `probe()`'s broad `catch (Throwable)`, so the
  observable effect is a silent, correct fallback to scalar rather than a
  startup crash, but it should still be treated as a build-config bug to
  fix (SIMD not activating anywhere is a missed-optimization bug, not a
  correctness bug, and should be visible in CI, see Section 4.6).
- No module-info.java exists anywhere in this project (`find . -name
  module-info.java` returns nothing) — everything runs on the unnamed
  classpath module, so this is purely an `--add-modules` flag concern, not
  a `module-info` `requires` concern.

### 4.6 CI / build visibility

- New Maven profile or `mvn test -DcpuLoops=vector` convention (mirroring however
  `-Prelease-sign`/`-Pcentral-publish` profiles already gate optional build
  behavior in the root `pom.xml`) that fails the build if
  `VectorSupport.isAvailable()` is `false` on the CI runner — so a CI
  environment silently missing `--add-modules jdk.incubator.vector` (or
  running on a JDK distribution without the incubator module) is caught
  as a build-config regression rather than quietly shipping scalar-only
  performance while `VECTOR` mode reports success everywhere.
- This is deliberately a separate, opt-in check (not a default test
  assertion), since `VECTOR` mode falling back to scalar on a given CI
  runner is not itself a bug — only *not noticing* that it happened would
  be.

## 5. Lower-priority follow-up (not in this pass's scope, noted for completeness)

- **RoPE vectorization** — `rope()`'s cost is dominated by
  `Math.cos`/`Math.sin` per element; `jdk.incubator.vector`'s
  `VectorOperators.SIN`/`COS` (where available) would batch the
  transcendental calls across `headDim/2` angles at once. Smaller
  relative payoff than the matmul paths above (RoPE is O(headDim) per
  token per layer, versus O(hiddenDim × ffnDim) for the FFN matmuls), so
  sequenced after everything in Section 4, only if profiling after
  Section 4 lands shows RoPE as a non-trivial fraction of remaining
  wall-clock time.
- **Vectorized softmax** (`softmax()` in `gqa()`) — same shape argument as
  RoPE: `Math.exp` per element dominates, smaller payoff, defer until
  profiling justifies it.

## 6. Suggested build order (smallest safe increments first)

1. `VectorSupport` + `VectorSupportTest` — pure capability probe, zero
   behavioral risk, lands independently of everything else.
2. `VectorMatVecOps.dotF32`/`matVecF32` + parity test (Section 4.1) — the
   simplest case, proves the FMA/lane-sum/tail-handling pattern once,
   reused by every subsequent method.
3. `CpuLoopMode` + `ConsoleMain` flag wiring (Section 4.4) — small, isolated,
   unblocks manual A/B benchmarking of every step after this one against a
   real running binary rather than only unit tests, same reasoning as
   sequencing `--prefill`'s flag work early in the batched-prefill plan.
4. Q8_0 quantized path (Section 4.2) — simplest quantized format (already
   byte-aligned), good second correctness checkpoint before the more
   intricate K-quant bit-packing.
5. Q4_K quantized path (Section 4.2) — highest real-world payoff (default
   format for the example models in `README.md`); land this before the
   remaining K-quant formats so the most common deployment benefits first.
6. Q5_K, Q6_K, Q3_K, Q2_K, F16 quantized paths (Section 4.2) — same
   pattern, decreasing real-world frequency, can proceed in any order or
   in parallel once step 5 has proven the pattern.
7. `gqa()` score/weighted-sum SIMD (Section 4.3) — independent of the
   matVec work, can be developed in parallel with steps 4-6 once step 2's
   `dotF32`/`fmaAccumulate` primitives exist.
8. CI visibility check (Section 4.6) — last, once there is something
   meaningful for it to guard.

Each step compiles and passes the full existing test suite (plus its own
new parity test) before the next begins, per KISS — matching the same
discipline the batched-prefill plan used.

## 7. Definition of done for this feature

- All new parity tests listed in Section 4 pass on at least one CI runner
  with `VectorSupport.isAvailable() == true`, and the full existing test
  suite continues to pass unmodified under `--cpuLoops scalar` (proving the
  fallback path is byte-for-byte today's behavior, not a re-derivation of
  it) and under `--cpuLoops vector` (the default) on a runner where SIMD
  is unavailable.
- A measured wall-clock improvement on the existing
  `docs/performance.md` reproduction commands (`./juno local --dtype
  FLOAT16 --max-tokens 50 --jfr 5m` against `tinyllama-1.1b-chat-v1.0-q4_k_m.gguf`)
  comparing `--cpuLoops scalar` vs `--cpuLoops vector` (the default),
  recorded the same way `docs/performance.md` already documents
  (`juno.MatVec.durationMs` p99, `juno.ForwardPass.durationMs`/`prefillMs`
  p95) — this plan should not be called done on code landing alone
  without a number to show for it, consistent with `docs/performance.md`'s
  own "Submitting results" section.
- No new public API breaks: `MatVec`, `ForwardPassHandler`,
  `LlamaTransformerHandler`'s package-private static method signatures are
  all unchanged — every change in this plan is to method bodies plus two
  new package-private classes and one new public enum + CLI flag.
- `--cpuLoops` appears in `--help` output and behaves identically on
  `scripts/run.sh`/`scripts/run.bat`, matching the Windows-parity bar the
  batched-prefill plan's `--prefill` flag was held to.
- Follow-up doc updates (explicitly deferred, not part of this plan's
  output, same posture as the batched-prefill plan): `docs/agent-arch.txt`
  (new classes/flag), `docs/howto.md` (`--cpuLoops` flag reference),
  `docs/performance.md`/`juno_test_matrix.html` re-measurement with the
  vector path enabled once implemented.
