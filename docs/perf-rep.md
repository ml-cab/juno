---

## Performance Issues — Priority Order

### 🔴 Critical: `matVecF32raw` — ByteBuffer allocated per row inside parallel lambda

```java
// CURRENT — creates a new ByteBuffer object for EVERY row, inside the parallel loop
java.util.stream.IntStream.range(0, rows).parallel().forEach(r -> {
    java.nio.ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN); // ← HERE
    ...
});
```

The comment says `ByteBuffer.wrap()` is "view-only (no copy)" — true, but it still allocates a new `ByteBuffer` object *and* a duplicate `HeapByteBuffer` wrapper for every row. For a 2048×2048 F32 matrix, that's **2048 ByteBuffer allocations per matVec call**, all flooding the GC. The fix is simple — read the float directly with bit manipulation, no ByteBuffer needed:

```java
// FAST — zero allocation, ~10× faster
private static float readF32LE(byte[] raw, int byteOffset) {
    int bits = (raw[byteOffset]     & 0xFF)
             | ((raw[byteOffset+1] & 0xFF) << 8)
             | ((raw[byteOffset+2] & 0xFF) << 16)
             | ((raw[byteOffset+3] & 0xFF) << 24);
    return Float.intBitsToFloat(bits);
}
```

---

### 🔴 Critical: ForkJoinPool not partitioned across JVMs

Every JVM uses `ForkJoinPool.commonPool()` with default parallelism = `availableProcessors() - 1`. On your laptop with e.g. 8 cores, **each of the 3 JVMs believes it owns all 8 cores** — so you get 3 × 7 = 21 ForkJoin worker threads fighting for 8 cores. This is the primary reason your throughput is ~1-2 tok/s instead of the expected ~5-8 tok/s.

Fix in `run.sh` — pass parallelism proportionally to each forked JVM:

```bash
# In the node JVM launch args:
-Djava.util.concurrent.ForkJoinPool.common.parallelism=$(($(nproc) / NUM_NODES))
```

Or better, the pipeline coordinator should set this when forking node processes.

---

### 🟠 High: ActivationCodec — sequential encode/decode, wrong endianness

Two issues in `ActivationCodec`:

**a)** The FLOAT16 encode/decode loops are **fully sequential** — no parallel stream — while sitting directly on the hot path between every pipeline node. For TinyLlama (hiddenDim=2048), every inter-node activation transfer does 2048 float→half conversions one at a time.

**b)** BIG_ENDIAN is used throughout, but every x86/ARM laptop is little-endian. This means **every `putFloat`/`getFloat`/`putShort`/`getShort` call byte-swaps**. There's no reason for big-endian here since both ends are always the same machine or same architecture cluster.

```java
// CURRENT
ByteBuffer buf = ByteBuffer.allocate(floats.length * 2).order(ByteOrder.BIG_ENDIAN); // ← slow
for (float f : floats) buf.putShort(floatToHalf(f));  // ← sequential

// BETTER: parallel + little-endian
byte[] out = new byte[floats.length * 2];
IntStream.range(0, floats.length).parallel().forEach(i -> {
    short h = floatToHalf(floats[i]);
    int off = i * 2;
    out[off]   = (byte)(h & 0xFF);        // little-endian — no swap
    out[off+1] = (byte)((h >> 8) & 0xFF);
});
```

Also: Java 20+ has `Float.floatToFloat16(f)` / `Float.float16ToFloat(s)` as built-in intrinsics that JIT-compiles to native FP16 conversion instructions. You're already using `--enable-preview` — if you're on Java 20+, drop the manual `floatToHalf`/`halfToFloat` in `ActivationCodec` entirely.

---

### 🟠 High: GC pressure — `new float[rows]` on every matVec

Every single `matVec` call allocates a fresh `float[rows]`. A TinyLlama forward pass does roughly **7 matVec calls × 22 layers = 154 allocations per token**, ranging from 256 to 32,000 floats. These are short-lived objects that pressure the young-gen GC, adding stop-the-world pause time between tokens.

Fix: use a `ThreadLocal<float[]>` scratch buffer with lazy resize:

```java
private static final ThreadLocal<float[]> SCRATCH = ThreadLocal.withInitial(() -> new float[0]);

static float[] matVec(float[] A, float[] x, int rows, int cols) {
    float[] y = SCRATCH.get();
    if (y.length < rows) { y = new float[rows]; SCRATCH.set(y); }
    // fill y[0..rows-1] in-place, return a trimmed view or pass length separately
    ...
}
```

This requires changing the return contract slightly (caller provides the output buffer or uses a scoped result), but eliminates essentially all matVec GC pressure.

---

### 🟡 Medium: No SIMD / Vector API

The `--enable-preview` flag is set but no `jdk.incubator.vector` usage exists anywhere in the codebase. The inner loop of every `matVec`:

```java
for (int c = 0; c < cols; c++)
    acc += A[base + c] * x[c];   // ← perfectly vectorizable
```

…is the hottest code in the entire system. With Java's Vector API this becomes 8-wide SIMD on AVX2 (4 float MAC operations per cycle vs 1 scalar), giving up to **8× speedup** on the matVec inner loop. This is the single highest-ROI code change for CPU-only inference.

---

