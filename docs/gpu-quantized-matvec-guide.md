# GPU quantized matVec — implementation guide

> Read this before touching any GPU or quantization code.
> Written for the next session that picks up this work.

---

## What this guide covers

The current GPU path (`CudaMatVec`) takes a `float[]` weight matrix, copies it
to device, runs `cublasSgemv`, and copies the result back. Because all projection
weights are stored as `QuantizedTensor` (raw Q5_K / Q6_K bytes), every GPU call
first dequantizes on the CPU, then does a full H2D copy of the float32 result.
For a 2048×2048 Q5_K matrix that is ~4.5 MB of raw bytes → 16 MB of float32 →
16 MB H2D per matVec call, per token, per layer. This is the single biggest
remaining performance gap.

The fix: keep raw quantized bytes resident on the GPU permanently. Launch a
CUDA kernel that dequantizes and multiplies in one pass — no float32 expansion,
no H2D weight traffic. This is exactly what llama.cpp's CUDA backend does
(`k_quants.cu`).

---

## Codebase facts to know first

**`GgufReader.QuantizedTensor`** — record holding `(String name, int type, long
nelems, byte[] data)`. The `type` integer is the GGML type:
- `0`  = F32 (4 bytes/elem)
- `8`  = Q8_0 (34 bytes/32 elems)
- `12` = Q4_K (144 bytes/256 elems)
- `13` = Q5_K (176 bytes/256 elems)
- `14` = Q6_K (210 bytes/256 elems)

TinyLlama Q5_K_M: projection weights are type 13 (Q5_K) and type 14 (Q6_K).
The raw `data` byte array is the exact on-disk GGUF block layout — no padding,
no reordering. The block layouts are documented in the existing
`matVecQ*raw` methods in `LlamaTransformerHandler`.

**`MatVec` interface** — two overloads:
1. `sgemv(float[] A, float[] x, int rows, int cols)` — host path (allocates per
   call, used in tests and CPU-only nodes)
2. `sgemv(DeviceFloatMatrix A, float[] x)` — resident float32 on GPU (exists but
   not used in inference because weights were moved to `QuantizedTensor`)

A third overload is needed: `sgemv(DeviceQuantizedMatrix A, float[] x)`.

**`DeviceFloatMatrix`** — uploads a `float[]` once H2D and keeps it resident.
Pattern to copy for `DeviceQuantizedMatrix`: same lifecycle, but stores raw
`byte[]` + GGML type integer on device instead of float32.

**`CudaMatVec`** — calls `cublasSgemv_v2`. The quantized kernel cannot use cuBLAS
— it is a custom CUDA kernel. `CudaMatVec` needs a new path that dispatches to
the kernel.

**`LlamaTransformerHandler`** — calls `matVec(QuantizedTensor, x, rows, cols)`.
This is a static method that dispatches to the `matVecQ*raw` CPU methods. When a
`MatVec` backend that supports `DeviceQuantizedMatrix` is active, those calls
should go to the GPU instead.

**`GpuContext`** — holds the `cublasContext` handle and device index. The kernel
launcher needs the device index (`ctx.deviceIndex()`). It does not need the
cuBLAS handle — custom kernels launch via the CUDA runtime API (`cudaLaunchKernel`
or the JavaCPP `CUfunction` path).

**`MatVecEvent`** — JFR event. The `backend` field string should be set to
`"cuda-quant-q5k"` / `"cuda-quant-q6k"` etc. so JFR captures the new path
separately from the float32 GPU path.

---

## Architecture of the new classes

```
GgufReader.QuantizedTensor   (existing — raw bytes on heap)
        │
        │  DeviceQuantizedMatrix.upload(ctx, qt)
        ▼
DeviceQuantizedMatrix        (NEW — raw bytes resident on GPU device)
        │
        │  CudaMatVec.sgemv(DeviceQuantizedMatrix, x)
        ▼
CudaQuantizedKernels          (NEW — launches .cu kernels via JNI / JavaCPP)
        │
        │  cudaLaunchKernel(...)
        ▼
 q5k_sgemv.cu / q4k_sgemv.cu / q6k_sgemv.cu   (NEW — CUDA kernels)
```

`LlamaTransformerHandler` needs a new map: `Map<String, DeviceQuantizedMatrix>`
built at load time when the backend is a `CudaMatVec`. The static `matVec()`
dispatch checks this map before falling back to the CPU path.

---

## Step 1 — `DeviceQuantizedMatrix`

New class, parallel to `DeviceFloatMatrix`. Store:
- `int ggmlType` — the GGML type integer
- `long nelems` — total element count (needed to derive row/block counts)
- `Pointer dData` — device pointer to raw quantized bytes (exactly `data.length`
  bytes, copied H2D once)
- `GpuContext ctx`, `int rows`, `int cols`

```java
public final class DeviceQuantizedMatrix implements AutoCloseable {

    private final GpuContext ctx;
    private final Pointer dData;
    private final int ggmlType;
    private final int rows;
    private final int cols;
    private volatile boolean closed;

    public static DeviceQuantizedMatrix upload(
            GpuContext ctx, GgufReader.QuantizedTensor qt, int rows, int cols) {
        long bytes = qt.data().length;
        PointerPointer pp = new PointerPointer(1);
        try {
            checkCuda(cudart.cudaSetDevice(ctx.deviceIndex()), "cudaSetDevice");
            checkCuda(cudart.cudaMalloc(pp, bytes), "cudaMalloc");
            Pointer d = pp.get(0);
            try (BytePointer h = new BytePointer(qt.data())) {
                checkCuda(
                    cudart.cudaMemcpy(d, h, bytes, cudart.cudaMemcpyHostToDevice),
                    "cudaMemcpy(quant H2D)");
            }
            return new DeviceQuantizedMatrix(ctx, d, qt.type(), rows, cols);
        } finally {
            pp.close();
        }
    }

    // rows(), cols(), ggmlType(), devicePointer(), isClosed(), close() — same pattern as DeviceFloatMatrix
}
```

Upload happens once at model load time (inside `LlamaTransformerHandler`
constructor, when `backend instanceof CudaMatVec`).

---

## Step 2 — The CUDA kernels

Three kernels, one per GGML type used by TinyLlama Q5_K_M:
- `q5k_sgemv_kernel` — type 13
- `q4k_sgemv_kernel` — type 12
- `q6k_sgemv_kernel` — type 14

### Launch geometry

Each kernel computes one output element `y[row]` per CUDA block (or per warp —
see below). Grid = `(rows, 1, 1)`. Block = `(BLOCK_SIZE, 1, 1)` where
`BLOCK_SIZE = 256` (one warp group for reduction). Within a block, threads
cooperate on the dot product for one row via warp shuffle reduction.

```
gridDim.x  = rows           // one block per output row
blockDim.x = 256            // 8 warps = 256 threads, good occupancy
```

### Q5_K kernel skeleton

Block layout (176 bytes / 256 elems):
`[d:f16(2)][dmin:f16(2)][sc:12][qh:32][qs:128]`

```c
__global__ void q5k_sgemv_kernel(
    const uint8_t* __restrict__ A,   // raw Q5_K bytes, row-major blocks
    const float*   __restrict__ x,   // input vector [cols]
    float*         __restrict__ y,   // output vector [rows]
    int rows, int cols)
{
    const int row        = blockIdx.x;
    const int tid        = threadIdx.x;       // 0..255
    const int blocksPerRow = cols / 256;
    const int bytesPerRow  = blocksPerRow * 176;

    float acc = 0.0f;

    // Each thread handles one or more 256-element blocks
    for (int b = tid; b < blocksPerRow; b += blockDim.x) {
        const uint8_t* block = A + row * bytesPerRow + b * 176;

        // Decode header
        float d    = __half2float(*((const __half*)(block + 0)));
        float dmin = __half2float(*((const __half*)(block + 2)));
        const uint8_t* sc = block + 4;   // 12 scale bytes
        const uint8_t* qh = block + 16;  // 32 high-bit bytes
        const uint8_t* qs = block + 48;  // 128 nibble bytes

        int xBase = b * 256;

        // 4 groups of 64 elements, each split into 2 sub-blocks of 32
        for (int g = 0; g < 4; g++) {
            int s0 = g * 2, s1 = s0 + 1;
            float sc0 = d    * q4k_scale(sc, s0);
            float mn0 = dmin * q4k_min(sc, s0);
            float sc1 = d    * q4k_scale(sc, s1);
            float mn1 = dmin * q4k_min(sc, s1);
            int hb0 = g * 2, hb1 = hb0 + 1;

            for (int l = 0; l < 32; l++) {
                int lo0 = qs[g * 32 + l] & 0x0F;
                int hi0 = (qh[l] >> hb0) & 1;
                acc += (sc0 * (lo0 | (hi0 << 4)) - mn0) * x[xBase + g * 64 + l];

                int lo1 = (qs[g * 32 + l] >> 4) & 0x0F;
                int hi1 = (qh[l] >> hb1) & 1;
                acc += (sc1 * (lo1 | (hi1 << 4)) - mn1) * x[xBase + g * 64 + 32 + l];
            }
        }
    }

    // Warp-shuffle reduction across 256 threads → one float per block
    acc = block_reduce_sum(acc);  // see below

    if (tid == 0) y[row] = acc;
}
```

### Block-sum reduction

```c
__device__ float block_reduce_sum(float val) {
    // Warp-level reduce first
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    __shared__ float smem[8];  // one slot per warp (256/32 = 8 warps)
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    if (laneId == 0) smem[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        val = (laneId < 8) ? smem[laneId] : 0.0f;
        for (int offset = 4; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xff, val, offset);
    }
    return val;
}
```

### Scale/min extraction helpers

These mirror `q4kScaleRaw` and `q4kMinRaw` from `LlamaTransformerHandler`:

```c
__device__ float q4k_scale(const uint8_t* sc, int j) {
    int v = (j < 4)
        ? sc[j] & 0x3F
        : ((sc[j + 4] & 0x0F) | ((sc[j - 4] & 0xC0) >> 2)) & 0x3F;
    return (float)v;
}

__device__ float q4k_min(const uint8_t* sc, int j) {
    int v = (j < 4)
        ? sc[j + 4] & 0x3F
        : (((sc[j + 4] & 0xFF) >> 4) | ((sc[j] & 0xC0) >> 2)) & 0x3F;
    return (float)v;
}
```

### Q6_K kernel sketch

Block layout (210 bytes / 256 elems):
`[ql:128][qh:64][sc:16][d:f16]`

```c
__global__ void q6k_sgemv_kernel(
    const uint8_t* __restrict__ A,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    int rows, int cols)
{
    const int row        = blockIdx.x;
    const int tid        = threadIdx.x;
    const int blocksPerRow = cols / 256;
    const int bytesPerRow  = blocksPerRow * 210;

    float acc = 0.0f;
    for (int b = tid; b < blocksPerRow; b += blockDim.x) {
        const uint8_t* block = A + row * bytesPerRow + b * 210;
        const uint8_t* ql = block;
        const uint8_t* qh = block + 128;
        const int8_t*  sc = (const int8_t*)(block + 192);
        float d = __half2float(*((const __half*)(block + 208)));
        int xBase = b * 256;

        for (int half = 0; half < 2; half++) {
            for (int l = 0; l < 32; l++) {
                int is  = l / 16;
                int qlL  = ql[half * 64 + l] & 0xFF;
                int qlL2 = ql[half * 64 + l + 32] & 0xFF;
                int qhL  = qh[half * 32 + l] & 0xFF;

                int q1 = ((qlL  & 0x0F) | (((qhL >> 0) & 3) << 4)) - 32;
                int q2 = ((qlL2 & 0x0F) | (((qhL >> 2) & 3) << 4)) - 32;
                int q3 = ((qlL  >>   4) | (((qhL >> 4) & 3) << 4)) - 32;
                int q4 = ((qlL2 >>   4) | (((qhL >> 6) & 3) << 4)) - 32;

                acc += d * sc[half * 8 + is]         * q1 * x[xBase + half * 128 + l];
                acc += d * sc[half * 8 + is + 2]     * q2 * x[xBase + half * 128 + l + 32];
                acc += d * sc[half * 8 + is + 4]     * q3 * x[xBase + half * 128 + l + 64];
                acc += d * sc[half * 8 + is + 6]     * q4 * x[xBase + half * 128 + l + 96];
            }
        }
    }
    acc = block_reduce_sum(acc);
    if (tid == 0) y[row] = acc;
}
```

---

## Step 3 — Building and loading the kernel in Java

JavaCPP does not provide a CUDA kernel compilation API. Two options:

**Option A (recommended): PTX embedded as a resource**
1. Write `q5k_sgemv.cu`, `q6k_sgemv.cu`, `q4k_sgemv.cu` in
   `node/src/main/resources/cuda/`.
2. Add a Maven `exec-maven-plugin` invocation that runs `nvcc --ptx` at build
   time (only on machines with NVCC; skip gracefully otherwise). Output goes to
   `target/classes/cuda/*.ptx`.
3. At runtime, `CudaQuantizedKernels` loads the PTX via the CUDA driver API
   (`cuModuleLoadData`, `cuModuleGetFunction`).

**Option B: pre-compiled .cubin per architecture**
Compile for `sm_75` (T4), `sm_86` (A10), `sm_89` (L4) and embed all three.
Select at runtime based on `cudaDeviceGetAttribute(cudaDevAttrComputeCapabilityMajor)`.
More portable for production. More build complexity.

### `CudaQuantizedKernels` skeleton

```java
public final class CudaQuantizedKernels implements AutoCloseable {

    private final GpuContext ctx;
    private CUmodule module;     // loaded PTX module
    private CUfunction q5kFn;   // kernel function handle
    private CUfunction q6kFn;
    private CUfunction q4kFn;

    public static CudaQuantizedKernels load(GpuContext ctx) {
        // cudart.cuModuleLoadData(module, ptxBytes)
        // cudart.cuModuleGetFunction(fn, module, "q5k_sgemv_kernel")
        ...
    }

    public void q5kSgemv(Pointer dA, Pointer dx, Pointer dy, int rows, int cols) {
        // cudaLaunchKernel(q5kFn, gridDim=(rows,1,1), blockDim=(256,1,1), args...)
    }

    public void q6kSgemv(Pointer dA, Pointer dx, Pointer dy, int rows, int cols) { ... }
    public void q4kSgemv(Pointer dA, Pointer dx, Pointer dy, int rows, int cols) { ... }

    @Override
    public void close() {
        // cudart.cuModuleUnload(module)
    }
}
```

JavaCPP CUDA driver API classes: `org.bytedeco.cuda.global.cuda` (not `cudart` —
the driver API lives in a different namespace). Use `CUmodule`, `CUfunction`,
`CUstream` from `org.bytedeco.cuda.CUmod_st`, etc.

---

## Step 4 — `CudaMatVec.sgemv(DeviceQuantizedMatrix, float[])`

Add this overload to `CudaMatVec`:

```java
public float[] sgemv(DeviceQuantizedMatrix A, float[] x) {
    int rows = A.rows(), cols = A.cols();

    MatVecEvent evt = new MatVecEvent();
    evt.begin();

    float[] y     = new float[rows];
    long bytesX   = (long) cols * 4;
    long bytesY   = (long) rows * 4;

    PointerPointer pX = new PointerPointer(1);
    PointerPointer pY = new PointerPointer(1);
    try {
        checkCuda(cudart.cudaSetDevice(ctx.deviceIndex()), "cudaSetDevice");
        checkCuda(cudart.cudaMalloc(pX, bytesX), "cudaMalloc(x)");
        checkCuda(cudart.cudaMalloc(pY, bytesY), "cudaMalloc(y)");

        Pointer dx = pX.get(0), dy = pY.get(0);
        try (FloatPointer hx = new FloatPointer(x)) {
            checkCuda(cudart.cudaMemcpy(dx, hx, bytesX, H2D), "cudaMemcpy(x)");
        }

        switch (A.ggmlType()) {
            case 13 -> kernels.q5kSgemv(A.devicePointer(), dx, dy, rows, cols);
            case 14 -> kernels.q6kSgemv(A.devicePointer(), dx, dy, rows, cols);
            case 12 -> kernels.q4kSgemv(A.devicePointer(), dx, dy, rows, cols);
            default -> throw new UnsupportedOperationException(
                "No GPU kernel for GGML type " + A.ggmlType());
        }

        try (FloatPointer hy = new FloatPointer(y)) {
            checkCuda(cudart.cudaMemcpy(hy, dy, bytesY, D2H), "cudaMemcpy(y)");
            hy.get(y);
        }

        evt.backend = "cuda-quant-" + ggmlLabel(A.ggmlType());
        evt.rows = rows;
        evt.cols = cols;
        return y;
    } finally {
        cudart.cudaFree(pX.get(0));
        cudart.cudaFree(pY.get(0));
        pX.close(); pY.close();
        evt.commit();
    }
}
```

`CudaMatVec` constructor now takes an optional `CudaQuantizedKernels kernels`
(null = quantized GPU path disabled). `GpuContext.init()` should create it.

---

## Step 5 — Wiring into `LlamaTransformerHandler`

At load time, when the backend is `CudaMatVec`, upload all weight tensors:

```java
// In the GgufReader constructor path:
if (backend instanceof CudaMatVec cuda) {
    this.deviceWeights = new HashMap<>();
    for (int li = 0; li < L; li++) {
        deviceWeights.put("wq." + li,    DeviceQuantizedMatrix.upload(cuda.ctx(), wq[li],    H, H));
        deviceWeights.put("wk." + li,    DeviceQuantizedMatrix.upload(cuda.ctx(), wk[li],    kvDim, H));
        // ... all 7 projection weights per layer
    }
}
```

Then `matVec(QuantizedTensor A, ...)` becomes:

```java
static float[] matVec(QuantizedTensor A, float[] x, int rows, int cols,
                       MatVec backend,
                       Map<String, DeviceQuantizedMatrix> deviceWeights) {
    if (backend instanceof CudaMatVec cuda && deviceWeights.containsKey(A.name())) {
        return cuda.sgemv(deviceWeights.get(A.name()), x);
    }
    // fall back to CPU quantized path
    return matVec(A, x, rows, cols);
}
```

The `QuantizedTensor.name()` field (already present in the record) is the lookup
key: `"blk.0.attn_q.weight"`, `"blk.0.ffn_gate.weight"`, etc. These are the
exact strings read from GGUF metadata, so they serve as stable identifiers.

---

## Step 6 — `MatVec` interface addition

Add the new overload with a default that throws `UnsupportedOperationException`,
matching the existing `DeviceFloatMatrix` pattern:

```java
default float[] sgemv(DeviceQuantizedMatrix A, float[] x) {
    throw new UnsupportedOperationException(
        "quantized device-resident weights not supported by this MatVec backend");
}
```

---

## Performance expectations

On a T4 (75 GB/s memory bandwidth, TinyLlama 2048-dim):

- Current CPU path per matVec: ~1.3 ms (includes CPU dequant + H2D copy of 16 MB)
- H2D weight copy alone at 10 GB/s PCIe: ~1.6 ms per 2048×2048 F32 matrix
- GPU dequant+SGEMV from VRAM at 75 GB/s: ~0.06 ms (176 bytes/256 elems × 8
  blocks × 2048 rows = 11 MB Q5_K → reads ~22 MB with x[] = ~0.3 ms at full BW)

The primary win is eliminating PCIe. Even with kernel launch overhead, a 4–8×
decode speedup is plausible once all weights are VRAM-resident and dequantized
on-GPU.

The x-vector H2D transfer (2048 floats = 8 KB) and y-vector D2H (8 KB) remain
per-call; at 10 GB/s PCIe those are ~1.6 µs each — negligible.

---

## Test plan

**Unit tests** (no GPU required — mock the kernel call):
- `DeviceQuantizedMatrix` uploads and releases without crash (CUDA required, skip
  otherwise — same pattern as `CudaMatVecBackendTest`)
- Kernel output matches `LlamaTransformerHandler.matVec(QuantizedTensor, ...)` for
  each GGML type, same random blocks used in `PhiQuantizedMatVecTest`

**Integration test** (`@Tag("gpu")` — requires CUDA):
- `q5k_sgemv_kernel` output matches CPU Q5_K path within 1e-3 for 2048×2048
- `q6k_sgemv_kernel` output matches CPU Q6_K path within 1e-3
- Full forward pass through `LlamaTransformerHandler` with `CudaMatVec` backend
  produces logits within 1e-2 of the CPU path (numerical drift from F32↔F16
  in kernel headers is expected)
- JFR `juno.MatVec` events show `backend = "cuda-quant-q5k"` during inference

**Existing tests must not regress**:
- `CudaMatVecBackendTest` (GPU float32 path) — unaffected
- `PhiQuantizedMatVecTest` — CPU path only, unaffected
- `QuantizedSimdMatVecTest` — CPU path only, unaffected
- `MatVecBackendContractTest` — CPU path only, unaffected

---

## Files to create

```
node/src/main/java/cab/ml/juno/node/DeviceQuantizedMatrix.java   (new)
node/src/main/java/cab/ml/juno/node/CudaQuantizedKernels.java    (new)
node/src/main/resources/cuda/q5k_sgemv.cu                        (new)
node/src/main/resources/cuda/q4k_sgemv.cu                        (new)
node/src/main/resources/cuda/q6k_sgemv.cu                        (new)
```

Files to modify:

```
node/src/main/java/cab/ml/juno/node/MatVec.java
  + default sgemv(DeviceQuantizedMatrix, float[]) overload

node/src/main/java/cab/ml/juno/node/CudaMatVec.java
  + CudaQuantizedKernels field
  + sgemv(DeviceQuantizedMatrix, float[]) implementation
  + ggmlLabel() helper

node/src/main/java/cab/ml/juno/node/GpuContext.java
  + CudaQuantizedKernels creation in init()
  + kernels() accessor
  + kernels.close() in close()

node/src/main/java/cab/ml/juno/node/LlamaTransformerHandler.java
  + Map<String, DeviceQuantizedMatrix> deviceWeights field
  + upload loop in constructor (when backend instanceof CudaMatVec)
  + matVec() dispatch updated to check deviceWeights

node/src/main/java/cab/ml/juno/node/MatVecEvent.java
  (no change — new backend label strings are sufficient)
```

---

## Known pitfalls

**`__half2float` requires `cuda_fp16.h`** — include it explicitly in every `.cu`
file. On older CUDA toolkits `__half` is not available in device code without it.

**Block-sum reduction must handle rows < blockDim.x** — when `blocksPerRow <
256`, many threads do zero work. The reduction still produces the correct sum
because idle threads contribute 0.0f. Verify with a 1-row, 256-col test.

**`cudaSetDevice` before kernel launch** — same rule as in `CudaMatVec`. The
device is thread-local in CUDA. `CudaQuantizedKernels.q5kSgemv()` must call
`cudaSetDevice(ctx.deviceIndex())` before the launch.

**`QuantizedTensor.name()` as a lookup key** — the name comes from GGUF metadata
and is stable across model loads of the same file. It does NOT include the
`.weight` suffix consistently — verify with a log print at upload time before
relying on it as a key. Alternatively, key by tensor object identity
(`IdentityHashMap<QuantizedTensor, DeviceQuantizedMatrix>`).

**VRAM budget** — TinyLlama 22 layers × 7 weights × Q5_K avg ~11 MB =
~1.7 GB raw. At Q5_K this is well within a 4 GB T4. Larger models (Llama 3 8B)
at Q5_K ≈ 5 GB — verify VRAM budget before upload via
`cudaMemGetInfo(&free, &total)` and log it prominently.

**`cublasSetPointerMode_v2` in `CudaMatVec`** — move this call to
`GpuContext.init()` as noted in `perf-rep2.md`. It should not be called inside
the hot path; it is a persistent handle property.
