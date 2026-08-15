(ch-2-4)=
# 2.4. GPU Acceleration

Juno supports two GPU backends through Panama FFI (`java.lang.foreign.Linker` and
`SymbolLookup`). JavaCPP/bytedeco is not used anywhere in the codebase. This page is the
authoritative description of GPU support; the feature overview links here instead of repeating
it.

## Backends

**NVIDIA (CUDA 12.x / cuBLAS).** `CudaBindings` resolves `libcudart.so.12` and
`libcublas.so.12`. `CudaMatVec` provides an FP32 host path and device-resident FP32/FP16 paths
via `cublasSgemv_v2` and `cublasHSSgemvStridedBatched`. Weights upload once as `DeviceHalfMatrix`
at load time, with deterministic release on shard unload.

**AMD (ROCm 6+ / rocBLAS).** `RocmBindings` resolves `libamdhip64.so` and `librocblas.so`.
`RocmMatVec` provides the same three compute paths via `rocblas_sgemv` and
`rocblas_hssgemv_strided_batched`. Tested on AMD Radeon RX 7900 XT (gfx1100, ROCm 7.2.x).

Both backends implement the sealed `GpuMatVec` interface. Transformer handlers
(`LlamaTransformerHandler`, `Phi3TransformerHandler`, `Qwen3TransformerHandler`,
`Qwen3MoeTransformerHandler`, `LoraTrainableHandler`) depend on `GpuMatVec`, not a concrete
vendor class, so device-resident weights upload the same way on any GPU vendor. Phi-3 is
supported on GPU; Gemma, Qwen 2, Qwen3, and Qwen3.5 GPU inference paths are under development.
See [Handler routing](#ch-2-3) for the full architecture support matrix.

## Backend selection

```{mermaid}
flowchart TD
    Start["JVM startup"]
    CheckCUDA{{"CUDA available? (CudaAvailability.isAvailable())"}}
    CheckROCm{{"ROCm available? (RocmAvailability.isAvailable())"}}
    UseCUDA["Use CudaMatVec (CudaBindings → libcudart.so.12\n+ libcublas.so.12)"]
    UseROCm["Use RocmMatVec (RocmBindings → libamdhip64.so\n+ librocblas.so)"]
    UseCPU["Use CpuMatVec (parallel IntStream, quantized)"]
    Override["juno.gpu.backend=cuda|rocm|auto OR JUNO_USE_GPU=false OR --cpu"]

    Start --> Override
    Override --> CheckCUDA
    CheckCUDA -->|"Yes"| UseCUDA
    CheckCUDA -->|"No"| CheckROCm
    CheckROCm -->|"Yes"| UseROCm
    CheckROCm -->|"No"| UseCPU
```

Backend selection is automatic via `selectBindings()` in `GpuContext`: CUDA first, then ROCm,
then CPU. Override with `-Djuno.gpu.backend=cuda|rocm|auto`. `selectBackend()` in
`ForwardPassHandlerLoader` reads `JUNO_USE_GPU` and `-Djuno.cuda.device` (defaults to `0`). Pass
`--cpu` or `JUNO_USE_GPU=false` to force CPU quantized matmul. Cluster coordinators always stay
CPU-only; each node JVM owns its own GPU context. All CUDA/HIP symbols are accessed through
`GpuBindings`.

## Design decisions

**Panama FFI instead of JavaCPP/bytedeco.** `GpuBindings` is a vendor-neutral interface resolved
at class-init via `java.lang.foreign.Linker` and `SymbolLookup`. The resulting `MethodHandle`
instances are thread-safe and carry zero per-call Java overhead (the JIT eliminates argument
boxing for typed `invokeExact` call sites). The `bytedeco/cuda-platform` Maven dependency and its
generated JNI wrappers are removed entirely. The only requirement is
`--enable-native-access=ALL-UNNAMED` on the JVM command line, injected automatically by
`node/pom.xml` surefire config and by all launcher scripts.

**Lazy dequantization on CPU, eager upload on GPU.** On the CPU path, dequantization runs one
256-element block at a time inside the matmul loop, keeping peak live float footprint around
1 kB instead of around 65 MB. On the GPU path, Llama and Phi-3 dequantize once on load and
upload to `DeviceHalfMatrix` (FP16 on device) via `GpuMatVec.uploadHalf()`. If `cudaMalloc` or
`hipMalloc` fails, both handlers close partial GPU buffers and fall back to CPU quantized matmul
for those projections.

**Explicit GPU weight lifecycle.** `ForwardPassHandler.releaseGpuResources()` closes all
`DeviceHalfMatrix` / `DeviceFloatMatrix` buffers. `EmbeddedNodeServer` calls it on shard unload,
reload, and handler swap, so VRAM is freed without waiting for garbage collection.

## See also

- [Chapter 2.3 -- Handler Routing](#ch-2-3)
- [Chapter 2.5 -- Key Design Decisions](#ch-2-5)
- [Chapter 7.3 -- Performance Report](#ch-7-3)

---

[<- 2.3 Handler Routing](#ch-2-3) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [2.5 Key Design Decisions ->](#ch-2-5)
