(ch-1-1)=
# 1.1. Requirements

- JDK 25+
- Maven 3.9+
- GPU nodes (optional): CUDA 12.x with an NVIDIA driver, or ROCm 6+ with an AMD driver.
  CPU-only inference requires neither.

**Windows:** `juno.bat` at the project root delegates to `scripts\run.bat`. Requires JDK 25+ on
`PATH` or `JAVA_HOME` set. CUDA GPU acceleration is supported (NVIDIA only; ROCm is Linux-only).
All flags and environment overrides in the [CLI reference](#ch-3-2) apply
equally on Windows. See [Windows notes](#ch-6-3) for the full list of
platform differences.

## Stack

Node coordination and inference RPCs use gRPC with protobuf contracts from the `api` module. GPU
matmul is backed by Panama FFI (`java.lang.foreign`) against two vendor libraries:

- **NVIDIA:** CUDA 12.x + cuBLAS. `CudaBindings` resolves `libcudart.so.12` and
  `libcublas.so.12`; `CudaMatVec` owns all device memory and stream lifecycle.
- **AMD:** ROCm 6+ + rocBLAS. `RocmBindings` resolves `libamdhip64.so` and `librocblas.so`;
  `RocmMatVec` mirrors the same device-resident FP32/FP16 paths.

Backend is auto-selected at startup: CUDA first, then ROCm, then CPU. Override with
`-Djuno.gpu.backend=cuda|rocm|auto`. A CPU quantized path is used when GPU is off or
unavailable. The coordinator HTTP surface (REST and SSE) is implemented with Javalin.

## See also

- [Chapter 1.2 -- Quickstart: Local Player](#ch-1-2)
- [Chapter 2.4 -- GPU Acceleration](#ch-2-4)
- [Chapter 1.4 -- Supported Models](#ch-1-4)

---

[Table of Contents](../index.md) &nbsp;|&nbsp; [1.2 Quickstart: Local Player ->](#ch-1-2)
