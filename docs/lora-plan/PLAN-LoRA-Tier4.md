# Tier 4: GPU LoRA Training

## Agent handoff

Read and follow `models/CLAUDE.md`, `PLAN-LoRA-ROADMAP.md`, and Tiers 1–3 before implementing. Also read the GPU bindings/backends, `LoraTrainableHandler`, tests, scripts, and GPU documentation.

Do not begin by moving tiny standalone LoRA GEMVs to the GPU. Without resident frozen transpose backward and batching, transfer and launch overhead can make training slower.

## Overview

Juno currently performs the frozen forward projection on GPU when device-resident weights fit, but adapter math, frozen transpose backward, gradients, clipping, and Adam remain on CPU. This tier turns that hybrid path into measured GPU training through resident transpose backward, microbatching, and selective device residency.

## Scope

Goals:

1. Reuse resident frozen matrices for GPU `W^T * g`.
2. Amortize host/device transfer through microbatched linears.
3. Keep activations and adapters resident only where benchmarks justify it.
4. Preserve CUDA and ROCm behavior through vendor-neutral contracts.
5. Keep the Tier 1–3 CPU path as correctness oracle and fallback.

Initial non-goals:

- Tensor-parallel LoRA training.
- Full custom GPU attention, RoPE, softmax, or loss kernels.
- GPU DoRA norm refresh.
- Replacing GGUF storage with a different training format.
- Calling the current forward-only hybrid path “GPU training.”

## 1. Establish the baseline

- Instrument `LoraTrainEvent` to separate frozen forward, attention/nonlinear, frozen transpose backward, adapter backward, transfer, and optimizer time.
- Benchmark CPU and current GPU-forward/CPU-backward paths on TinyLlama, sequence lengths 64/128, rank 8, qv and all-linear.
- Record tokens/s, forward/backward/optimizer time, H2D/D2H bytes, peak heap, and peak VRAM.
- Keep this baseline in `docs/performance.md`; do not claim production GPU training yet.

## 2. Add vendor-neutral resident transpose backward

Resident matrices are row-major `W[rows,cols]`.

- Forward `W*x` uses BLAS transpose mode because the device interprets row-major storage as column-major transpose.
- Backward `W^T*g` must use BLAS no-transpose mode with the same resident matrix.

Update:

- `GpuBindings.java`: add `opNoTranspose()` and transpose-capable BLAS contracts.
- `CudaBindings.java`: bind CUDA no-transpose and any required SGEMV/GEMM symbols.
- `RocmBindings.java`: bind ROCm equivalents.
- `GpuMatVec.java`: add `sgemvTranspose(DeviceFloatMatrix, float[])` and `sgemvTranspose(DeviceHalfMatrix, float[])`.
- `CudaMatVec.java` and `RocmMatVec.java`: implement resident FP32/FP16 transpose.
- `LoraTrainableHandler.java`: route all frozen transpose operations, including output projection, through resident GPU matrices when available; retain quantized CPU transpose fallback.

Constants:

- CUDA `CUBLAS_OP_N = 0`, `CUBLAS_OP_T = 1`.
- ROCm `rocblas_operation_none = 111`, `rocblas_operation_transpose = 112`.

Handle hardware limitations:

- Probe FP16 resident transpose support.
- If FP16 HSS transpose is unavailable, use FP32-resident transpose.
- Align `LoraTrainableHandler` with `supportsHalfResident()` behavior already used by other handlers.
- On allocation failure, close partial buffers and fall back to the CPU quantized path.

## 3. Test transpose correctness before handler integration

Add GPU-group tests:

- Operation constants for CUDA and ROCm.
- Dense FP32 and FP16 adjoint identity: `dot(W*x,g) == dot(x,W^T*g)`.
- GPU transpose versus CPU dense and quantized-decode references.
- Rectangular matrices covering Q/K/V/O/FFN dimensions.
- Invalid dimensions, closed buffers, unsupported half kernels, and fallback.

Required tolerance:

- FP32 maximum relative/absolute error defined by existing backend tests, targeting `1e-4`.
- FP16 mixed path targeting `2e-3`, adjusted only with documented hardware evidence.

Do not integrate or market the path if the adjoint gate fails.

## 4. Milestone 1: GPU frozen backward

- Use resident transpose for `wq,wk,wv,wo,wgate,wup,wdown` and output projection.
- Keep `LayerState`, attention, LoRA A/B, gradients, clipping, and Adam on host.
- Add backend labels and JFR timings distinguishing resident forward and resident transpose backward.
- Add `--lora-train-device auto|gpu|cpu` and `LORA_TRAIN_DEVICE`; `gpu` fails if required GPU primitives are unavailable, while `auto` may fall back.

Milestone gate:

- CPU/GPU loss and adapter gradients agree within declared tolerance.
- GPU backward is at least 2× faster than CPU backward.
- End-to-end training is at least 1.5× faster on the reference GPU.
- CUDA and ROCm both pass correctness; do not merge a CUDA-only public training feature.

## 5. Add microbatched frozen linear algebra

Prefer the lowest-risk supported path:

1. Extend existing HSS strided-batched GEMV to `batchCount > 1`, or
2. Bind FP32 `cublasSgemm_v2` and `rocblas_sgemm`.

Do not depend on mixed-dtype GEMMEx as the primary design; existing project history reports unsupported/invalid behavior on common stacks.

Add:

- `GpuBlasOps.java` for explicit batched forward/transpose contracts.
- `DeviceActivationBatch.java` for reusable device scratch.
- Batch layout documentation and CPU reference operations.
- Tier 1 integration: microbatch inside an accumulation group; normalization remains by total prediction count.

Pack compatible token-position inputs and perform fewer launches per projection. Keep attention/nonlinear work on host initially if moving it would require custom kernels. Measure actual transfer reduction rather than assuming GEMM is faster.

Milestone gate:

- Batched loss and summed gradients agree with sequential CPU execution within tolerance.
- Batch 8 is at least 2× faster than Milestone 1 or at least 4× faster than CPU.
- H2D/D2H bytes per token decline as batch size increases.
- Document maximum batch size and VRAM for TinyLlama and one 7B model.

## 6. Add device-resident adapter math selectively

Add `DeviceLoraAdapter.java` or `LoraDeviceState.java` under the node module:

- Upload A/B once, preferably FP32 initially.
- Use batched GEMM for `H=A*X` and `delta=B*H`.
- Compute batched `gradA`, `gradB`, and LoRA input gradients on device.
- Preserve Tier 2 dropout masks and LoRA+ A/B learning-rate semantics.
- Preserve Tier 3 standard/rsLoRA scaling and DoRA coefficient application.

Do not enable this path merely because it works:

- At batch 1, host adapter math must not regress wall time by more than 10%; otherwise keep it on host.
- Enable GPU adapter math only above a measured batch/rank/dimension threshold.
- Never upload A/B for each call.

## 7. Complete residency only if synchronization dominates

Initially download accumulated adapter gradients once per optimizer update and use the Tier 2 host optimizer. This may be optimal because adapter state is small.

Only add `LoraAdamOptimizerGpu.java` if profiling shows gradient transfer or host updates are material:

- Keep A/B, gradients, moments, clipping, and updates resident.
- Match host global-norm, AdamW, LoRA+, schedule, and magnitude-group semantics.
- Synchronize once per optimizer update, not once per projection.
- Keep checkpoint serialization deterministic by downloading parameters at save time.

## 8. Fallback and safety policy

Ordered supported paths:

1. CPU quantized full path — correctness oracle.
2. GPU frozen forward plus CPU transpose — compatibility fallback, not advertised as full GPU training.
3. GPU frozen forward plus FP32 resident transpose — FP16 transpose fallback.
4. GPU batched frozen linears plus host adapters — valid when adapter GPU intensity is insufficient.
5. Fully resident adapter path — only after performance gates.

Fail closed on operation-layout or adjoint mismatch. Never silently continue with numerically incorrect transpose mapping.

Explicitly reject tensor-parallel adapter training and playback until shard-aware adapter slicing/collectives are designed.

## 9. Tests first

New/extended tests:

- GPU operation constants and binding delegation.
- `CudaMatVecTransposeTest` and `RocmMatVecTransposeTest`.
- FP32 GEMM/batched-GEMV parity tests.
- `LoraTrainableHandlerGpuBackwardTest`: CPU/GPU loss, A/B gradients, clipping input, and updates.
- `DeviceLoraAdapterTest`: batched forward/backward versus host `LoraAdapter`.
- Dropout determinism across CPU/GPU.
- LoRA+ ratio and rsLoRA scale parity.
- DoRA coefficient-scaled gradients when DoRA GPU integration is enabled.
- OOM, unsupported FP16, closed-resource, and CPU-fallback tests.
- Gated TinyLlama integration: loss decreases; save/playback works; checkpoint values remain within declared drift tolerance.

## 10. Expected files

Likely modified:

- `node/src/main/java/cab/ml/juno/node/GpuBindings.java`
- `node/src/main/java/cab/ml/juno/node/CudaBindings.java`
- `node/src/main/java/cab/ml/juno/node/RocmBindings.java`
- `node/src/main/java/cab/ml/juno/node/GpuMatVec.java`
- `node/src/main/java/cab/ml/juno/node/CudaMatVec.java`
- `node/src/main/java/cab/ml/juno/node/RocmMatVec.java`
- `node/src/main/java/cab/ml/juno/node/MatVecBackend.java`
- `node/src/main/java/cab/ml/juno/node/LoraTrainableHandler.java`
- `node/src/main/java/cab/ml/juno/node/LoraTrainEvent.java`
- `juno-player/src/main/java/cab/ml/juno/player/LoraTrainingConfig.java`
- `juno-player/src/main/java/cab/ml/juno/player/ConsoleMain.java`
- `scripts/run.sh`, `scripts/run.bat`

Likely new:

- `node/src/main/java/cab/ml/juno/node/GpuBlasOps.java`
- `node/src/main/java/cab/ml/juno/node/DeviceActivationBatch.java`
- `node/src/main/java/cab/ml/juno/node/DeviceLoraAdapter.java`
- Optional `node/src/main/java/cab/ml/juno/node/LoraAdamOptimizerGpu.java`
- Corresponding GPU tests and benchmark harness.

Update `README.md`, `CHANGELOG.md`, `RELEASE_NOTES.md`, `docs/LoRA.md`, `docs/howto.md`, `docs/features.md`, `docs/arch.md`, `docs/agent-arch.txt`, and `docs/performance.md`.

## Verification and product gate

Reference benchmark:

- TinyLlama Q4_K_M.
- Sequence length 128.
- Rank 8.
- qv and all-linear scenarios.
- 10 warm-up and at least 20 measured updates.
- NVIDIA and AMD reference hardware.

Before documentation says “GPU LoRA training”:

1. Resident transpose correctness is green on CUDA and ROCm.
2. CPU/GPU gradients and updates agree within declared tolerances.
3. GPU backward is at least 2× faster than CPU backward.
4. End-to-end tokens/s shows a meaningful gain.
5. Peak VRAM and fallback behavior are documented.
6. Device adapter math is disabled where it is transfer-bound.

## Implementation todos

1. Add transpose/binding tests and baseline instrumentation.
2. Implement CUDA/ROCm resident transpose with FP32 fallback.
3. Integrate GPU frozen backward and pass Milestone 1 gates.
4. Add batched frozen linears and reusable activation buffers.
5. Add device adapter math only after intensity benchmarks.
6. Add device optimizer only if profiling justifies it.
7. Wire CLI/JFR/docs and publish benchmark evidence.
