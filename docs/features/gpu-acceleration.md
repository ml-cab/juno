# GPU acceleration

CUDA 12.x drives matrix work through cuBLAS-backed paths (`CudaMatVec`) using ByteDeko/JavaCPP presets; weights upload as FP16 on GPU load with deterministic release on shard unload. Pass `--cpu` or `JUNO_USE_GPU=false` to force CPU quantised matmul; cluster coordinators stay CPU-only while each node JVM owns its GPU context.

Lifecycle and handler routing are described under GPU sections of [arch.md](../arch.md). CPU vs GPU throughput snapshots appear in [juno_test_matrix.html](../juno_test_matrix.html).
