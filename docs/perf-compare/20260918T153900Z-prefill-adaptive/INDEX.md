# Prefill adaptive chunk sizing — 20260918T153900Z

Model: `mistral-7b-instruct-v0.1-q4_k_m.gguf` · prompt: 488 tokens (real chat prompt, not raw-prompt
padding) · backend: GPU (`--gpu-layers auto --mmq auto --schedule static`) · `max_tokens=16`,
temperature 0.

Gathered manually against a live local API instance (`compare-prefill-batch.sh` always passes
`--prefill-batch` explicitly and cannot exercise the new no-flag adaptive default) — same real-request
methodology as `PLAN-Infra-Tier20.md`'s own Finding 4 A/B. Real GTX 1080, JFR `--jfr 10m`.

| `--prefill-batch` | resolved chunk | `PrefillBatch` calls | prefill total (JFR) | MatVec calls | Attention calls | request wall |
|---:|---:|---:|---:|---:|---:|---:|
| `32` (explicit, old fixed default) | 32 | 16 | 15710 ms | 7008 | 1024 | 16971 ms |
| *(none — new adaptive default)* | 24889 | 1 | 11000 ms | 2289 | 544 | 12354 ms |

Speedup (adaptive over fixed-32): **prefill -30.0%** (1.428x), **request wall -27.2%** (1.374x),
MatVec call count **-67.3%** (3.06x fewer launches), Attention call count **-46.9%** (1.88x fewer).

Adaptive chunk size derivation: free VRAM queried live via `GpuContext.freeVramBytes()`
(`cudaMemGetInfo`) after model residency + KV pool allocation = 3111 MiB;
`24889 = floor(3111 MiB * 0.5 / 65536 bytes-per-token)`, comfortably covering the whole 488-token
prompt in one window (ceiling 65536, floor 32 — the adaptive path can only grow the chunk relative to
today's fixed default, never shrink it).

Correctness: response content was coherent and well-formed on both runs (no functional regression);
`GgufReader`/handler unit-level chunk-boundary numeric parity is covered generically (any chunk size
vs whole-window) by the pre-existing
`node/src/test/java/cab/ml/juno/node/LlamaTransformerHandlerPrefillChunkParityTest.java`, which already
asserts identical logits across chunk sizes 1, 32, and whole-window — the adaptive resolver only picks
a different (larger) fixed value fed through the same `PrefillChunker`/`prefillBatch` call path, so no
new numeric behavior is introduced.

## Pinned host-staging memory (Phase A step 1)

`CudaMatVec`'s `Fp16Scratch`-backed batched-GEMM paths (`sgemmHalfBatched`, `sgemmHalfBatchedGemm`,
`sgemmQ4KBatchedGemm`) and `CudaRmsNorm.normalizeBatch` now stage H2D/D2H buffers through
`GpuBindings.hostMalloc`/`hostFree` (`cudaMallocHost`/`cudaFreeHost`, vendor-neutral —
`hipHostMalloc`/`hipHostFree` on ROCm) instead of plain `Arena.ofConfined()`, grown-and-kept-max the
same way the existing device-side scratch already is.

**Note on `nsys` verification:** `PLAN-Infra-Tier20.md`'s exit gate calls for the same `nsys` timeline
methodology used to originally find the >10x `cudaMemcpyAsync` host-time gap (Finding 3). In this
session's environment `nsys profile` fails on every invocation, including the most minimal
(`nsys profile -- echo hi`), with `option is ambiguous` against its own internal default argument set
— reproduced with bare `-o`/`--force-overwrite` and with no options at all, ruling out anything in the
Juno command line. This is an environment/installation defect in the `nsys` 2022.4.2.50 build present
here, not something this change can work around from application code. Correctness (parity tests, live
smoke test) and the end-to-end wall-clock win above are relied on instead; a future session with a
working `nsys` install should re-run the Finding 3 methodology directly for a quantitative
`cudaMemcpyAsync`-collapse number.

## Phase B checkpoint — GPU-resident Rope/SwiGlu at prefill scale (re-measured under pinned memory)

Re-ran `PLAN-Infra-Tier20.md` Finding 6's throwaway microbenchmark (not checked in, same as the
original — `CudaRmsNorm.normalizeBatch` as a round-trip-shape proxy, batch=136) now that its staging
buffers are pinned host memory instead of `Arena.ofConfined()`:

| Shape | CPU scalar (whole-batch, per-row loop) | GPU round trip (pinned memory) | GPU/CPU |
|---|---:|---:|---:|
| batch=136, dim=4096 | 0.697 ms/call | 1.519 ms/call | 2.18x slower |
| batch=136, dim=14336 | 2.948 ms/call | 4.614 ms/call | 1.56x slower |

**No-go.** Pinned memory did not close the gap Finding 6 flagged as inconclusive — the GPU round trip
is still slower than CPU scalar at prefill batch size, by a similar or slightly wider margin than the
original (confounded) 1.30-1.32x measurement. This confirms Tier 19's original diagnosis was the real
bottleneck all along: a single ad-hoc GPU launch (kernel + H2D/D2H + sync) has an irreducible per-call
cost that pinned memory's faster DMA does not amortize away, because there is still exactly one launch
per op with no activation-residency chain to spread that cost across. Per the tier plan, this closes
the avenue rather than proceeding to build `RopeKernel`/`SwiGluKernel` on an unfavorable number — the
next legitimate lever for GPU-resident elementwise ops remains the bigger redesign Tier 19 already
named (device-pointer-accepting `sgemv`/`sgemm` overloads enabling a real multi-op chained graph), not
a per-op port.

## Regression gates

- `compare-llama-cpp.sh --gpu --no-tuned-lane` (default 4-model set): `failures=0` —
  [`20260918T204809Z`](../20260918T204809Z/).
- `compare-lora.sh --gpu --baseline release-0.1.2`: **ok** — train_total_ms ratio=1.00, ms_per_pass
  ratio=1.00, playback_tps ratio=0.87 (≥0.80 gate) —
  [`20260918T204959Z-lora`](../20260918T204959Z-lora/).
- `compare-vision.sh`: reasoned skip — Phi-2's vision batched prefill uses its own CPU
  weight-stationary kernels, never `CudaMatVec`'s batched-GEMM paths or `CudaRmsNorm` (same
  precedent Tier 17/19 already found and re-verified here by code inspection, not assumed).
