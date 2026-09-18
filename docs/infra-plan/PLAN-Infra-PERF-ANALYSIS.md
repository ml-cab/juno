# Infra Performance Analysis — llama.cpp Bake-off (2026-08-31)

Authoritative artifact: [`docs/perf-compare/README.md`](../perf-compare/README.md) and run dirs `20260831T230258Z` (CPU) / `20260831T231403Z` (GPU).

Execution schedule: [`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) **phases P0–P5** (tier number ≠ execution order).

Host: **medion-Precision-T3610** · Xeon E5-1650 v2 (12 threads) · 62.7 GiB RAM · **NVIDIA GTX 1080 (8 GiB)**.

Workload: `n_prompt=128`, `n_gen=64`, temperature 0, Juno `--vector 0` (scalar), JFR metrics (`--jfr 30m`).

## Executive summary

| Gap class | Juno vs llama.cpp (GPU, tg) | Root cause (JFR) | Highest-leverage tiers |
|-----------|------------------------------|------------------|------------------------|
| Single-stream decode | **0.17–0.22×** (models that fit VRAM) | `juno.MatVec` ≈ **93–96%** of decode `ForwardPass` | **Tier 13** (fused quant / MMQ), vector SIMD track |
| Prompt eval (pp) | **0.005–0.011×** | No prefill JFR events; API path ~21 tok vs bench 128; sequential prefill | **Tier 8**, fix compare parity, instrument prefill JFR |
| 7B on 8 GiB | **0.01×** tg (CPU fallback) | 100% `MatVec.backend.cpu` (OOM / no partial offload) | **Tier 5** (`--gpu-layers`) |
| Multi-session throughput | Not measured yet | `BatchConfig.disabled()` in production | **Tier 1**, then **Tiers 14–16** (P1) |
| Memory fit (N contexts) | Not measured yet | Dense `float[][]` KV to `MAX_SEQ_LEN` | **Tier 6**, **Tier 14** |

**Strategic conclusion:** Phase P1 (Tiers 14–16) raises aggregate QPS under concurrent load but does **not** close the ~4–6× single-stream decode gap. That gap is MatVec-dominated and requires Phase P0 (kernel path) before peer claims on interactive latency.

## Measured baselines

### GPU decode (JFR tg t/s) — `20260831T231403Z`

| Model | llama.cpp tg | Juno tg | Ratio | MatVec backend | MatVec % decode |
|-------|-------------:|--------:|------:|----------------|----------------:|
| tinyllama-1.1b Q4_K_M | 186 | 31.4 | 0.17 | cuda_resident_fp16 | 96% |
| qwen2.5-3b Q4_K_M | 68.0 | 13.3 | 0.19 | cuda_resident_fp16 | 93% |
| Phi-3.5-mini Q4_K_M | 57.8 | 12.6 | 0.22 | cuda_resident_fp16 | 93% |
| mistral-7b Q4_K_M | 35.2 | 0.48 | 0.01 | **cpu only** | 99% |

Per-token decode (Phi-3.5 GPU): `ForwardPass.decode.p95` ≈ **84 ms** vs llama ≈ **17 ms** (~4.9×).

### CPU decode (JFR tg t/s) — `20260831T230258Z`

| Model | llama.cpp tg | Juno tg | Ratio |
|-------|-------------:|--------:|------:|
| tinyllama-1.1b Q4_K_M | 0.61* | 3.12 | — |
| qwen2.5-3b Q4_K_M | 3.80 | 1.01 | 0.27 |
| Phi-3.5-mini Q4_K_M | 3.54 | 0.84 | 0.24 |
| mistral-7b Q4_K_M | 2.18 | 0.48 | 0.22 |

\* TinyLlama llama.cpp tg is a single-rep outlier; prior baseline ~6.8 t/s.

CPU Phi-3.5: MatVec **100% cpu**, p95 **11.4 ms**/call, decode p95 **1338 ms**/token.

### Prefill (pp) — methodology caveat

llama-bench evaluates **128 raw prompt tokens**. Juno API chat path tokenizes the same *text* but applies a chat template → **~21 prompt tokens** (Phi-3.5). Reported Juno pp uses `(API latency − decode total_ms) / prompt_tokens` and is **not directly comparable** to llama-bench pp until:

1. Compare script adds a **raw / non-template** prompt mode with matched token count, or
2. llama-bench runs the templated prompt, or
3. JFR `ForwardPass.prefill.*` is populated on the API prefill path.

Observed Juno pp (GPU, wall-derived): **8–20 t/s** vs llama **610–3583 t/s** — treat as directional only until parity is fixed. **Tier 8 owns** compare-script parity and prefill JFR wiring.

## JFR hotspot breakdown (GPU, resident models)

Example: **Phi-3.5-mini**, 64 decode tokens.

| Metric | Value | Notes |
|--------|------:|-------|
| `MatVec.duration.total_ms` | 4606 | 93% of decode forward |
| `MatVec.backend.cuda_resident_fp16.count` | 18881 | Full weight residency |
| `MatVec.duration.p95_ms` | 0.52 | Per-projection call |
| `ForwardPass.decode.total_ms` | 4941 | 64 passes → ~77 ms/token |
| `ForwardPass.prefill.count` | 0 | **Prefill not instrumented on this path** |
| `Tokenizer` + `TemplateFormat` | <10 ms total | Negligible |

Non-MatVec decode time (~7%) is attention softmax, KV IO, sampling, and JVM overhead — not the primary lever until MatVec is addressed.

Example: **mistral-7b** on GTX 1080 — entire model falls back to CPU quantized MatVec (`MatVec.backend.cpu.count` = 15009, p95 **27 ms**/call). GPU residency path never activates → tg ≈ CPU tg (~0.48 t/s).

## Tier impact matrix

| Tier | Phase | Perf lever | Expected uplift (order of magnitude) | Depends on |
|------|-------|------------|--------------------------------------|------------|
| **13 Phase B** | P0 step 5 | Single-stream decode (fused Q4 MMQ) | **2–5×** tg if MMQ closes llama gap | Phase A go memo |
| **8** | P0 step 3 | TTFT / pp | **2–10×** pp on long prompts (GPU) | Tier 1 |
| **5** | P0 step 2 | 7B+ on 8 GiB | **10–70×** tg vs full CPU fallback | None |
| **9 / 12** | P4 | Effective tg | **1.2–2×** at high acceptance | Tier 8 |
| **1** | P0 step 2 | Aggregate TPS | **~linear in batch** for non-stream | None |
| **6** | P1 step 1 | VRAM + KV bandwidth | Fit > speed; minor tg if KV-bound | Tier 5 rec. |
| **14** | P1 step 2 | VRAM fit + continuous KV | Memory scales on continuous path; **0× single-stream** | Tier 1, Tier 6 rec. |
| **14** gather-tax gate | P1 | Continuous viability | Must be ≤ ~10–15% attention at batch 8 / ctx 8k to start Tier 15 | Tier 14 microbench |
| **15–16** | P1 | Concurrent SSE QPS (local/single-shard) | vLLM-class aggregate on one machine; **0× single-stream**; cluster = static | 14 gate, 1, 6, 8 (16 only) |
| **Vector SIMD** | P0 step 4 + parallel track | CPU MatVec / long CPU prefill | TBD — publish `--vector 0` vs `1`; vision-safe Q5_K policy | Not a bake-off-only checkbox |

Tiers **2–4, 7, 10, 11** (P2/P3) are API/DX — no direct tg/pp uplift.

## Execution alignment

This analysis amends [`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md). Follow **phases**, not tier numbers:

| Phase | Steps | Gate |
|-------|-------|------|
| **P0** | 13A ✓ → 5† → 1† → 8† → **Vector SIMD track** → 13B († feature complete; Phi-3.5 0.5× **open**; mistral 0.15× **met**) | Phi-3.5 tg ≥ **0.5×** llama (**0.33×** current); mistral ≥ **0.15×** with `--gpu-layers auto` (**0.43×** **met**) |
| **P1** | 6 → 14 → 15 → 16 | Gather-tax gate; continuous SSE |
| **P2** | 2 → 3 → 4 (after Tier 1 feature complete) | API parity |
| **P3** | 7, 10, 11 | Per-tier gates |
| **P4** | 9 → 12 (after Tier 8) | Token identity |
| **P5** | 13 FlashAttn subset (after Tier 8 baselines) | Separate go memo |

**Parallel tracks** (Vision I2T, LoRA, model E2E, Vector SIMD): see [`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) → Parallel tracks. Vision + SIMD share MatVec/prefill risk; §2 requires `compare-vision.sh` when those paths change.

P1 begins after P0 **gate met** for peer claims (feature-complete P0 tiers may already be landed). Tier 14 gather-tax microbench is a **hard gate** before Tier 15. Cluster continuous deferred to `Tier 15b` follow-on.

## Compare-script action items (owned by Tier 8)

1. Add `--raw-prompt` / `--no-template` mode so Juno ingests exactly `n_prompt` tokens (match llama-bench).
2. Populate `ForwardPass.prefill.*` JFR on API chat prefill (today `prefill.count` = 0).
3. Add `--gpu-layers N` passthrough once Tier 5 lands (today `--gpu` is all-or-CPU).
4. Default published baselines should include both `--vector 0` and `--vector 1` rows.

## Tier 13 Phase A memo (complete)

**Question:** Does attention or dequant+GEMM dominate decode after resident GPU weights?

**Answer:** Dequant+GEMM (`juno.MatVec`) accounts for **>90%** of GPU decode time on Phi-3.5, Qwen2.5-3B, and TinyLlama. Attention/softmax/KV are secondary on this bench.

**Recommendation:** **Go** for Tier 13 Phase B (P0 step 5) scoped to **fused quantized matmul (MMQ)** on the Q4_K resident path — not FlashAttn first (decode seq len = 1; FA wins on long prefill/context). Re-evaluate FlashAttn in **P5** after Tier 8 long-context prefill baselines exist.

**Pending:** Record final memo section in `docs/performance.md`.

**Cheaper fixes ruled out or insufficient alone:**

- Batching (Tier 1): no single-stream uplift.
- KV quant (Tier 6): memory, not the 4× MatVec gap on decode.
- Partial offload (Tier 5): essential for fit, not for resident-model kernel efficiency.

## Post-MMQ GPU idle-time finding (2026-09-18) — the kernel is not the remaining bottleneck

**Context:** the Phase A memo above (and the `93–96%` MatVec-share figure throughout this doc) was
measured on the **`cuda_resident_fp16`** backend — i.e. before Tier 13B's Q4_K MMQ kernel shipped.
`PROMPT-P0-Gate.md` step 2 asks a future session to profile *why the current `q4k_gemv` kernel loses
to FP16 cuBLAS* using Nsight Compute. This session attempted that pass and found a different, more
fundamental answer than a kernel-tiling problem.

**Method:** `ncu` (hardware perf counters) hit `ERR_NVGPUCTRPERM` — this account lacks the elevated
GPU counter access `ncu` needs, and there is no passwordless `sudo` in this environment to grant it
non-interactively. As a substitute, `nsys profile --trace=cuda` (CUDA API/kernel **timeline** tracing,
which does not need the restricted counter permission) was used instead, on a live `./juno local
--mmq on --gpu-layers all --gpu` run (TinyLlama and Phi-3.5-mini, GTX 1080, short prompt, 20-40 decode
tokens). `nsys`'s own `.qdstrm → .nsys-rep` importer was broken in this install (`Importer binary and
its dependencies were not found`); worked around by invoking `QdstrmImporter` directly from
`/usr/lib/nsight-systems/host-linux-x64/`, then querying the resulting `.sqlite` export with Python
(`sqlite3` module) for `CUPTI_ACTIVITY_KIND_KERNEL` / `CUPTI_ACTIVITY_KIND_MEMCPY` timestamps.
Cross-validated against a `--jfr` recording of the identical TinyLlama run.

**Finding 1 — memcpy is not the decode bottleneck once model load is excluded.** The naive
`cudaMemcpyAsync`/`cudaMemcpy` totals across a whole process capture are dominated by the **one-time**
model-to-VRAM upload (2.5 GB HtoD, ~389 ms, all *before* the first `q4k_gemv` launch). Restricting to
the generation window only (first `q4k_gemv` launch → process end) drops memcpy to **13.7 ms of a
1007 ms window (1.4%)** — negligible during steady-state decode/prefill.

**Finding 2 — the GPU is idle ~73–82% of wall time during generation.** Summing all CUDA kernel
execution intervals inside the generation window: TinyLlama **203.7–218.6 ms busy of ~1120–1137 ms
window (17.9–19.6% utilization)**; Phi-3.5-mini **270.5 ms busy of 1007 ms window (26.8%
utilization)**. `q4k_gemv`/`q5k_gemv`/`q6k_gemv` themselves run in a reasonable 55–140 µs/launch
(consistent with near-bandwidth-bound execution for their data volume, not an obviously broken
kernel) — the kernel is not slow, it is idle most of the time.

**Finding 3 — it is not `cudaStreamSynchronize` (host blocked on GPU) either.** That API totals **17.8
ms** across the whole captured trace (well under 2% of any generation window) — the host is not stuck
waiting on the device; it is doing something else entirely off the CUDA API surface.

**Finding 4 — turning on the already-shipped `--gpu-attention` does not close the gap at short
context.** TinyLlama `--gpu-attention off` vs `on`, same prompt/token count: GPU utilization **17.9%
→ 19.6%**, wall time **1137 ms → 1116 ms** — noise-level, not the dominant lever at this context
length. (This does not contradict the ROADMAP's separate `ctx≈512` finding that attention reaches
64.2% of decode wall time at longer context — that is a context-length-dependent cost this session's
short-prompt test does not exercise.)

**Finding 5 — JFR cross-validation lands on the same number two independent ways.** A `--jfr`
recording of the identical TinyLlama run: `juno.MatVec` (cuda backend) totals **204 ms** over 3715
calls (avg 54.9 µs), `juno.Attention` totals **35 ms** over 902 calls (avg 38.8 µs), against a
`juno.ForwardPass` total of **894 ms**. `(204+35)/894 = 26.7%` — matching the Nsight Systems
GPU-utilization number (27%) almost exactly, via a completely independent measurement path (software
JFR spans vs. hardware/API kernel-timeline tracing).

**Reconciling with the historical 93–96% figure:** that number was real for the `cuda_resident_fp16`
backend, where the FP16 weight volume made each cuBLAS SGEMV call itself slow enough to dominate wall
time. Tier 13B's Q4_K MMQ kernel is much faster per call (confirmed above: 55–140 µs), but nothing
shrank the **host-side** cost that surrounds each of the thousands of per-projection kernel launches a
decode token issues (`Q4KMmqKernel.launchPacked`/`quantizeX` each build a fresh `Arena.ofConfined()`
and marshal several `MemorySegment`s per call — outside any JFR span, invisible to both `ncu`/`nsys`
GPU-side tracing and to the `juno.MatVec` span's own reported duration). Speeding up the kernel used to
help because the kernel *was* the bottleneck; now that it is 3-5x faster, the fixed per-launch host
overhead that used to be a rounding error is the new ceiling. **A hypothetical zero-cost kernel could
only improve Phi-3.5 decode by ~27% at most** on this evidence — nowhere near enough to close 0.326×
→ 0.5×. Code inspection also confirms `rmsNorm`/`rope`/residual-add/SwiGLU (`silu(gate)*up`) all run as
plain scalar Java loops with no JDK Vector API and no GPU kernel — architecturally consistent with
"non-GPU, non-MatVec-JFR-instrumented work fills the idle time," though this session did not
instrument those loops directly to apportion the remaining ~73% between host/FFI call overhead and
raw CPU elementwise compute.

**Implication for the next P0 lever:** tile/`mul_mat_vec`-class kernel tuning (`PROMPT-P0-Gate.md`
step 2's original framing) has a low ceiling now and is **not** the recommended next step. Higher-
leverage candidates, in rough order of expected leverage: (a) reduce per-launch host/FFI overhead —
CUDA graphs to capture-and-replay a whole layer's launch sequence, or fusing more of the ~7
per-layer projections into fewer/larger launches beyond the existing `sgemvSameX` QKV fusion; (b) move
`rmsNorm`/`rope`/residual-add/SwiGLU onto the GPU (already named as the "optional second lever" in
`PROMPT-P0-Gate.md`, now upgraded to primary-candidate status by this evidence) or at minimum vectorize
them on CPU. Neither is implemented by this session — this is a diagnosis, not a fix; see
`PROMPT-P0-Gate.md`'s dated addendum for the same evidence recorded against its exit checklist.

## Success metrics (program-level gates)

| Metric | Current (GPU JFR) | P0 target | P1 target |
|--------|-------------------|-----------|-----------|
| Phi-3.5 tg vs llama | **0.33×** (`20260911T235203Z`, `--mmq on`; was 0.22× FP16) | **≥0.5×** | ≥0.5× (unchanged) |
| mistral-7b tg on 8 GiB | **0.43×** (`20260911T235203Z`, `--gpu-layers auto` + `--mmq on`; was 0.01× CPU) | **≥0.15×** with `--gpu-layers auto` (**met**) | — |
| Long prefill pp (512 tok, matched count) | not measured | **≥0.3×** llama with Tier 8 | Tier 16 under load |
| 8 concurrent SSE aggregate TPS | not measured | — | ≥0.5× vLLM on same SKU |

## References

- [`docs/perf-compare/20260831T231403Z/INDEX.md`](../perf-compare/20260831T231403Z/INDEX.md) — GPU summary
- [`docs/perf-compare/20260831T230258Z/INDEX.md`](../perf-compare/20260831T230258Z/INDEX.md) — CPU summary
- [`scripts/performance-tests/compare-llama-cpp.sh`](../../scripts/performance-tests/compare-llama-cpp.sh) — harness
- Per-model JFR: `*-juno-jfr.json` under each run dir
