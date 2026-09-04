# Agent prompt: plan LoRA MMQ wiring (risk analysis first)

Copy everything below the line into a new agent session.

---

## Task

**Produce a written plan only** — do **not** implement code in this session.

Write **`docs/infra-plan/PLAN-Infra-LoRA-MMQ.md`**: how to wire fused Q4_K MMQ (`--mmq` / `JUNO_MMQ`) into **`LoraResidentWeights` / `LoraTrainableHandler`** (and related LoRA training handlers) without breaking LoRA train, playback, merge, or default inference.

Also update:

- [`PLAN-Infra-Tier13.md`](PLAN-Infra-Tier13.md) — short cross-link under Phase B / architecture follow-up
- [`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) — one-line note that LoRA MMQ is a planned 13B adjacency (plan doc), **not** a new Infra tier and **not** started until this plan is reviewed

**Output of this session:** the plan markdown + ROADMAP/Tier13 pointers + list of files for preview. No Java/CUDA changes.

## Why this exists

Tier 13 Phase B landed MMQ on **`LlamaTransformerHandler`** only (`DeviceQ4KMatrix`, `Q4KMmqKernel`, PTX `q4k_gemv`).

Verified gap (2026-09-04):

```text
./juno local --model-path models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --mmq on --nodes 1 --lora-play models/tinyllama-1.1b-chat-v1.0.Q4_K_M.lora
```

- LoRA recall works (`My name is Juno`) — flags coexist.
- MMQ is **not** used: no log `Fused Q4_K MMQ enabled`; path is `LoraTrainableHandler` → `LoraResidentWeights.uploadQuant` → dequant → FP16/FP32 `ResidentWeightMatrix` → cuBLAS.

## Read first (mandatory)

1. [`models/CLAUDE.md`](../../models/CLAUDE.md) — KISS, prefer new classes, list files, no zip
2. [`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) — Execution rules §1–§6 (esp. §2 LoRA compare gate, §5 all handlers, **§6 cross-feature matrix**)
3. [`PLAN-Infra-Tier13.md`](PLAN-Infra-Tier13.md) — Phase B status; Llama-only MMQ today
4. [`docs/performance.md`](../performance.md) — MMQ / Vector SIMD notes
5. [`docs/howto.md`](../howto.md) — `--mmq`, `--lora-play`, LoRA train flags
6. `.cursor/rules/juno-docs-no-competitors.mdc`, `juno-no-infra-tier-labels.mdc`, `juno-infra-lora-perf.mdc`

### Code to analyze (mandatory)

| Area | Files |
|------|--------|
| MMQ (inference, done) | `MmqOptions`, `DeviceQ4KMatrix`, `Q4KMmqKernel`, `CudaDriverBindings`, `CudaMatVec`, `LlamaTransformerHandler` upload/hot path |
| LoRA frozen residency | `LoraResidentWeights`, `ResidentWeightMatrix`, `LoraResidentUpload`, `LoraMicrobatch` |
| LoRA handlers | `LoraTrainableHandler`, `Qwen2LoraTrainableHandler`, `Phi3LoraTrainableHandler`, `Qwen3LoraTrainableHandler`, `LoraTrainingHandlerFactory` |
| Loader / CLI | `ForwardPassHandlerLoader.load(..., adapters)`, `ConsoleMain` `--lora-play` / `--mmq` |
| Backward | `ResidentWeightMatrix.sgemvTranspose`, `LoraResidentWeights.transposedMatVec*`, FP16 transpose NaN notes in `LoraTrainableHandler` |
| Gates | `scripts/performance-tests/compare-lora.sh`, published LoRA baselines under `docs/perf-compare/` |

## Constraints for the plan

1. **Plan-only** — no implementation, no drive-by refactors.
2. **Default stays safe** — `--mmq off` must keep today’s LoRA FP16/FP32 behavior bit-compatible for train + play.
3. **Do not break training** — forward + **transpose** + microbatch (`--lora-microbatch`) + VRAM ladder (`juno.lora.train.device`, OOM → FP16 → CPU).
4. **Do not break playback** — `--lora-play` recall (TinyLlama Q&A gate) must remain true with `--mmq on` and `off`.
5. **No competitor names** outside `docs/infra-plan/` / `docs/perf-compare/`.
6. **No Infra tier numbers** in proposed CLI/JFR/user-doc strings (plan may say “Tier 13B adjacency” only inside `docs/infra-plan/`).
7. **ROCm** — call out explicitly: CUDA-only today; ROCm stays FP16/FP32 unless a later HIP kernel plan exists.
8. Prefer **shared helpers** over copy-paste of Llama upload logic into every LoRA handler (KISS + ROADMAP §5).

## Risk analysis (required section in the plan)

The plan **must** include a risk register with severity, likelihood, mitigation, and how to test. At minimum cover:

| Risk ID | Topic | Why it matters |
|---------|--------|----------------|
| R1 | **Transpose / backward** | MMQ kernel is forward GEMV only. Training needs `W^T * g`. No Q4 transpose kernel today. Blindly putting Q4 on device breaks train or forces silent CPU transpose (perf + NaN history on FP16 transpose). |
| R2 | **Microbatch / FP32 residency** | `LoraMicrobatch` > 1 prefers FP32 for `GpuBlasOps`. Q4 packed residency conflicts with batched GEMM. Plan must say: MMQ only when microbatch=1? dual storage? play-only? |
| R3 | **Playback vs train split** | `--lora-play` is inference-only; train uses same handler class. Flag policy: MMQ for play only until transpose exists? |
| R4 | **VRAM / OOM ladder** | `LoraResidentWeights.tryRecoverFromUploadOom`, `LoraResidentUpload` FP32→FP16→CPU. Q4 upload changes sizes; OOM recovery must not leak device buffers or leave half-Q4 / half-FP16 state. |
| R5 | **Non-Q4_K tensors** | Q4_K_M models often use Q6_K (or other) for some tensors (e.g. output). Hybrid residency like Llama MMQ path. |
| R6 | **Adapter math** | LoRA delta is `B(Ax)` on activations after frozen `W*x`. Frozen path change must not alter adapter application order or dtype. |
| R7 | **Phi-3 / Qwen fused layouts** | Fused QKV / different row ranges; row-slice Q4 upload harder than Llama separate projections. |
| R8 | **JFR / gates** | Need `cuda-resident-q4k` counts under LoRA play; `compare-lora.sh` §2 ratios vs baseline; false “MMQ on” claims if still FP16. |
| R9 | **Default-off regress** | Accidental `auto` enabling MMQ on LoRA train without transpose. |
| R10 | **Vision / SIMD** | LoRA path ≠ vision encoder, but shared MatVec/upload helpers must not reintroduce vision-scale hangs or change Vector policy. |

Add any further risks found while reading code.

## Plan document structure (required)

`PLAN-Infra-LoRA-MMQ.md` should contain:

1. **Purpose / status** — adjacency to Tier 13B; plan-only until approved
2. **Current vs desired architecture** — short mermaid or table (Llama MMQ vs LoRA residency)
3. **Recommended approach** — pick one primary strategy and justify:
   - **A.** Playback-only MMQ (`--lora-play` / inference forward); train stays FP16/FP32 until transpose exists
   - **B.** Dual resident (Q4 for forward + FP16/FP32 for transpose) — VRAM cost
   - **C.** Defer LoRA MMQ until a Q4 transpose kernel exists
   - **D.** Other (must be concrete)
4. **API / type design** — e.g. extend `ResidentWeightMatrix` vs new `ResidentQ4KWeight` vs shared upload helper used by Llama + LoRA
5. **Handler scope** — Llama `LoraTrainableHandler` first; Phi3/Qwen2/Qwen3 follow-up checklist
6. **Flag semantics** — interaction of `JUNO_MMQ` with `juno.lora.train.device`, microbatch, `--cpu`
7. **Risk register** — full table from above + discoveries
8. **Test plan** — unit/parity first; then:
   - `compare-lora.sh --gpu --baseline …` with `--mmq off` (no regress) and `--mmq on` (playback gate)
   - Manual TinyLlama `--lora-play` recall with JFR proving `cuda-resident-q4k` when on
   - Train smoke: one `/train-qa` step with `--mmq on` must not NaN / must match policy in §3
9. **Exit criteria** — when implementation may start; when LoRA MMQ may be called done
10. **Out of scope** — FlashAttn, ROCm HIP kernel, full ggml MMQ, changing default `--mmq` to on
11. **Preview files** — expected touch list for a *future* implementation PR (not this session)

## Suggested analysis order

1. Trace `--lora-play` load → `uploadResidentWeights` → `matVecLayer` / `frozenTranspose`.
2. Compare to `LlamaTransformerHandler` MMQ branch (`tryMmq`, `uploadProjection`, `matVecProjection`).
3. Decide playback-only vs train-capable from R1/R2 evidence.
4. Draft plan; update Tier13 + ROADMAP pointers.
5. Stop. Do not implement.

## Done when

- [ ] `docs/infra-plan/PLAN-Infra-LoRA-MMQ.md` exists with risk register and a single recommended approach
- [ ] Tier13 + ROADMAP link the plan
- [ ] No production code changed
- [ ] Preview file list printed for the user
