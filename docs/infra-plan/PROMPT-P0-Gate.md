# Agent prompt: close Phase P0 exit gate (MatVec / decode tg)

Copy everything below the line into a new agent session.

---

## Task

**Close the Phase P0 exit gate** — measured single-stream GPU decode throughput, not more feature flags.

P0 tiers (5, 1, 8, Vector SIMD, 13B VRAM-fit) are already **feature complete**. The **program gate is unmet**. Do **not** start a new Infra scheduler/API tier. Do **not** treat “`--mmq` exists” as success.

### P0 gate (must all pass on published bake-off)

| Metric | Target | Last known |
|--------|--------|------------|
| Phi-3.5 mini Q4_K_M GPU **tg** vs peer (same GGUF / SKU) | **≥ 0.5×** | **~0.22×** |
| Mistral-7B Instruct Q4_K_M on **8 GiB** with `--gpu-layers auto` (use `--mmq` if needed for fit) | **≥ 0.15×** | **~0.026×** without fit; MMQ fit once ~**0.14×** (near miss) |

### Deferred speed exit (also required for honest MMQ speed claims)

From [`PLAN-Infra-Tier13.md`](PLAN-Infra-Tier13.md):

- Tile / `mul_mat_vec`-class Q4_K kernel: **≥ 1.3×** decode tg vs `--mmq off` (FP16-resident) on Phi-3.5 or TinyLlama, **or** ≥ **1.5×** long-context prefill  
- Today’s PTX is **slower** than FP16 on GTX 1080 (Phi-3.5 MMQ ~**7.4** tg vs FP16 ~**12.2**) — that is the problem to fix

If after honest engineering the gate still fails, **report fail with numbers** — do not amend the ROADMAP gate downward unless the user explicitly asks.

## Why this exists

JFR ([`PLAN-Infra-PERF-ANALYSIS.md`](PLAN-Infra-PERF-ANALYSIS.md)): `juno.MatVec` ≈ **93–96%** of GPU decode. P1 (continuous / mixed prefill) improves concurrent load only and does **not** close interactive tg. Interactive peer claims are blocked until P0 gate is **met**.

**New evidence (2026-09-16), read before profiling:** a JFR `juno.Attention` span now exists (landed
alongside Tier 17, see [`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) "Attention JFR span landed"
note) and shows attention is **not** negligible at every context length the way Tier 13 Phase A's
short-context (64-token) finding implied. At `ctx≈512` (`compare-prefill-batch.sh --gpu --n-prompt
512`, TinyLlama), attention is **64.2%** of *decode* wall time and 78.3% of prefill wall time —
MatVec dominance is a short-context finding, not a universal one. Two implications for this gate:

1. **Re-baseline before touching kernels.** Step 1 below ("record Phi-3.5 and Mistral ratios") is
   also this program's overdue re-verification that the last published Phi-3.5 ratio (**0.33×**,
   2026-09-11, pre-Tier-17) still holds — nothing after that session touched the decode-kernel path,
   but no one has re-run the number since. Do this before any new kernel work, not after, so effort
   isn't spent chasing a stale target.
2. **`--gpu-attention` (Tier 13C, feature complete, default off) is a second, already-shipped
   decode-side lever for longer-context sessions**, not only the prefill lever it was built and
   measured for (`docs/performance.md:424-440`'s 3.85× pp / 78.7%→11.0% prefill-share numbers). It
   is untested as a *decode* tg lever at long context. If step 1's short-prompt re-baseline still
   shows Phi-3.5 short of 0.5×, consider a second measurement at longer context (`--gpu-attention
   on` vs `off`, decode tg only) before committing to new kernel authorship — it may close part of
   the gap for free on already-wired architectures (Llama-family/Mistral/Qwen2; Phi-3 itself is a
   named `--gpu-attention` follow-up per Tier 13's interaction matrix, so this lever does not apply
   to Phi-3.5-mini specifically — check TinyLlama/Qwen2.5-3B/Mistral instead, or land the Phi-3
   follow-up first).

## Read first (mandatory)

1. [`models/CLAUDE.md`](../../models/CLAUDE.md) — tests first, KISS, prefer new classes, list changed files (no zip)
2. [`docs/infra-plan/PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) — Execution rules **§1–§6**; P0 exit gate; Status vocabulary (feature complete vs **gate met**)
3. [`docs/infra-plan/PLAN-Infra-PERF-ANALYSIS.md`](PLAN-Infra-PERF-ANALYSIS.md) — MatVec dominance; success metrics table
4. [`docs/infra-plan/PLAN-Infra-Tier13.md`](PLAN-Infra-Tier13.md) — deferred tile-kernel exit; `sgemvSameX` Phase 1; what not to claim
5. [`docs/infra-plan/PLAN-Infra-Tier5.md`](PLAN-Infra-Tier5.md) — `--gpu-layers auto` mistral fit path
6. [`docs/performance.md`](../performance.md) — current MMQ / bake-off notes
7. [`docs/perf-compare/README.md`](../perf-compare/README.md) — publish layout
8. `.cursor/rules/juno-no-infra-tier-labels.mdc`, `juno-docs-no-competitors.mdc`, `juno-infra-lora-perf.mdc`

### Code already landed (extend — do not rip out VRAM-fit path)

| Piece | Location |
|-------|----------|
| Q4 MMQ PTX / load | `q4k_gemv.ptx`, `Q4KMmqKernel`, `DeviceQ4KMatrix` |
| CUDA path | `CudaMatVec` (`sgemv(DeviceQ4KMatrix)`, `sgemvSameX`, `supportsQ4KMmq`) |
| Handlers | `LlamaTransformerHandler`, `Phi3TransformerHandler`, `Qwen3TransformerHandler` Q4 projections |
| CLI | `--mmq on\|off\|auto`, `--gpu-layers N\|all\|auto` |
| Shared-activation Phase 1 | `MatVec.sgemvSameX` (Llama QKV / gate-up) |
| LoRA play MMQ | [`PLAN-Infra-LoRA-MMQ.md`](PLAN-Infra-LoRA-MMQ.md) Phase 1 |
| Bake-off harness | `scripts/performance-tests/compare-llama-cpp.sh` |

## Design constraints

- **Primary lever:** faster GPU MatVec on the Q4 decode path (tile / `mul_mat_vec`-class kernel and/or fuller device-resident activations). Partial offload alone will not hit Phi-3.5 **0.5×**.
- Keep `--mmq` default **off** unless bake-off proves speed **and** you update howto/CHANGELOG honestly.
- Prefer new kernel / binder classes over mega-editing handlers; keep parity tests green (existing Q4 MMQ parity tests + extend as needed).
- ROADMAP **§5**: all architectures that already use MMQ stay correct; Llama-only kernel experiment is OK only if other handlers keep working (or stay on prior path with documented follow-up **before** user-facing speed claims).
- ROADMAP **§6**: no silent `--mmq` / `--gpu-layers` no-ops on wired surfaces; warn or fail-closed.
- §2: after MatVec changes, run inference compare **and** `compare-lora.sh --gpu --baseline <last or release-0.1.2>` (touches forward/MatVec).
- No Infra tier numbers in user-facing docs, CLI help, JFR `@Description`, or CHANGELOG. No competitor product names outside `docs/infra-plan/` / `docs/perf-compare/`.
- **Out of scope:** FlashAttn (P5); rewriting continuous/mixed prefill (P1 done); P2 API fields; Metal/Vulkan; porting entire peer MMQ stacks wholesale.

## Suggested approach

1. Baseline: `./scripts/performance-tests/compare-llama-cpp.sh --gpu --vector 0` — record Phi-3.5 and Mistral ratios + JFR `MatVec` backend/share.
2. Profile why current `q4k_gemv` loses to FP16 cuBLAS (launch config, memory traffic, lack of tiling, host sync). Fix with a **tile / `mul_mat_vec`-class** kernel (or measured equivalent), behind existing `--mmq` wiring.
3. Optional second lever if still short of **0.5×**: extend device-resident activations beyond Phase‑1 `sgemvSameX` (norm/attn/residual on GPU) — only with parity tests.
4. Mistral on 8 GiB: `--gpu-layers auto` + `--mmq on` (or `auto`); prove JFR is **not** 100% `MatVec.backend.cpu`; hit **≥ 0.15×**.
5. Internal MMQ vs FP16 micro-gate: Phi-3.5 or TinyLlama decode tg **≥ 1.3×** `--mmq on` vs `off` (same residency), or document prefill **≥ 1.5×** if that is the winning path.
6. Publish under `docs/perf-compare/<timestamp>/` (+ LoRA `-lora/`); update `docs/perf-compare/README.md`, `docs/performance.md`, ROADMAP status to **P0 gate met** only if both program metrics pass; howto/CHANGELOG may then state measured throughput honestly (Juno terms only).
7. If gate unmet: leave ROADMAP as **gate open**, publish failing run, list next kernel levers — do not claim peer latency.

## Exit when

1. Published GPU compare shows **Phi-3.5 tg ≥ 0.5×** peer.
2. Published GPU compare shows **Mistral-7B on 8 GiB ≥ 0.15×** with `--gpu-layers auto` (MMQ allowed for fit); JFR proves GPU MatVec path.
3. Tile-kernel (or equivalent) bake-off row shows **≥ 1.3×** vs `--mmq off` decode **or** **≥ 1.5×** long prefill — required before any docs claim MMQ is a speed win.
4. §2 LoRA compare ok (train ≤ 1.25×, playback tps ≥ 0.80× baseline).
5. Parity tests for touched MatVec/handlers green; §6 still honest.
6. ROADMAP / Tier13 / `docs/performance.md` / CHANGELOG / howto updated only to match measured reality.
7. FlashAttn / new Infra tiers beyond this MatVec speed follow-on **not** started.

## Preview

List changed/added files for preview; never zip.
