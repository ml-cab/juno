# Agent prompt: fix vision regression (working `47-vision` → broken merge/release)

Copy everything below the line into a new agent session.

---

## Task

**Diagnose and fix** why `POST /v1/vision/chat` works on branch **`47-vision`** but hangs / never finishes on **`47-vision-merge`**, **`release-0.1.2`**, and current inference branches.

Goal: restore a working moondream vision path on the branch you are fixing (likely `67-inference` or a fix branch off `47-vision-merge` / `release-0.1.2`), such that:

```bash
./scripts/performance-tests/compare-vision.sh --gpu --current HEAD --no-publish
```

exits **0** with `status=success`, `prompt_tokens≈741`, non-empty reply, within a reasonable wall time (target: **≤ ~10–15 min** on this host; working baseline was ~8.4 min latency).

Do **not** “fix” by disabling vision or lowering quality gates. Prefer a correct root-cause fix over a workaround.

## Evidence already collected (do not re-litigate)

### Working baseline — `47-vision` @ `a5255d3`

Ran successfully 2026-09-04:

```bash
./scripts/performance-tests/compare-vision.sh --gpu --current 47-vision
```

| Field | Value |
|-------|-------|
| Status | **success** |
| Prompt tokens | **741** |
| Completion tokens | 32 |
| `latency_ms` | **502863** (~8.4 min) |
| Decode tps (JFR `TokenProduced.tps`) | ~1.41 |
| Reply | non-empty (describes color squares) |
| Artifacts | `target/perf-compare-vision/20260904T141315Z/` and `docs/perf-compare/20260904T141315Z-vision/` |

JFR notes on the **working** run (important):

- `MatVec.backend.cpu.count` = **234837**, `cuda.count` = **0** — even with `--gpu`, this host ran **CPU MatVec** for moondream (GTX 1080 / CUDA libs present but residency not used).
- `ForwardPass.prefill.count` = 3, `decode.count` = 2189 — request **did** complete forward passes and produce tokens.

### Broken — `47-vision-merge` @ `29d2559`

Ran 2026-09-03:

```bash
./scripts/performance-tests/compare-vision.sh --gpu --current 47-vision-merge
```

| Field | Value |
|-------|-------|
| Boot | OK — API healthy, **Vision routes registered** |
| Request | Accepted (`queueDepth: 1`) |
| Outcome | **curl timeout** after ~2h (`http=000`, `rc=28`) |
| Artifacts | `target/perf-compare-vision/20260903T181951Z/` (`status: failure`) |

Runtime observations while hung:

- ~900% CPU in ForkJoin workers for the entire ~2h.
- Stack samples stuck in **`LlamaTransformerHandler.sgemmQ5KWeightStationary`** → `VectorQuantKernels.dot` (SIMD).
- GPU residency ~**118 MiB** only.
- On exit/timeout JFR showed: `Tokenizer.encode=1`, **`ForwardPass.count=0`**, **`TokenProduced.count=0`**, `MatVec.count=0` — prefill never recorded a completed forward pass.

Same class of breakage reported on **`release-0.1.2`** and current inference work (do not use those as the “known good” reference).

## Commit window to bisect / review

Known-good tip → first known-bad tip:

```text
47-vision (a5255d3)  …  47-vision-merge (29d2559)
```

High-suspicion commits (vision / prefill / SIMD):

| Commit | Summary |
|--------|---------|
| `baa285a` | VisionEncoder batched sgemm; **Phi2 `forwardBatch` / `runLayersBatch`**; **Q5_K weight-stationary CPU kernel**; KV `evict(requestId)` wired through `VisionAwareForwardPassHandler` / `GenerationLoop` |
| `2ebe837` | Vector API SIMD kernel spec + vision handling (#74) |
| `6b2d240` | SIMD quantized matmul + `SimdThreadPool` (#75) |
| `29d2559` | `#68` `CHAT_BOUNDARY_PIECES` filter for vision sub-tokens |

Also inspect diffs for:

- `vision/VisionEncoder.java`
- `vision/VisionAwareForwardPassHandler.java`
- `node/Phi2TransformerHandler.java` (batched prefill)
- `node/LlamaTransformerHandler.java` (`sgemmQ5KWeightStationary`)
- `node/VectorQuantKernels.java`, `node/SimdThreadPool.java`
- `coordinator/GenerationLoop.java`
- `node/ForwardPassHandler.java` / loader

```bash
git log --oneline a5255d3..29d2559
git diff a5255d3..29d2559 -- \
  vision/ node/src/main/java/cab/ml/juno/node/Phi2TransformerHandler.java \
  node/src/main/java/cab/ml/juno/node/LlamaTransformerHandler.java \
  node/src/main/java/cab/ml/juno/node/VectorQuantKernels.java \
  node/src/main/java/cab/ml/juno/node/SimdThreadPool.java \
  coordinator/src/main/java/cab/ml/juno/coordinator/GenerationLoop.java
```

Suggested approach: **git bisect** between `a5255d3` (good) and `29d2559` (bad) using a short smoke (see Verification), or surgically compare Phi2 batched prefill + Q5_K path vs the sequential path that still works on `47-vision`.

## Host / scenario constraints

- Machine: **medion-Precision-T3610**, GTX 1080 8 GiB, CPU Xeon E5-1650 v2 (12 threads).
- Model: `models/moondream2-q5_k.llamafile` (embedded vision, **no** `--mmproj-path`).
- Image: `scripts/performance-tests/fixtures/vision-bench.jpg`.
- Script: `scripts/performance-tests/compare-vision.sh` (already exists).
- Prefer `--api-port 18081` (or free port); stop any leftover `./juno local` first.
- `CUDA_HOME` / `CUDA_PATH` are often **unset** here; distro CUDA is under `/usr/lib/x86_64-linux-gnu`. Script already extends `LD_LIBRARY_PATH`. Do **not** assume GPU MatVec is active — the working baseline was CPU MatVec.
- Do not treat “GPU flag set” as proof of CUDA residency.

## Read first (mandatory)

1. [`models/CLAUDE.md`](../../models/CLAUDE.md) — KISS, minimal scope
2. [`docs/infra-plan/PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) — Execution rules §1–§5
3. [`docs/infra-plan/PLAN-Infra-SUPPORTED-MODELS.md`](PLAN-Infra-SUPPORTED-MODELS.md) — vision wraps text handlers
4. Working artifacts: [`docs/perf-compare/20260904T141315Z-vision/`](../perf-compare/20260904T141315Z-vision/)
5. Failed artifacts (if still present): `target/perf-compare-vision/20260903T181951Z/`
6. [`scripts/performance-tests/compare-vision.sh`](../../scripts/performance-tests/compare-vision.sh)
7. Vision path: `VisionChatHandler` → `VisionAwareForwardPassHandler` → encoder + Phi2/Llama handler

## Likely failure modes to check first

1. **Phi2 batched prefill / Q5_K weight-stationary** (`baa285a`) never completes for B≈741 on this CPU — algorithmic hang, wrong loop bounds, or pathological complexity vs sequential path on `47-vision`.
2. **SIMD `VectorQuantKernels` / `SimdThreadPool`** correctness or livelock under vision batch sizes.
3. **KV `evict` / request-id wiring** interacting badly with vision splice path (less likely for a pure hang before first token, but check).
4. **`CHAT_BOUNDARY_PIECES` (#68)** — more likely empty/wrong reply than a 2h hang; deprioritize unless bisect lands here.
5. Progress metrics: if stuck again, jstack should show whether time is in `VisionEncoder`, `sgemmQ5KWeightStationary`, or elsewhere; JFR should eventually show MatVec/ForwardPass if the instrumented path is reached.

## Fix requirements

- Restore functional vision chat on the target branch for moondream llamafile.
- Keep vision encoder batching benefits **if** they remain correct; if batched Q5_K prefill is the bug, fix the kernel/batch path rather than permanently deleting batching unless that is the only safe fix.
- Add or extend a **regression test** if feasible (unit/integration around Phi2 vision prefill or Q5_K batched sgemm). Do not rely only on the 8–15 min shell bench for CI.
- Update `docs/perf-compare/README.md` with a short note once a fixed run is published (optional if user did not ask to publish).
- No unrelated refactors; no competitor names in user-facing docs outside allowed trees; no Infra tier labels in shipped surfaces.

## Verification

Smoke (manual, known-good still works):

```bash
./scripts/performance-tests/compare-vision.sh --gpu --current 47-vision --no-publish
# expect: status=success, prompt_tokens=741, exits 0
```

After fix (on your branch / HEAD):

```bash
./scripts/performance-tests/compare-vision.sh --gpu --current HEAD --no-publish
# expect: status=success, prompt_tokens in 700–800, non-empty reply, exits 0
# wall time should not approach the 2h curl timeout
```

Optional A/B once fixed:

```bash
./scripts/performance-tests/compare-vision.sh --gpu --baseline 47-vision --current HEAD --no-publish
```

If you use bisect, a single vision request with a shorter curl timeout (e.g. 20–30 min) is enough to label good/bad — full JFR publish is not required for each step.

## Deliverables checklist

- [ ] Root cause identified (commit + mechanism)
- [ ] Code fix on the intended branch
- [ ] `compare-vision.sh --gpu --current HEAD --no-publish` exits 0
- [ ] Brief note of cause + fix in the PR / session summary
- [ ] Regression test if practical

## Out of scope

- Replacing moondream with another model for the gate
- Building a new Java perf harness (shell script already exists)
- Fixing CUDA residency / GPU MatVec on this host (nice-to-have; not required for the functional regression)
- Comparing against llama.cpp / other engines in user-facing docs
