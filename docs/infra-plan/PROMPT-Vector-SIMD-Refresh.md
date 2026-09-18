# Agent prompt: refresh the Vector SIMD `--vector 0` vs `--vector 1` bake-off

Copy everything below the line into a new agent session.

---

## Task

Re-run and re-publish the CPU `--vector 0` vs `--vector 1` comparison
(`PLAN-Infra-Vector-SIMD.md`) against current HEAD. The last published pair
([`20260904T194612Z`](../perf-compare/20260904T194612Z/) /
[`20260904T195731Z`](../perf-compare/20260904T195731Z/), Juno tg v1/v0 ~= 0.99-1.03) predates Tier
13 Phase B (fused Q4 MMQ), Tier 13 Phase C (`--gpu-attention`), and Tier 17 (GPU batched-prefill
GEMM) — none of those changed the CPU weight-stationary accumulate path (`VectorQuantKernels`
policy is unchanged: Q4_K/Q5_K stay scalar accumulate for the vision-hang-fix reason documented in
`PLAN-Infra-Vector-SIMD.md`), but every recent GPU-side bake-off in `docs/perf-compare/` has been
published with `juno_use_vector: 0` (see e.g. `docs/perf-compare/20260916T043335Z/host.json`), so
the CPU SIMD path has gone unverified for several sessions' worth of changes. This is a
**verification refresh, not new kernel work** — confirm the near-parity finding still holds, do not
attempt to make Vector faster on the hot accumulate path (that was already tried and reverted for
vision-hang reasons; see `PLAN-Infra-Vector-SIMD.md` "Chosen approach").

This is a **parallel track** re-verification, not a new Infra tier — it does not block or get
blocked by the active P0 decode-kernel gate (`PROMPT-P0-Gate.md`).

## Read first (mandatory)

1. [`models/CLAUDE.md`](../../models/CLAUDE.md) — tests first, KISS, no zip
2. [`PLAN-Infra-Vector-SIMD.md`](PLAN-Infra-Vector-SIMD.md) — the track this refreshes; do not
   change the chosen policy (scalar Q4/Q5 accumulate) without new evidence of a regression
3. [`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) — Execution rules §2 (perf compare), §3 (no
   competitor names outside this tree)
4. `scripts/performance-tests/compare-llama-cpp.sh --cpu` invocation, `--vector 0|1` flag wiring

## Suggested approach

1. `./scripts/performance-tests/compare-llama-cpp.sh --cpu --vector 0 --reps 1` (default 4-model
   set) — publish under `docs/perf-compare/<timestamp>/`.
2. `./scripts/performance-tests/compare-llama-cpp.sh --cpu --vector 1 --reps 1` — publish under a
   second `docs/perf-compare/<timestamp>/`.
3. Compute Juno tg v1/v0 ratio per model; compare against the last published `~0.99-1.03` baseline.
   Flag any model that regresses materially (either direction beyond noise) rather than silently
   averaging it away.
4. If `--gpu-attention`/Tier 17 changes touched any CPU-shared code path (check
   `VectorQuantKernels`, `SimdThreadPool`, `LlamaTransformerHandler.sgemmQ*WeightStationary` git
   history since the last refresh) and the ratio moved, investigate and document why before
   concluding parity still holds — do not assume "close to 1.0" without checking the diff is
   actually explainable.
5. Re-run `compare-vision.sh` only if step 4 finds a change to the shared CPU MatVec dispatch path
   (per `PLAN-Infra-Vector-SIMD.md` §3's own vision re-run condition) — otherwise state explicitly
   that it was skipped and why (unchanged dispatch path), not silently omitted.
6. Update `PLAN-Infra-Vector-SIMD.md`'s "Status vs plan" table with the new bake-off links and
   confirm/deny "Feature complete" status; update `docs/perf-compare/README.md`.

## Exit when

1. Fresh `--vector 0`/`--vector 1` pair published under `docs/perf-compare/`, default 4-model set.
2. Ratio compared against the `20260904` baseline with an explicit verdict (still near-parity, or
   regressed with a named cause).
3. `PLAN-Infra-Vector-SIMD.md` and `docs/perf-compare/README.md` updated.
4. No change to `VectorQuantKernels` policy unless the refresh finds a genuine regression — if it
   does, stop and report rather than silently patching the hot path (that space has a documented
   vision-hang history; any change needs its own plan, not a fix folded into a verification prompt).

## Preview

List changed/added files for preview; never zip.
