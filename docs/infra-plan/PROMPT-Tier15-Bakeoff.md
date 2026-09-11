# Agent prompt: continuous schedule bake-off vs static (Tier 15 exit)

Copy everything below the line into a new agent session.

---

## Task

**Finish continuous batching (`--schedule continuous`) to feature complete** by publishing the bake-off evidence the exit checklist still requires. The engine is largely landed; **do not mark the step done** until numbers exist under `docs/perf-compare/` and `docs/performance.md`.

**Required bake-off evidence (all three):**

1. **Multi-session TPS** — continuous vs `--schedule static` (same model, session count, max tokens, GPU when available). Record aggregate tokens/s and latency.
2. **Concurrent SSE TTFT / TPOT** — overlapping `stream: true` clients under continuous; record time-to-first-token and inter-token latency (JFR and/or wall). Prove streams share `forwardBatch` / `juno.ContinuousStep`, not only HTTP 200.
3. **Prefix hit-rate + TTFT** — fixed shared-system-prompt workload; use landed `PrefixCache` / `KVCacheManager` lookup/hit counters; document hit rate and TTFT. Confirm continuous does not always wipe the trie after a cohort.

**Also required for feature complete (ROADMAP §2):**

- Publish inference + LoRA compares (KV/batching surface).
- Close open §6 cross-feature smoke checkboxes in [`PLAN-Infra-Tier15.md`](PLAN-Infra-Tier15.md).
- Update ROADMAP / CHANGELOG / howto / README status only after exit gates pass.

**Out of scope this session:** mixed chunked prefill + decode fairness ([`PLAN-Infra-Tier16.md`](PLAN-Infra-Tier16.md)). That is the **next P1 step after** this bake-off marks continuous batching feature complete. **Do not start Tier 16.**

## Why this exists

Session 74 landed the continuous engine (running set, SSE share, cluster auto-fallback, prefix counters, multi-LoRA fail-closed). [`docs/performance.md`](../performance.md) and [`PLAN-Infra-Tier15.md`](PLAN-Infra-Tier15.md) still say bake-off / JFR TTFT / hit-rate **numbers** are open. ROADMAP Execution rule §1: one Infra tier in flight; §2 bake-off before feature complete.

P1 phase gate language (ROADMAP): continuous SSE should beat static on local/single-shard for the measured workloads — record pass/fail honestly; do not invent peer claims without numbers.

## Read first (mandatory)

1. [`models/CLAUDE.md`](../../models/CLAUDE.md) — tests first, KISS, prefer new classes, list changed files (no zip)
2. [`docs/infra-plan/PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) — Execution rules §1–§6; P1 step 3; Tier 15 exit bullets
3. [`docs/infra-plan/PLAN-Infra-Tier15.md`](PLAN-Infra-Tier15.md) — exit checklist; todos 4 + 6 open on bake-off; interaction matrix §6 smokes unchecked
4. [`docs/infra-plan/PLAN-Infra-Tier1.md`](PLAN-Infra-Tier1.md) — static baseline; `static` must stay ≡ Tier 1 (dense KV, SSE isolated)
5. [`docs/performance.md`](../performance.md) — continuous note; **stale** smoke row `continuous + --parallel 2` still says “scheduler not enabled yet” — refresh or replace
6. [`docs/perf-compare/README.md`](../perf-compare/README.md) — publish layout; static multi-session baseline [`20260901T173121Z-parallel`](../perf-compare/20260901T173121Z-parallel/)
7. `.cursor/rules/juno-no-infra-tier-labels.mdc`, `juno-docs-no-competitors.mdc`, `juno-infra-lora-perf.mdc`

### Code already landed (extend / measure — do not rewrite the engine)

| Piece | Location |
|-------|----------|
| Continuous engine | `coordinator/.../ContinuousBatchEngine.java`, `ContinuousStepEvent.java` |
| Schedule policy / cluster fallback | `ServeSchedulePolicy.java`, `ServeScheduleOptions` |
| LoRA × continuous | `ContinuousLoraPolicy.java` |
| Scheduler wiring | `RequestScheduler.java`, `GenerationLoop.java`, `OpenAiChatHandler.java` |
| Prefix counters | `kvcache/.../PrefixCache.java`, `KVCacheManager.prefixLookups/Hits/HitRate()` |
| Launchers | `ConsoleMain`, `JunoPlayer`, `CoordinatorMain`, `scripts/run.sh` / `run.bat` |
| Unit tests | `ContinuousLoraPolicyTest`, `RequestSchedulerContinuousTest`, `ServeSchedulePolicyTest`, `PrefixCacheTest` |

## Design constraints

- **Local / in-process only** for continuous bake-off (`juno local` / player). Cluster stays auto-fallback to static — do not claim cluster continuous.
- **Default remains `static`.** Measure continuous vs static; do not flip API default.
- Prefer extending [`compare-parallel.sh`](../../scripts/performance-tests/compare-parallel.sh) or adding a dedicated script (e.g. `compare-schedule.sh`) over undocumented one-off shell. Publish under `docs/perf-compare/<timestamp>-continuous/` (or similar) with `INDEX.md`.
- Capture **JFR** (`juno.ContinuousStep`, `juno.TokenProduced`, existing latency events) where practical.
- Shared-prefix workload: same system prompt across N requests; record lookups, hits, hit rate, TTFT.
- Concurrent SSE: prove shared steps (logs/JFR `ContinuousStep` with `decodeBatchSize ≥ 2`), not only that HTTP returns 200.
- No infra tier numbers in user-facing docs, CLI help, JFR `@Description`, or CHANGELOG prose. No competitor product names outside `docs/infra-plan/` / `docs/perf-compare/`.
- Do not start mixed chunked prefill. Admit-time / full-prefill-before-decode under continuous is acceptable for this exit.

## Suggested approach

1. Confirm `mvn test -pl coordinator,juno-player,juno-master,kvcache -am` still green (or targeted continuous + scheduler tests if full suite is too long).
2. **Multi-session TPS:** same recipe as `compare-parallel.sh` (TinyLlama Q4_K_M, 8 sessions, `max_tokens=64`) with `--schedule static` vs `--schedule continuous` (and `--parallel` as the continuous running-set cap). Publish aggregate TPS + notes.
3. **Concurrent SSE:** N streaming clients; measure TTFT (time to first SSE token) and TPOT (inter-token); confirm continuous path in logs/JFR.
4. **Prefix:** fixed shared system prompt; dump `prefixLookups` / `prefixHits` / `prefixHitRate` + TTFT; document LoRA/tools conflict behavior if not already in howto.
5. **§2:** `compare-llama-cpp.sh --gpu --vector 0` (regression) + `compare-lora.sh --gpu --baseline <last or release-0.1.2>`; publish rows in `docs/perf-compare/README.md`.
6. Finish Tier 15 §6 smoke checkboxes; replace the stale `docs/performance.md` smoke row that claims continuous scheduler is not enabled.
7. Mark continuous batching **feature complete** in ROADMAP + Tier15 plan + CHANGELOG only after bake-off artifacts exist. State next Infra = mixed chunked prefill ([`PLAN-Infra-Tier16.md`](PLAN-Infra-Tier16.md)) — **do not implement it**.

## Exit when

1. Published continuous-vs-static multi-session TPS under `docs/perf-compare/`.
2. Concurrent SSE TTFT/TPOT recorded (JFR and/or wall) in bake-off INDEX + `docs/performance.md`.
3. Prefix hit-rate and TTFT on shared-system-prompt workload documented with counter numbers.
4. §2 inference + LoRA compares published; §6 smoke checklist closed for wired / no-op cells.
5. ROADMAP / `PLAN-Infra-Tier15.md` status = feature complete; CHANGELOG session note; howto/README accurate for continuous vs cluster.
6. `mvn test` for touched modules passes.
7. Mixed chunked prefill (Tier 16) **not** started.

## Preview

List changed/added files for preview; never zip.
