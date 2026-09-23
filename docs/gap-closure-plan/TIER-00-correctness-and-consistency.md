# Tier 00: Correctness & consistency audit

Status: not started
Gap analysis refs: §2.1, §2.4, §2.5, §2.6, §2.7, §2.9, plus the real-file architecture findings in
[`INVENTORY.md`](INVENTORY.md)

## Objective

Close out every suspected correctness bug and doc/code drift item found during the gap analysis
before any new feature work begins, so later tiers build on a foundation that's known-correct
rather than "probably fine." This tier produces no new user-facing features; it produces confidence
and a clean baseline.

## Why this tier, why now

Two of the five items below (§2.4, §2.5) are *suspected real bugs*, not stylistic issues — one of
them (§2.5) risks serving one request's response using another request's never-written KV data,
under ordinary production traffic (two clients sharing a system prompt, static batching enabled).
Building six more tiers of features on top of an unverified correctness bug in the KV/prefix-cache
path would make root-causing any future regression much harder. The doc-drift items (§2.1, §2.7,
§2.9) are cheap to fix and, left alone, actively mislead whoever picks up the next tier (a future
session, human or agent, that trusts `TensorShardContext`'s javadoc in Tier 09 would waste real
time before discovering it's unwired).

## Scope

### In scope

1. **§2.5 — `GenerationLoop.generateBatch()` prefix-cache session gating.** Confirm or refute the
   suspected bug with a real reproduction, then fix if confirmed.
2. **§2.4 — MoE/unrecognized-architecture fail-closed audit.** For each of the four real files in
   `models/` with an architecture string outside `{phi2, phi3, qwen3, qwen3moe}`
   (`qwen35`, `gemma4`, `mistral3`, `minimax-m2`), determine and document the actual current
   behavior, and make the outcome explicit and fail-closed where it isn't already.
3. **§2.1 — Sampler pipeline doc drift.** Fix the three-way disagreement between `Sampler.java`'s
   real order, `SamplingStep.java`'s javadoc, and `RepetitionPenaltyStep.java`'s "Step 5" label.
4. **§2.6 — `RegistryService`/Hazelcast decision.** Decide: implement it for real, or remove the
   proto RPCs and the unused Hazelcast dependency from every module `pom.xml`. This tier makes the
   decision and either removes the dead surface (small, same-tier work) or files it as an explicit,
   scoped follow-up tier (large work, deferred) — it does not attempt a partial implementation. While
   auditing this, also check `CHANGELOG.md`'s "Actors — Design Decisions" section, which describes a
   "Hazelcast distributed `IMap`" model registry and Hazelcast `CP FencedLock`-based leader/standby
   coordinator election — a repo-wide grep for actual Hazelcast API usage (not just the dependency
   declaration) turns up nothing in `src/main` anywhere, so this section appears to document a design
   that was never built, the same pattern as the proto RPCs themselves. Correct or annotate it as
   historical/aspirational in the same pass, since `CHANGELOG.md` is not in Tier 14's later doc-audit
   scope and this would otherwise never get caught.
5. **§2.7 — "JDK Vector API CPU kernels" doc correction.** Update `CLAUDE.md`'s module table (and
   any other doc making the same claim) to describe what's actually on the hot path by default.
6. **§2.9 — Minor naming fixes.** `CLAUDE.md`'s "Adam optimizer" → "AdamW (with LoRA+)" for the
   `lora` module description; add a one-line comment on the GBNF bounded-repetition cap explaining
   the `max ≤ min + 8` choice or removing it if no real reason survives investigation. Also fix
   `CLAUDE.md`'s "Build, test, run" `mvn test -pl tokenizer,lora,node,coordinator,sampler,kvcache,
   health,registry,juno-player` command, which silently omits two real modules with their own test
   suites — `vision` and `metrics` — add them to the documented command (or explain in the same
   place why they're intentionally run separately).
7. **§2.3 — `FaultTolerantPipeline` wiring status.** Not fixed in this tier (that's a real feature,
   scoped into Tier 13), but document its current unwired state explicitly in
   `docs/agent-arch.txt` so it stops reading as if it's load-bearing.
8. **§2.2 — `TensorShardContext` wiring status.** Same treatment as §2.3: document the true state
   now (its javadoc currently overclaims), defer the actual fix to Tier 09.
9. **Tier-number leakage into shipped code.** `CLAUDE.md` already bans internal "Infra tier" numbers
   from shipped docs, code comments, JFR `@Description`, CLI help, and error messages — but as of
   this writing that rule is already violated in at least `LlamaTransformerHandler.java` ("Tier 19"),
   `LoraTrainingHandlerFactory.java` ("Tier 6"), `CpuFrozenBatchOps.java` ("Tier 9"),
   `LoraTrainableHandler.java` ("Tier 9"), `CudaDriverBindings.java` ("Tier 19"),
   `LoraTrainEvent.java` ("Tier 4"), `ContinuousLoraPolicy.java` ("Tier 10"), `Q8_0KvCodec.java`
   ("Tier 6"), `LoraTrainingConfig.java` ("Tier 1/2/3/8"), `ConsoleMain.java` ("Tier 16"), and
   `CudaRmsNorm.java` ("Tier 19"). Grep every `src/main` tree for `Tier [0-9]+`/`Infra tier` and
   rewrite each hit to describe the actual mechanism instead of the internal planning-tier number
   that introduced it (e.g. "Tier 19 Phase A GPU-resident RMS norm" → "GPU-resident RMS-norm path,
   currently unconstructed by default — see class javadoc for why"). This is the same class of
   finding as §2.7/§2.9 above — cheap, mechanical, and worth doing before 13 more tiers have a chance
   to add more of the same pattern. Tier 14 re-runs this grep at the very end as a regression check
   (alongside its existing competitor-name grep), the same relationship Tier 00/14 already have for
   the competitor-name rule.

### Out of scope (deferred to later tiers, tracked explicitly so they aren't lost)

- Building real per-layer tensor-parallel compute (§2.2) → Tier 09.
- Wiring `FaultTolerantPipeline` into the production cluster path (§2.3) → Tier 13.
- Building real handlers for `qwen35`/`gemma4`/`mistral3`/`minimax-m2` → Tier 08. This tier only
  ensures the *current* (unsupported) behavior is safe, not that these models become supported.
- Implementing `RegistryService` for real, if that's the decision reached in step 4 → filed as a
  new tier at that point, not built here.

## Cross-surface compatibility checklist

| # | Surface | What Tier 00 must verify |
|---|---|---|
| 1 | CPU inference | §2.5 repro must be run CPU-side (cheaper, faster iteration) |
| 2 | CUDA GPU inference | §2.5 repro re-run on GPU to confirm the bug (or its absence) isn't CPU-only |
| 3 | ROCm GPU inference | N/A — no code path changes touch ROCm-specific files in this tier |
| 4 | Static schedule | §2.5 is specifically a static-schedule (`generateBatch`) bug — primary target |
| 5 | Continuous schedule | confirm `ContinuousBatchEngine`'s existing `hasSession` gate is unaffected by any fix |
| 6 | Single-node local mode | all four unrecognized-architecture models load (or fail) via `./juno local` |
| 7 | Pipeline-parallel cluster | N/A this tier |
| 8 | Tensor-parallel cluster | N/A this tier (see §2.2 documentation-only scope above) |
| 9 | LoRA training | N/A this tier |
| 10 | LoRA playback | confirm §2.5's fix doesn't change prefix-cache behavior under `--lora-play` (LoRA changes the tokenized prefix and already invalidates cross-turn hits per `docs/howto.md:59` — must stay true) |
| 11 | Vision | N/A this tier |
| 12 | OpenAI REST surface | reproduce §2.5 via two concurrent `/v1/chat/completions` calls sharing a system prompt, `--parallel 2+` |
| 13 | Native REST surface | same repro via `/v1/inference` |
| 14 | CLI | `./juno gguf-info` + `./juno local --model-path <each unrecognized-architecture file>` for the §2.4 audit |

## Implementation steps

1. **Reproduce §2.5 first, before writing any fix.** Write a new integration test (see Tests
   below) that starts two stateless (no `sessionId`) requests sharing an identical system-prompt
   prefix through `RequestScheduler`/`GenerationLoop.generateBatch()` with `--parallel 2`, using
   `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`, and asserts both responses are coherent and correct. If
   the test fails (garbage/wrong output on the second request), the bug is confirmed — proceed to
   fix by adding the same `hasSession` gate `generate()`/`ContinuousBatchEngine` already use to
   `generateBatch()`'s prefix-cache read (`findLongestPrefix`) and write (`cachePrefix`) calls. If
   the test passes, keep it anyway as a permanent regression test and document in this file why the
   theoretical risk didn't materialize (e.g. some other guard already prevents it) — don't just
   delete the concern.
2. **Audit §2.4 for each of the four architecture-mismatched files.** For each, run
   `./juno gguf-info` to confirm exact tensor names present, then `./juno local --model-path
   <file>` and observe: does it load and produce output (possibly wrong), or does it fail at load
   time (and why — architecture mismatch, or the separate unsupported-quant issue for
   `Devstral`/`minimax-m2.5`)? Record findings in a table in this file. For any file that loads and
   *runs* despite an unrecognized architecture string (i.e., silently falls through to
   `LlamaTransformerHandler` and produces output without error), add an explicit architecture
   allowlist/guard: `ForwardPassHandlerLoader` should only silently fall through to
   `LlamaTransformerHandler` for architecture strings it has verified are tensor-layout-compatible
   (today: dense Llama, Mistral, Qwen2 per `docs/howto.md:78`) — anything else should raise a clear
   error naming the unrecognized architecture, rather than guessing. This is the fail-closed fix.
3. **Fix §2.1** by making `Sampler.java`'s real order the single source of truth: update
   `SamplingStep.java`'s class javadoc to match, fix `RepetitionPenaltyStep.java`'s stale "Step 5"
   label to reflect its real position, and add one comment on `Sampler.create()` pointing out that
   this file is the only authoritative description of pipeline order.
4. **Decide §2.6.** Read `RegistryService`'s proto definition and every `hazelcast` dependency
   declaration; make the implement-vs-remove call (default recommendation: remove now, since
   dynamic cluster membership isn't needed by any currently-planned tier and an unused dependency
   is pure liability — but confirm this with the user before deleting proto RPCs, since that's a
   public API-surface change even if unimplemented). If removing: delete the RPCs from
   `inference.proto`, remove `hazelcast` from every `pom.xml` that declares it, regenerate
   protobuf stubs, `mvn clean package -DskipTests` to confirm nothing referenced the removed
   symbols.
5. **Fix §2.7 and §2.9** as straightforward doc edits.
6. **Document §2.2 and §2.3's true state** in `docs/agent-arch.txt` — one paragraph each, stating
   plainly what's built vs. wired, with a forward pointer to the tier that will finish the job
   (Tier 09 / Tier 13) stated in Juno-native terms ("planned for a future tensor-parallelism
   milestone," not a literal path into this plan tree — see execution rule 5).

## Tests to write/upgrade before implementation

- **New integration test**: `coordinator/src/test/java/cab/ml/juno/coordinator/
  StaticBatchPrefixCacheSessionGatingTest.java` (or extend an existing `GenerationLoop` batch test
  if one already covers similar ground — check `GenerationLoopTest`/`RequestSchedulerTest` first) —
  two concurrent stateless requests, shared system prompt, `--parallel 2`, asserting both outputs
  are correct (not just non-crashing).
- **`ForwardPassHandlerLoaderTest`**: add one case per unrecognized architecture string
  (`qwen35`, `gemma4`, `mistral3`, `minimax-m2`) asserting the new guard's behavior (either a clear
  rejection, or — if step 2's audit finds a given fallback is actually safe — an explicit
  allowlisted pass-through with a comment explaining why it's safe).
- **`ModelLiveRunnerIT`**: add a check that loads each of the four architecture-mismatched files via
  `./juno local` and asserts the documented outcome (clean rejection or correct output — never a
  crash or silent garbage).
- **New bash smoke script**: `scripts/performance-tests/smoke-tier00-consistency.sh` — runs
  `./juno gguf-info` and `./juno local --model-path` against all four architecture-mismatched
  files plus one known-good file (`tinyllama`), asserting exit codes and log output match what
  step 2's audit table says they should.
- No perf-compare rerun is required for this tier unless the §2.5 fix changes the prefix-cache hot
  path's behavior for the *passing* case (it shouldn't — the fix only changes behavior for the
  previously-unguarded cross-request case) — confirm this expectation explicitly rather than
  skipping the gate by default.

## Models needed

All models needed for this tier are already present (`tinyllama`, `qwen35`, `gemma4`, `mistral3`
via `Devstral`, `minimax-m2` via `minimax-m2.5-tiny`). No downloads required for Tier 00.

## Exit criteria

- [ ] §2.5 repro test exists and passes (either because the fix was applied, or because
      investigation showed the theoretical risk doesn't materialize — documented either way).
- [ ] Each of the four architecture-mismatched files has a documented, tested, intentional
      behavior (load-and-run-correctly, or fail-closed-with-clear-error) — none of them silently
      produce wrong output without an error.
- [ ] `Sampler.java`/`SamplingStep.java`/`RepetitionPenaltyStep.java` agree on pipeline order.
- [ ] `RegistryService`/Hazelcast decision made and executed (removed, or filed as a new
      explicitly-scoped tier — not left ambiguous); `CHANGELOG.md`'s "Actors — Design Decisions"
      section corrected or annotated as historical/aspirational for the same Hazelcast claim.
- [ ] `CLAUDE.md` module table and LoRA optimizer description corrected; its `mvn test -pl ...`
      command includes `vision` and `metrics` or explains why it doesn't.
- [ ] GBNF bounded-repetition cap has a real justifying comment or is removed.
- [ ] `docs/agent-arch.txt` states the true wiring state of `TensorShardContext` and
      `FaultTolerantPipeline` in Juno-native language, with no plan-tree references.
- [ ] A repo-wide grep for `Tier [0-9]+`/`Infra tier` across every `src/main` tree returns no hits
      outside `docs/infra-plan/`/`docs/lora-plan/`.
- [ ] Full cross-surface checklist above resolved (PASS/N/A/FAIL-CLOSED, no blanks).
- [ ] `mvn test` (all modules, including `vision` and `metrics`) and `mvn verify -pl juno-master`
      pass with zero regressions.
- [ ] `CHANGELOG.md` entry added.
