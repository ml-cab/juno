# Tier 00: Correctness & consistency audit

Status: complete (see "Execution record" below)
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
  StaticBatchPrefixCacheSessionGatingTest.java` (`GenerationLoopBatchTest` and `RequestSchedulerBatchTest`
  exist but do not cover the prefix cache) — a repeated or extended prompt in a *later* batch (a shared
  system prompt alone never hits, and one batch cannot hit itself), asserting each output is correct and
  that no request continues from KV it never wrote. Written against a KV-ownership test double
  (`KvTrackingPipeline`), plus `ContinuousPrefixCacheGatingTest` and a real-model
  `TinyLlamaStaticBatchLiveTest`.
- **`ForwardPassHandlerLoaderArchitectureTest`** (new class: no `ForwardPassHandlerLoaderTest` exists, only
  three narrower loader test classes): add one case per unrecognized architecture string
  (`qwen35`, `gemma4`, `mistral3`, `minimax-m2`) asserting the new guard's behavior (either a clear
  rejection, or — if step 2's audit finds a given fallback is actually safe — an explicit
  allowlisted pass-through with a comment explaining why it's safe).
- **`ModelLiveRunnerIT`**: for each model whose architecture has no verified handler, assert in-process
  that the loader rejects it with an error naming the architecture (a cluster node would swallow the
  failure, see the execution record), instead of running the suite on it.
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

## Execution record (2026-09-23)

Status: **complete**. Every in-scope item is implemented and verified. Three further findings that widened the
tier (cluster nodes swallowing model-load failures, the dead `RegistryService`/Hazelcast surface, and a REST
`temperature` of 0 not being greedy) were approved by the owner and are done (see "Decisions taken").

### Plan-versus-code drift found before starting

- HEAD is `93773dd`, not `0c519f1`. Commit `20c8715` changed `src/main` (`ConsoleMain`, `CudaRmsNorm`,
  `CudaBindings`, `CudaMatVec`, `GpuBindings`, `GpuContext`, `RocmBindings`, `PrefillBatchOptions`) after
  the snapshot. None of Tier 00's claims were affected.
- **§2.5 confirmed in code, with a nuance the plan missed.** `generateBatch()` called
  `findLongestPrefix` and `cachePrefix` with no gate. But `PrefixCache` only returns a hit when a whole
  previously cached prompt is a prefix of the new prompt (only trie leaves carry a key), callers use
  `matchedTokens()` and ignore the key, and within one batch every lookup happens before every write. Two
  requests that merely share a system prompt therefore do not trigger it, and the test as worded here (two
  concurrent stateless requests, shared system prompt) would pass vacuously. The trigger is a repeated or
  extended prompt in a **later** batch. `docs/howto.md` already documents the violated contract ("prefix KV
  reuse is session-scoped").
- **The plan's fix recipe is insufficient.** Adding the `hasSession` gate to `generateBatch()` would still
  let a *session* request inside a batch read a hit: `generateBatch()` keys pipeline KV by `requestId`
  (not `kvCacheKey()`) and evicts it on completion, so the matched positions live under a different key.
  Confirmed: `session_request_inside_a_static_batch_is_fully_prefilled` fails on the original code. The
  batch path now never reads or writes the trie.
- **`ContinuousBatchEngine` was only half gated.** Reads are (a hit is acted on only for sessions), but every
  non-hit slot registered `<key>:prefix`, which dangles for stateless slots because their KV is evicted at
  retirement. A later session request with the same prompt matched it and skipped prefill. Confirmed by
  `ContinuousPrefixCacheGatingTest` failing on the original code, then fixed.
- `ForwardPassHandlerLoaderTest` does not exist (three narrower loader test classes do). A new class,
  `ForwardPassHandlerLoaderArchitectureTest`, holds the new cases.
- §2.7: `docs/agent-arch.txt` and `docs/performance.md` already describe the CPU SIMD state accurately; only
  `CLAUDE.md` oversold it. Also, `SimdThreadPool.forEachRow` *is* on the hot path (a common-pool parallel
  stream); what is unused is its dedicated `POOL`.
- Item 9 named 11 files; the repo-wide grep this tier's rule 8 requires found 48 hits in 33 files under
  `src/main` (two of them in exception messages, two in `.cu` files, plus pointers to internal planning files).
  Not touched, because they are outside this tier's `src/main` criterion, and queued for the documentation
  audit tier: `docs/agent-arch.txt` (6 lines), `docs/performance.md` (26), `CHANGELOG.md` (39, historical
  entries) and 23 `src/test` files.
- The `mvn test -pl ...` command omitted `vision` and `metrics` in `CLAUDE.md` and twice in
  `docs/howto.md`. The documented live-IT invocation `-Dmodels=/path/to/models` is actually
  `-DMODELS=/abs/a.gguf,/abs/b.gguf` (`juno-master/pom.xml`); corrected in both docs.
- `CudaRmsNorm`'s javadoc said the path "activates automatically"; `LlamaTransformerHandler` deliberately
  leaves it `null` (and `tryCreate` is only called from tests). Corrected while rewriting its tier-number text.
- `TensorShardContext` was worse than "unwired": nothing reads `tensorRank`, so every node loads and runs the
  full model and the coordinator sums N complete logit vectors (N times the memory and compute, logits scaled
  by N). A `ClusterHarness` comment claimed "the node uses its tensorRank to slice weights". Corrected.
- `CLAUDE.md` is not tracked by git and the uncommitted `.gitignore` change also ignores it, so its
  corrections do not show in `git status` or `git diff`.
- Maven 3.8.7 is installed while `CLAUDE.md` says 3.9+. Everything built and tested offline on 3.8.7.

### §2.5 reproduction and fix

| Check | Original code | Fixed code |
|---|---|---|
| `StaticBatchPrefixCacheSessionGatingTest` (6 cases, KV-ownership test double) | 4 fail: `forward at position 14 but position 0 was never written under this key` | 6 pass |
| `ContinuousPrefixCacheGatingTest` (2 cases) | 1 fails (session request resumes from a stateless slot's dangling entry) | 2 pass |
| `TinyLlamaStaticBatchLiveTest` (real tinyllama Q4_K_M, CPU) | fails: repeated prompt in a later batch decodes `[22443, 23600, 6845, 15945, 13, 13]`, single-request output is `[1576, 7483, 310, 3444, 338, 3681]` | passes |

Over REST on real tinyllama (`scripts/performance-tests/smoke-tier00-consistency.sh`, `--parallel 2`, top-k pinned
to 1 because a REST `temperature` of 0 is not greedy, see decision 3), run against the original build and against
the fixed build:

| Check | Original build (CPU) | Fixed build |
|---|---|---|
| repeated prompt in a later concurrent round, `/v1/inference` | fails: `' Резуpitiful""\n\nBased'` instead of `The capital of France is Paris.` | passes (CPU and GPU) |
| session request after batched stateless traffic | fails: the same garbage text | passes (CPU and GPU) |
| server prefix-cache hits for that traffic | 3 | 0 (its single lookup is the session request) |
| continuous schedule: session request after a stateless request | fails: the same garbage text | passes |

The final run of the whole script on the fixed build passed 46 of 46 checks (architecture audit, CPU and GPU
batching over both REST surfaces, session checks, continuous check).

Controls that passed on the original code (so the fix does not remove legitimate behavior): a shared
system prompt with distinct user turns, and session prefix reuse on the single-request path
(`session_reuse_on_the_single_request_path_still_skips_the_cached_prefix`, and the continuous equivalent).

### §2.4 audit: the four architecture-mismatched files

Observed with `./juno gguf-info` and `./juno local --cpu --nodes 1` on the original code. All four reached
`LlamaTransformerHandler`; none was rejected for its architecture.

| File | `general.architecture` | Original outcome | Why it is not a safe fallback | Now |
|---|---|---|---|---|
| `Qwen3.5-0.8B.Q4_K_M.gguf` | `qwen35` | `Tensor not found: blk.0.ffn_norm.weight` (accidental, misleading) | hybrid: `ssm_*` recurrent tensors on 3 of 4 layers, fused `attn_qkv` + `attn_gate`, `post_attention_norm` | rejected by name |
| `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` | `gemma4` | default 4 GB heap: `OutOfMemoryError`; with `--heap 24g`: loads layers 0-23, then `Tensor not found: blk.24.attn_k.weight` | all tensor types supported (Q4_0/F32); sliding-window attention (512, patterned), final logit softcapping (30), per-layer input embeddings, KV-shared layers. A variant without KV sharing would load and run silently wrong | rejected by name |
| `Devstral-Small-2-24B-Instruct-2512-UD-IQ1_S.gguf` | `mistral3` | `No byte-size formula for GGML type 16` (unsupported quant, unrelated to architecture) | YaRN rope (factor 48); not verified as Llama-compatible | rejected by name |
| `minimax-m2.5-tiny-24e-iq4_nl-imat.gguf` | `minimax-m2` | `Unsupported tensor type 20 for tensor token_embd.weight` (unsupported quant) | routed MoE (24 experts, 8 used), qk-norm, no dense `ffn_gate/up/down` | rejected by name |

The loader now sends an architecture to `LlamaTransformerHandler` only if it is in
`LlamaFamilyArchitectures` (`llama`, `mistral`, `tinyllama`, `qwen2`, `qwen2.5`, the values already used by the
LoRA allowlist and seen on the real files); everything else without a dedicated handler raises an
`IOException` naming the architecture. `ForwardPassHandlerLoader.isSupportedArchitecture` exposes the same
rule. Verified on the real files by the smoke script and by `ModelLiveRunnerIT`.

Mixtral-shaped files (`llama` architecture with `ffn_*_exps` tensors) are not covered by this guard; from
code reading (not from a real file, none is present) they fail at `Tensor not found: blk.0.ffn_gate.weight`.
The structural expert-tensor check belongs to the model-architecture-breadth tier.

### Batched versus single-request numerics (measured, real weights, CPU)

Probe run on `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` with the same KV state fed to `forward` and
`forwardBatch` (batch of 2 identical rows), for three real prompts: max |logit difference| 2.8e-5 to
3.6e-5; the two identical rows inside one batch are bit-identical; the first-token argmax agrees with
top-1/top-2 gaps of 1.4 to 2.9. So batched and single decode are not bit-identical, but the difference is
about five orders of magnitude below a token decision, and it cannot explain the large divergences seen in
the first smoke run (those were random sampling, decision 3).

### Decisions taken (2026-09-23) and what was done

The execution above surfaced three things beyond the plan. All three were put to the owner and all three were
approved: fix the cluster fail-open path, remove `RegistryService`/Hazelcast, and make a REST `temperature` of 0
greedy.

1. **Cluster nodes swallowed model-load failures (fixed).** `EmbeddedNodeServer.loadShard` caught any exception from
   the real-model load, installed a stub handler returning fixed logits and replied `success = true` with a message
   beginning `Stub shard (model load failed: ...)`; `ProcessPipelineClient` and `TensorParallelPipelineClient` only
   logged that message and never checked `success`. So in `./juno cluster` an unsupported architecture (or
   quantization, missing tensor, missing file, bad adapter) produced a healthy-looking cluster serving dummy
   output. Now: the node replies `success = false` with the reason and installs the new `UnloadedShardHandler`,
   which refuses forward passes (a real-model node also refuses them before its first successful load); both clients
   fail `loadShards()` on any node that reports failure; `ClusterHarness.start()` stops the forked node JVMs before
   rethrowing. Stub mode (no model path) is unchanged. Tests written first and seen failing for the right reason:
   `EmbeddedNodeServerLoadFailureTest` (3 of 4 failed before), failure cases in `LoadShardsParallelTest` and
   `TensorParallelPipelineClientTest` (both failed before with "expecting code to raise a throwable"), and
   `UnsupportedArchitectureClusterIT` (forked JVMs, metadata-only GGUF; its no-leftover-JVMs assertion failed until
   the harness cleanup was added). The existing fake node in `LoadShardsParallelTest` never set `success`; it does
   now. On the real files, `ModelLiveRunnerIT` asserts that pipeline- and tensor-parallel starts fail naming the
   architecture, and the smoke script does the same through `./juno cluster` and checks that no `NodeMain`
   process is left.
2. **`RegistryService` / Hazelcast (removed).** Confirmed before removal: no `com.hazelcast` API was used
   anywhere, and the seven `RegistryService` messages had no implementation and no caller. Removed: the proto
   section (`RegistryService` plus `GetShardMap*`, `ShardAssignmentProto`, `RegisterNode*`, `RecomputeShards*`), the
   dependency in `node`, `health`, `kvcache`, `registry` and `coordinator`, the root `hazelcast.version` property
   and managed dependency, the unused port 5701 rule in `scripts/aws/juno-deploy.sh`, and the javadoc in
   `NodeHealth`, `HealthEvaluator`, `NodeDescriptor`, `ShardMap`, `ModelRegistry`, `ModelDescriptor` and
   `HealthReactor` that described a Hazelcast-backed registry (now describing the in-memory code). Module
   descriptions in the `health`, `kvcache` (which also claimed a Disk tier that no longer exists) and `registry`
   poms were corrected. `docs/dev-notes.txt` and `assets/meta-juno-doc.md` keep their Hazelcast text: both already
   carry an "out of date, kept for historical purposes" banner above it. The `CHANGELOG.md` design section is
   annotated. Regenerating the protobuf stubs and a full clean build confirmed nothing referenced the removed
   symbols. Tiers 07 and 13 were updated: any elastic-membership discovery is now designed from scratch in Tier 13.
3. **A REST `temperature` of 0 was not greedy (fixed).** `OpenAiChatHandler`, `InferenceApiServer` and
   `VisionChatHandler` only call `withTemperature(...)`; `TemperatureStep` skipped scaling below 1e-6 (an effective
   temperature of 1.0) and `SampleStep` drew randomly unless `greedy` was set, which only the CLI did. Found because
   the first batching smoke run (REST responses at `temperature: 0`) reported 18 differences that were random
   draws. Fixed once in the sampler rather than in each adapter: `SamplingParams.effectivelyGreedy()` (flag, or a
   temperature below `ZERO_TEMPERATURE` = 1e-6) is now used by the temperature, top-k, top-p and sample steps, which
   covers all three REST surfaces and the embedding facades. `SamplerTest` failed for the right reason before (300
   seeded draws at temperature 0 were not all the argmax) and passes now, with a control that a positive temperature
   still samples. The smoke script no longer pins top-k; it uses a plain `temperature: 0`.

### Cross-surface checklist, resolved

| # | Surface | Result | Evidence |
|---|---|---|---|
| 1 | CPU inference | PASS | `StaticBatchPrefixCacheSessionGatingTest`, `TinyLlamaStaticBatchLiveTest` (real tinyllama), smoke CPU batching rounds |
| 2 | CUDA GPU inference | PASS | smoke GPU batching rounds on the GTX 1080 (OpenAI and native, 16 of 16 match the single-request output, zero prefix-cache hits) |
| 3 | ROCm GPU inference | N/A | no ROCm-specific file changed; not run (no AMD hardware) |
| 4 | Static schedule | PASS | reproduction, fix and smoke above; batched requests always prefill in full |
| 5 | Continuous schedule | PASS on unit tests and one real-model REST check | `ContinuousPrefixCacheGatingTest`, `RequestSchedulerContinuousTest`, smoke `continuous` check (session request after a stateless request); reads were already gated and are unchanged |
| 6 | Single-node local mode | PASS | all four files rejected with an error naming the architecture (smoke and `ModelLiveRunnerIT`); tinyllama loads and answers |
| 7 | Pipeline-parallel cluster | PASS (FAIL-CLOSED for the four files) | `UnsupportedArchitectureClusterIT`, the client and node tests above, `ModelLiveRunnerIT` and the smoke script on the real files: start fails naming the architecture, no node JVMs left |
| 8 | Tensor-parallel cluster | PASS (FAIL-CLOSED for the four files); state of `TensorShardContext` documented, not changed | `TensorParallelPipelineClientTest`, `ModelLiveRunnerIT` and the smoke script on the real files; `docs/agent-arch.txt` |
| 9 | LoRA training | N/A | no training path changed |
| 10 | LoRA playback | PASS by construction, not run | the batch path no longer touches the trie, and the continuous engine's `--lora-play` read gate is unchanged; `LoraTrainingHandlerFactory`'s own allowlist is unchanged |
| 11 | Vision | N/A | vision text handlers (`llama`, `phi2`) are on the verified/dedicated list; `vision` unit tests pass |
| 12 | OpenAI REST surface | PASS | smoke (chat completions, batched and session; plain `temperature: 0` is deterministic) |
| 13 | Native REST surface | PASS | smoke (`/v1/inference`, batched; plain `temperature: 0` is deterministic) |
| 14 | CLI | PASS | `./juno gguf-info`, `./juno local` and `./juno cluster` (both `--pType` values) on the real files |


### Pre-existing flaky test

`metrics` `JfrMetricsExtractorAttentionTest.mixedPrefillAndDecode_aggregatesIndependently` failed once in the first
full `mvn verify` (`expected: 0.0023020000000000002 but was: 0.002302`). It asserts exact double equality of
`total_ms` against `prefill_ms + decode_ms` over real measured JFR durations, and `metrics` depends on no other
Juno module. Repeated 10 times it failed 3 times in the modified tree and once in the untouched original tree, so it
is timing-dependent independent of this tier. Not edited. The second full run passed it, with
`-Dsurefire.rerunFailingTestsCount=2` enabled and no flake reported.

### Verification commands and results

- `mvn -o clean verify -Dsurefire.rerunFailingTestsCount=2` at the repository root (all 16 modules, after all
  changes including the regenerated protobuf stubs): BUILD SUCCESS, 0 failures and no reported flakes in any module
  (`node` 578 tests / 41 skipped, `coordinator` 294 / 1, `sampler` 66, `vision` 95, `metrics` 38, `juno-player`
  104 / 3), and the forked-JVM integration tests `InProcessClusterIT` 6, `ThreeNodeClusterIT` 8,
  `TensorParallelClusterIT` 5 and `UnsupportedArchitectureClusterIT` 1. The skips are GPU/ROCm/missing-model-gated
  tests that were already skipped. An earlier full run (before the three decisions) also passed everything except
  the pre-existing flaky `metrics` test described below.
- `mvn -o verify -pl juno-master -am -Pintegration -DMODELS=<tinyllama + the four rejected files>
  -Dit.test=ModelLiveRunnerIT`: 5 cases, 0 failures (tinyllama full cluster suite 41 s; each rejected file 15 to
  21 s, because it also forks and fails a pipeline and a tensor cluster). A first attempt used
  `-Dtest=NoSuchTest`, which also filtered failsafe and silently ran nothing; that run was discarded.
- `scripts/performance-tests/smoke-tier00-consistency.sh`: 54 of 54 checks pass (4 `gguf-info`; 4 local rejections plus the tinyllama control; 8 `./juno cluster` rejections, both `--pType` values, with no leftover `NodeMain` process; 16 CPU and 16 GPU batched-versus-single comparisons over both REST surfaces; 3 session checks on the static and continuous schedules; 2 zero-prefix-cache-hit checks) on the fixed build; 5 failures on the
  original build (the same batching and session checks, run against a build of the original source with
  `JUNO_JAR`).

### Performance gate

Not run, and not required: the prefix-cache change removes work from the batch path (a trie lookup and a trie
write per batched request) and one redundant write per continuous slot; the cluster fail-closed change only
alters the load-failure path; the sampler change is one comparison per sampled token. The decode, prefill and
MatVec paths are untouched and no MatVec, GPU-residency, KV-layout or quantization code changed. The only
observable difference is that `prefixLookups` no longer counts stateless batched requests.

## Exit criteria

- [x] §2.5 repro test exists and passes (either because the fix was applied, or because
      investigation showed the theoretical risk doesn't materialize — documented either way).
      *Fix applied, plus the same fix for the continuous engine; see the execution record.*
- [x] Each of the four architecture-mismatched files has a documented, tested, intentional
      behavior (load-and-run-correctly, or fail-closed-with-clear-error) — none of them silently
      produce wrong output without an error.
      *Tested on the real files in local mode and in pipeline- and tensor-parallel cluster mode (decision 1).*
- [x] `Sampler.java`/`SamplingStep.java`/`RepetitionPenaltyStep.java` agree on pipeline order.
- [x] `RegistryService`/Hazelcast decision made and executed (removed, or filed as a new
      explicitly-scoped tier — not left ambiguous); `CHANGELOG.md`'s "Actors — Design Decisions"
      section corrected or annotated as historical/aspirational for the same Hazelcast claim.
      *Removed (proto RPCs, dependencies, dead port rule, stale comments); CHANGELOG annotated; a full clean
      build regenerated the stubs without any reference to the removed symbols.*
- [x] `CLAUDE.md` module table and LoRA optimizer description corrected; its `mvn test -pl ...`
      command includes `vision` and `metrics` or explains why it doesn't.
      *`CLAUDE.md` is untracked and now ignored by git, so this edit is not visible in `git status`.*
- [x] GBNF bounded-repetition cap has a real justifying comment or is removed.
      *Commented honestly (a size guard; the value is not measured); removal belongs to the sampling tier.*
- [x] `docs/agent-arch.txt` states the true wiring state of `TensorShardContext` and
      `FaultTolerantPipeline` in Juno-native language, with no plan-tree references.
- [x] A repo-wide grep for `Tier [0-9]+`/`Infra tier` across every `src/main` tree returns no hits
      outside `docs/infra-plan/`/`docs/lora-plan/`.
      *Checked with `grep -rnE "[Tt]iers? ?-?[0-9]+|Infra[ -][Tt]ier|PLAN-Infra"` over every `src/main` tree.*
- [x] Full cross-surface checklist above resolved (PASS/N/A/FAIL-CLOSED, no blanks).
      *Resolved table in the execution record; rows 3 and 9 are N/A, row 10 is PASS by construction and was
      not run.*
- [x] `mvn test` (all modules, including `vision` and `metrics`) and `mvn verify -pl juno-master`
      pass with zero regressions.
      *One pre-existing flaky `metrics` test failed once in the first full run; see the execution record.*
- [x] `CHANGELOG.md` entry added.
