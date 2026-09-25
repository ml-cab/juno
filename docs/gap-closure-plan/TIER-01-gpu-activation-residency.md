# Tier 01: GPU activation-residency redesign

Status: in progress (see "Execution record" below)
Gap analysis refs: §1.6, §2.8

## Objective

Fix the single architectural root cause behind three independently-measured regressions
(GPU-resident RMSNorm/RoPE/SwiGLU, draft-model speculative decoding, and the CPU SIMD hot path
rejection): every GPU op in `MatVec`/`GpuBindings` today is host-`float[]`-in/host-`float[]`-out,
so chaining GPU ops within a single transformer layer pays a host round trip between every op
regardless of whether each individual op is cheap on-device. This tier builds the minimum
activation-residency primitive needed to keep an activation vector/batch on-device across a
*chain* of ops (norm → projection → attention → norm → FFN) within one layer, and re-measures the
previously-shelved GPU-resident RMSNorm using it, as the tier's proof point.

## Why this tier, why now

This is the highest-leverage architectural investment identified in the gap analysis (§1.6): it is
a prerequisite for GPU-resident RMSNorm/RoPE/SwiGLU (§2.8, currently dormant scaffolding), for
draft-model speculative decoding to stop regressing (§1.5, currently 0.52x), and for any future
fused-attention kernel (Tier 02 depends on being able to keep QK^T and softmax intermediate state
on-device). Doing it once now, immediately after Tier 00's correctness pass, means Tiers 02-13 can
build GPU-side features without independently re-discovering and re-shelving the same dispatch-cost
wall three more times.

## Scope

### In scope

1. A new on-device activation-residency abstraction (name TBD during design —
   e.g. `DeviceActivation`/`ResidentTensor`) that represents a batch of activation vectors living in
   device memory across multiple consecutive GPU ops, only materializing back to host `float[]`
   when a downstream consumer that isn't yet GPU-resident needs it (e.g. handing off to a CPU-side
   sampler, or to a node that will forward the activation to a different process over gRPC).
2. Wiring this through the one existing, already-tested-but-dormant op: `CudaRmsNorm`/
   `CudaGraphSession`. Re-measure RMSNorm decode latency with residency in place, on the same
   GTX 1080/TinyLlama/Mistral-7B workloads `docs/performance.md`'s Phase B analysis used, to confirm
   the 6.9x-slower-than-CPU-scalar finding is actually fixed by residency (not just "somewhat
   better") before calling this tier done.
3. Extending residency to at least one additional op in the same layer (RoPE is the natural next
   op after the Q/K/V projection) to prove the chain — not just a single isolated op — is what
   fixes the round-trip cost. `RopeKernel`, mentioned as "never built" in the gap analysis, gets
   built here, using the residency primitive from day one rather than the old ad-hoc per-op
   pattern.
4. A clear, tested boundary: where does residency start and end within one decode step? (e.g.
   resident from post-embedding-lookup through the FFN output, materializing to host only at the
   point the residual stream needs to leave the GPU — for the LM head, for gRPC handoff in cluster
   mode, or for CPU-side sampling.) Document this boundary since every later tier that adds a
   GPU-side op needs to know whether it should participate in residency or not.

### Out of scope (explicitly deferred)

- Full-layer GPU residency for every op (attention, FFN, residual-add) — this tier proves the
  pattern on norm+RoPE; extending it to the full layer for every transformer handler
  (`LlamaTransformerHandler`, `Phi2TransformerHandler`, `Phi3TransformerHandler`,
  `Qwen3TransformerHandler`, `Qwen3MoeTransformerHandler`) is real follow-on work sized into later
  tiers as they touch each op (Tier 02 for attention, Tier 06 for speculative decoding's
  draft-model forward passes).
- Re-fixing draft-model speculative decoding's 0.52x regression — that's Tier 06's job, but Tier 06
  depends on this tier's primitive existing first.
- ROCm-side residency — CUDA only for this tier (ROCm has no CUDA-graph equivalent readily
  available; a ROCm residency design is a separate, later decision, tracked in Tier 10).
- Multi-GPU/cross-process residency — out of scope; this is single-device, single-process only.

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | must be provably unaffected — residency is a CUDA-only code path; scalar CPU path is the correctness oracle for comparison |
| 2 | CUDA GPU inference | primary target; must show a measured, not just theoretical, improvement over the Phase B baseline |
| 3 | ROCm GPU inference | N/A this tier — explicitly out of scope, must not regress ROCm's existing (already GEMV-only) behavior |
| 4 | Static schedule | residency must work for the existing static micro-batch path (batch size ≥ 1) |
| 5 | Continuous schedule | residency must work across mixed prefill/decode steps — verify separately, since `ContinuousBatchEngine` calls `forwardBatch`/`prefillBatch` differently than the static path |
| 6 | Single-node local mode | primary dev/test surface |
| 7 | Pipeline-parallel cluster | activation must still correctly materialize to host at the point it's serialized over gRPC to the next node — this is the clearest test that the residency boundary is correctly drawn |
| 8 | Tensor-parallel cluster | same materialization requirement, at the AllReduce boundary |
| 9 | LoRA training | training's frozen-weight forward pass currently stays FP16/FP32 host-resident by design (`LoraMmqPolicy`) — confirm residency doesn't change training numerics; training keeps ignoring `--mmq`/GPU-attention as before, unaffected |
| 10 | LoRA playback | LoRA delta add-on (`applyLoraInPlace`) happens after the base projection — confirm it correctly reads a materialized (or residency-aware) activation, not stale host data |
| 11 | Vision | `VisionEncoder` reuses the same `MatVec` primitives — confirm residency doesn't silently apply to vision's very different batch shapes (vision batches are wide, ~741, vs. decode's batch=1) without being re-validated there first; if it's not extended to vision this tier, it must be explicitly bypassed, not accidentally partially applied |
| 12 | OpenAI REST surface | end-to-end chat completion latency/output must be byte-identical to pre-tier for CPU, and correct (not necessarily identical, since GPU float ops can have different rounding) for GPU |
| 13 | Native REST surface | same |
| 14 | CLI | `./juno local`/`cluster` unaffected in flags/UX; this is an internal change |

## Implementation steps

1. Write the correctness/perf test harness first (see Tests below) so the "regresses decode" claim
   can be re-measured objectively before any residency code exists — this gives a before/after
   baseline.
2. Design and implement the minimal residency primitive, scoped to RMSNorm + RoPE only.
3. Wire `CudaRmsNorm` and the new `RopeKernel` through it inside `LlamaTransformerHandler`'s decode
   path (the only handler with a fused GQA attention kernel today), gated behind the existing
   `--gpu-attention`-style opt-in flag pattern so it can be toggled off if something regresses.
4. Re-run the Phase B-style microbenchmark (batch=1, dim=2048, N iterations) and the full
   `compare-lora.sh`/model-sweep perf gate, plus `compare-llama-cpp.sh` for a llama.cpp-relative
   reading on the same workload; publish results under `docs/perf-compare/`.
5. If the measured win is confirmed: wire the flag on by default and update
   `docs/howto.md`/`docs/performance.md` accordingly (Juno-native language, no competitor names).
   If not confirmed: follow the contingency in this tier's exit criteria (document, mark
   partial-complete, escalate to the user) rather than continuing to iterate past this checkpoint
   without a decision.

## Tests to write/upgrade before implementation

- **Unit tests**: extend `CudaGraphSessionTest`/add `CudaRmsNormTest` cases that assert
  bit-for-bit-or-within-tolerance correctness of the residency-backed path against the existing CPU
  scalar RMSNorm and the existing ad-hoc (non-resident) `CudaRmsNorm`, on real GPU hardware.
- **New unit test** for the residency primitive itself: allocate, chain two ops, materialize,
  confirm no leaked device memory (add a device-memory-accounting assertion — Juno already tracks
  VRAM pressure per `GpuBindings.memGetInfo`, reuse that).
- **`LlamaTransformerHandlerVerifyParityTest`** (already exists for speculative decoding verify
  parity) or a new sibling test: confirm decode output is unchanged with residency on vs. off for a
  fixed seed/greedy run.
- **`ModelLiveRunnerIT`**: add a GPU-path check (if not already gated to GPU-only hosts) asserting
  correct output with the new flag on, using `tinyllama` and `mistral-7b`.
- **New bash smoke script**: `scripts/performance-tests/smoke-tier01-gpu-residency.sh` — runs
  `./juno local` with the residency flag on/off against `tinyllama`, `mistral-7b`, and
  `llama-1-30b` (the large model, to catch any VRAM-accounting regression under memory pressure),
  diffing greedy-decode output between the two flag states (must be identical or within documented
  floating-point tolerance) and asserting the on-state doesn't crash or leak VRAM across repeated
  requests.
- **Perf gate (required — this is a hot-path change)**: `compare-lora.sh` plus a
  targeted RMSNorm/RoPE microbenchmark rerun of the existing Phase B methodology; publish under
  `docs/perf-compare/<timestamp>-tier01-residency/`.

  **Threshold.** All measured on the GTX 1080, median of three runs with min/max published per the
  README's noise-floor rule.
  - The residency-backed RMSNorm+RoPE chain must reach **>= 1.0x** the CPU scalar path at decode
    width (batch 1, dim 2048) — parity at minimum, not the ~6.9x-slower Phase B result. This is the
    proof point the whole tier turns on, and the contingency in the exit criteria is what happens if
    it is missed.
  - Chaining must beat op-at-a-time: the two-op resident chain must cost **<= 0.7x** the same two
    ops run through today's host-round-trip-between-ops path, since eliminating one round trip is
    the entire mechanism being tested. If the chain is no cheaper than the sum of its parts, the
    primitive is not doing what it was built to do, whatever the absolute numbers say.
  - No regression elsewhere: end-to-end tg within **0.95x** of the pre-tier baseline on every sweep
    model with residency enabled, and `compare-lora.sh` at train **>= 0.95x**, wall-clock playback
    tps **>= 0.80x**.
  - Device memory returns to its pre-request level after each request (no leak across repeated
    requests), asserted via `GpuBindings.memGetInfo` rather than inferred from not crashing.
- **Benchmark-parity preconditions (blocking, and they come first).** This tier carries the harness
  corrections described in [`README.md`](README.md)'s "Benchmark parity preconditions" — prompt-token
  parity, Juno warmup and repetitions, matched thread count, fixed heap, recorded clock state, one
  shared JFR configuration — because it is the first tier to publish a llama.cpp-relative ratio and
  every later tier's gate inherits whatever this one establishes. Land them and re-baseline *before*
  measuring any residency result, so the tier's own before/after comparison is taken on one side of
  the change rather than across it. The `--threads` control the parity work wants does not exist
  until Tier 10 item 5; until then, record Juno's effective parallelism in the run metadata and state
  the mismatch in the published INDEX rather than leaving it implicit.

  Two of these are bigger than they look and neither is optional:

  - **Prompt-token parity is the one that moves the numbers.** `RAW_PROMPT` defaults to `0` today,
    so every published pp ratio compared llama-bench's `-p N` against Juno's hard-coded 20-to-30
    token sentence. Flipping the default and enforcing the 10% `prompt_tokens`-versus-`n_prompt`
    check is what makes the pp column mean anything. Expect the re-baselined pp ratios to move
    substantially, and in the unflattering direction.
  - **The JFR work is net-new tooling, not a settings tweak.** Add
    `scripts/performance-tests/juno-perf.jfc` and point all four recording sites at it — `run.sh`'s
    `cmd_test`, `ClusterHarness`'s forked-node flag, and both `Configuration.getConfiguration("default")`
    calls in `ConsoleMain`; the last two are what `compare-llama-cpp.sh` actually exercises, since it
    passes `--jfr` as an app argument. Then extend `JfrMetricsExtractor` with the `jdk.*` bucket the
    README specifies, because today it consumes only `juno.*` events and **no gate in this plan can
    report a GC pause or an allocation rate until it does**. Tier 10's cost breakdown and its
    allocation-rate exit criterion both read from what this tier builds here.

- **Re-derive the program target's pp row.** The README's `>= 0.15x` GPU pp target was set against
  readings that were never like-for-like. Once the parity-corrected re-baseline exists, restate that
  target against it in this tier's own file — keep it, raise it, or lower it, but state the number
  and the reasoning, because Tiers 01B through 14 are all scored against it.

## Models needed

`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`, `mistral-7b-instruct-v0.1-q4_k_m.gguf`, and
`llama-1-30b.Q4_K_M.gguf` are already present and sufficient. No downloads needed.


## Execution record

### 2026-09-23 — benchmark-parity preconditions, part 1 (measurement tooling)

Scope of this pass: exit criteria 1 and 2 only. No residency code was written, and no residency
number was measured. These two land first because every later reading in this plan is taken through
them.

**Plan-versus-code drift found before starting.** None affecting this tier. HEAD is `3830501`, not
the `0c519f1` both this tree and the gap analysis were snapshotted against; Tier 00 has since been
executed and closed out. Its claims were re-verified against the current tree rather than trusted:
`generateBatch()` no longer touches the prefix trie, `ForwardPassHandlerLoader` routes through
`LlamaFamilyArchitectures` and rejects anything unverified, no Hazelcast or `RegistryService`
symbol survives anywhere, the repo-wide grep for internal tier numbers over `src/main` is clean, and
`CLAUDE.md` lists `vision` and `metrics` in its test command. All six claims hold.

Two small corrections to this tier's own text, found while doing the work:

- The tier names **four** JFR recording sites. There are six in the repository. The two it does not
  name are `scripts/run.bat`s test command (the Windows counterpart of the one it does name, also on
  `settings=profile`) and three sites in `scripts/aws/juno-deploy.sh`, also `settings=profile`.
  The `run.bat` site is fixed here because it is the same command on another platform. The AWS
  deployment script is left alone deliberately: it launches remote instances rather than the
  benchmark host, no published number comes from it, and the settings file it would have to name is
  not present on those instances.
- The tier names the two in-process sites as `startLocalJfr` and the cluster-coordinator recording.
  `startLocalJfr` in fact delegates to `startProgrammaticJfr`, which also serves LoRA mode, so
  fixing the two call sites covers three entry points rather than two.

**What shipped.**

| Item | Where |
|---|---|
| The single JFR configuration | `scripts/performance-tests/juno-perf.jfc` (new) |
| Packaged so a jar-launched run resolves the same file | `juno-player/pom.xml` resource entry |
| Resolver: system property, then packaged copy, then working copy | `JunoJfrSettings` (new, `juno-player`) |
| Local and LoRA recording site | `ConsoleMain.startProgrammaticJfr` |
| Cluster-coordinator recording site | `ConsoleMain.startClusterJfr` |
| Forked cluster-node JVMs | `ClusterHarness` |
| Launcher test command, both platforms | `scripts/run.sh`, `scripts/run.bat` |
| The `jdk.*` accounting | `JdkEventBucket` (new, `metrics`), called from `JfrMetricsExtractor` |
| Prompt-token parity, the parity gate, GC/allocation in every result JSON and INDEX | `scripts/performance-tests/compare-llama-cpp.sh` |

**The configuration.** Derived from the JDK stock low-overhead configuration. All twenty `juno.*`
events are enabled by name, so a recording no longer depends on each event class annotation default
or on which entry point started it. Six JVM events are pinned: `jdk.GCPhasePause` (threshold 0),
`jdk.ThreadAllocationStatistics` (500 ms), `jdk.ObjectAllocationSample` (300/s, with stack traces),
`jdk.ExecutionSample` (10 ms), `jdk.JavaMonitorEnter` and `jdk.ThreadPark` (10 ms thresholds). The
pinned settings carry no `control` attribute, so the value in the file is the value used.

One deliberate departure from the README specification. `jdk.NativeMethodSample` is left at its
stock 20 ms rather than tightened. The stock configuration already enables it, and
`startProgrammaticJfr` carries a comment recording that tightened native-thread stack sampling was
behind a `JfrSamplerThread ShouldNotReachHere()` crash under `--gpu --mmq` on `--lora-play`, a JDK
bug suspending a thread inside a Panama downcall. The README asks for `jdk.ExecutionSample` at
10 ms, which is the Java-side sampler and a different event; that one is tightened as specified.

**Fallback behaviour.** An unresolved settings file falls back to the JDK stock configuration and
says so on the console and in the launcher output. It does not fail the run, but it does not stay
quiet either: a recording that cannot state which settings produced it must not be compared against
one taken under the Juno settings.

**The `jdk.*` metrics.** `JdkEventBucket` is a new class rather than more code inside
`JfrMetricsExtractor`; the extractor gained three lines. It emits
`jdk.GCPhasePause.count`/`.max_ms`/`.total_ms`,
`jdk.ThreadAllocationStatistics.bytes_total`, `jdk.ObjectAllocationSample.count`/
`.weight_total_bytes`/`.top_sites.<site>.bytes`, `jdk.ExecutionSample.count`/
`.top_methods.<method>.samples`, `jdk.JavaMonitorEnter.count`/`.total_ms` and
`jdk.ThreadPark.count`/`.total_ms`. Every key is written on every run, zero or not, so a consumer
never has to distinguish "absent" from "none".

Two decisions worth recording. `jdk.ThreadAllocationStatistics#allocated` is a running per-thread
total, not a per-event delta, so `bytes_total` is the sum over threads of each thread largest
sample. Summing the events instead multiplies the figure by roughly the number of samples per
thread, which would make a bytes-per-token ceiling read as passing when it is not; there is a test
for exactly this. And the two attribution histograms are bounded to ten entries: they exist to point
a reader at the next thing to look at, and unbounded they would put one JSON key per distinct method
into every published result file.

**Tests, written before the implementation.** `JfrMetricsExtractorJdkEventsTest`, 8 cases in the
`metrics` module. Six failed on the original code for the right reason — the `jdk.*` keys did not
exist, so every lookup returned null — and the two that passed were the bounding case (vacuously,
with no keys to bound) and a control asserting the `juno.*` metrics are undisturbed.

The sampled and periodic JDK events cannot be synthesised the way the `juno.*` events are: the JVM
emits them, and how many land in a given window is timing-dependent. So every value assertion
re-reads the same recording with `RecordingFile` and computes what the extractor should have
produced from it. The assertions are exact whether the window caught three events or three hundred,
and they still hold at zero. Run five times in a row: 8 of 8 each time.

**Verified on a real run**, not only in tests. `./juno local --cpu --jfr 2m` on tinyllama produced,
among others, a 657 ms maximum GC pause and a per-site allocation breakdown naming
`GgufReader.tensorRaw` and `LlamaTransformerHandler.sgemmQ4KWeightStationary` as the top two
allocators. The README motivates this rule with a historical 622 ms pause that contaminated a short
JFR-window measurement and read as a regression; the first run under the new configuration produced
a pause of the same order. The tooling is load-bearing immediately, not eventually.

**Prompt-token parity.** `RAW_PROMPT` now defaults to `1`, with a new `--no-raw-prompt` to turn it
off for a Juno-only measurement. Every Juno result JSON records `n_prompt` alongside the real
`prompt_tokens` and the resulting `prompt_token_deviation`. `write_pair_summary` withholds the
prefill ratio — emits `null` plus a `prompt_parity` object stating the reason — when the deviation
exceeds the 10% tolerance, and `INDEX.md` prints `withheld` in that cell, lists the reason, and
carries a new Juno-actual/requested prompt-token column. The generation ratio is unaffected and
still published, since prompt length does not bear on it.

Exercised both ways with fixtures: at 124 tokens against `n_prompt` 128 (3.1% off) the ratio is
published; at 20 against 128 — the shape of every previously published run, taken with
`raw_prompt: 0` — it is withheld, reading `juno prefilled 20 tokens against n_prompt 128: 84% off,
over the 10% tolerance`. One bug was found and fixed during that check: the INDEX reason list used
`.prompt_parity.ok // true`, and jq treats `false` as absent for `//`, so the reason would never
have printed.

**GC and allocation in every published run.** `jfr_summary_json` now surfaces `gc_pause_count`,
`gc_pause_max_ms`, `gc_pause_total_ms`, `allocated_bytes_total`, `allocated_bytes_per_token`,
`execution_sample_count`, ranked `top_methods` and `top_allocation_sites`, `monitor_enter_total_ms`
and `thread_park_total_ms`. `INDEX.md` gained GC-max and allocated-bytes-per-token columns and a
note that a run whose GC maximum is a large fraction of its window should be re-run rather than
scored.

**Thread-count mismatch, recorded rather than removed.** Juno still has no thread-count control
reaching the hot path; that arrives with the CPU hot-path tier. `JUNO_EFFECTIVE_PARALLELISM` (the
common pool default, one fewer than the available processors) is now recorded in every run metadata
block, and every published INDEX states the mismatch against the reference tool `-t` explicitly.

**Not done in this pass**, and blocking exit criteria 3 and 4:

- Preconditions 2, 3, 5 and 6 from the README list: `--juno-warmup N` (default 2), `--juno-reps N`
  (default 3, median reported, min/max recorded, reusing the `compare-lora.sh` implementation), a
  fixed per-model `COMPARE_HEAP`, and CPU governor plus GPU clock state captured into run metadata.
- The corrected re-baseline run itself, and therefore the re-derivation of the program target GPU
  prefill row. Both `llama-bench` binaries this host needs are present
  (`../llama.cpp/build-cuda/bin` for GPU, `../llama.cpp-bin/llama-b9551` for CPU), so this is
  runnable here; it was not started because it is a long run whose result this tier is scored
  against, and it must be taken after the remaining preconditions land, not across them.

**Performance gate: not run, and not required for this pass.** Nothing here touches the forward
pass, MatVec, GPU residency, batching, KV or quantization. The only change reaching a running
inference process is the JFR configuration itself, which alters what is recorded, not what is
computed. That change does break strict comparability with everything already published under
`docs/perf-compare/`, exactly as the README anticipates: the first run taken under the new
configuration is the new reference, and no number in this tier may be scored across that boundary.

**Verification commands and results.**

- `mvn -o test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,vision,metrics,juno-player`
  on the final source: BUILD SUCCESS, all eleven modules, **1582 tests, 0 failures, 0 errors**
  (`node` 578/41 skipped, `coordinator` 294/1, `juno-player` 104/2, `vision` 95, `registry` 93,
  `tokenizer` 89/2, `kvcache` 78, `sampler` 66, `metrics` 46, `health` 23, `lora` 116). The skips are
  the GPU-, ROCm- and missing-model-gated tests that were already skipped. `metrics` is 46 where it
  was 38, the difference being this tier's eight new cases. The `SEVERE FAILED to load real model`
  lines in the log are the previous tier's intentional fail-closed logging inside passing tests.
- `mvn -o clean verify -pl juno-master` (after `mvn -o install -DskipTests`): BUILD SUCCESS, **20
  tests, 0 failures, 0 errors** — `ThreeNodeClusterIT` 8, `InProcessClusterIT` 6,
  `TensorParallelClusterIT` 5, `UnsupportedArchitectureClusterIT` 1. Run because this pass changes
  the flag `ClusterHarness` builds for each forked node JVM, and these are the tests that fork them.
  A first attempt without `clean` reported three errors reading `ClusterHarness cannot be resolved`
  and `NoClassDefFoundError`; that was stale IDE output, not a real failure. `juno-master`'s
  `target/test-classes` predated this session and carried `java.lang.Error: Unresolved compilation
  problems` inside the class files, which only the Eclipse compiler emits, and the resolved
  `juno-player` artifact predated these changes. Both the installed and the freshly built jar
  contain `ClusterHarness`. Cleaning and installing the current artifacts first resolves it; the
  finding is recorded here because the same trap will catch the next tier that runs `verify` without
  `clean`.
- `JfrMetricsExtractorJdkEventsTest` alone, five consecutive runs: 8 of 8 each time. Run because this
  plan already carries one timing-dependent `metrics` test and these cases read real JVM events.
- `./juno local --cpu --jfr 2m` on tinyllama from a directory outside the repository, and the same
  through a bare `java -jar` on the shaded jar: both resolved the settings without warning, the first
  from the working copy by way of the launcher, the second from the packaged copy. The resulting
  `metrics.json` carried every `jdk.*` key.
- The forked-JVM resolution exercised directly from outside the repository with only the packaged
  copy on the classpath: it materializes a readable temporary file whose content is the real
  configuration, which is what `ClusterHarness` hands each node as `settings=`.
- `jfr_summary_json` and `write_pair_summary` exercised against the real `metrics.json` above and
  against in-parity and out-of-parity fixtures; `write_run_index` exercised on both.

**Docs updated for this pass** (the tier's docs exit criterion stays unticked, since it also covers
the residency work that has not started): `docs/howto.md` gains the `jdk.*` metric table and a
section on the single recording configuration and how to override it; `docs/agent-arch.txt` gains
the `metrics` module entry it never had, including the per-thread-cumulative allocation rule and the
settings resolution order; `docs/performance.md` gains a measurement-boundary note at the top, so a
reader comparing an entry recorded before the change against one recorded after knows there is a
boundary between them.

**Cross-surface checklist: deferred to the residency work.** The rows in this tier checklist are
about residency, and no residency code exists yet. The change in this pass reaches every surface
that starts a recording — local, LoRA, cluster coordinator and forked cluster nodes, on both
launcher platforms — and all of those resolve the same settings file or say why they could not.

### 2026-09-24 — benchmark-parity preconditions, part 2 (warmup, repetitions, fixed heap, clock state)

Scope of this pass: the rest of exit criterion 4's harness work — README preconditions 2, 3, 5 and 6
(`--juno-warmup`, `--juno-reps` with a median and a published spread, a fixed per-model heap, and
captured CPU/GPU clock state). No residency code was written and no residency number was measured.
The re-baseline run itself was deliberately not taken; see "Not done in this pass" below.

**Plan-versus-code drift found before starting.**

- HEAD is `c91f879`. Tier 00 is complete and its exit criteria are all ticked, so this tier is the
  active one. Tier 00's claims were re-verified rather than trusted: `generateBatch()` no longer
  touches the prefix trie (it carries an explicit "No cachePrefix call" comment where the write used
  to be), `ForwardPassHandlerLoader` routes through `LlamaFamilyArchitectures.requireVerified` and
  rejects anything unverified, a repo-wide grep finds no `com.hazelcast` or `RegistryService` symbol
  in any source or pom, the grep for internal tier numbers over every `src/main` tree is clean, and
  `CLAUDE.md` lists `vision` and `metrics` in its test command. `TensorShardContext` is still
  referenced only by its own test and by a `ClusterHarness` comment, and `FaultTolerantPipeline` is
  still reachable only through `HealthReactor`, both as Tier 00 documented them. All hold.
- **[`INVENTORY.md`](INVENTORY.md) understates what is on disk.** It lists "a working `qwen3moe`
  GGUF" and "a plain `qwen3` GGUF" as models Tier 08 needs and does not have. Both are present:
  `Qwen3-1.7B-Q4_K_M.gguf` reports `general.architecture=qwen3` and
  `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` reports `qwen3moe`, read from the file headers. The
  inventory is corrected. This changes nothing for this tier; it removes two of Tier 08's four
  stated model gaps.
- **Precondition 1, ticked in the previous pass, did not achieve parity in practice.** Prompt-token
  parity was implemented and verified against fixtures at a 3.1% deviation, but on a real run the
  deviation is structural rather than incidental: the chat template wraps every request in role and
  control tokens, about 19 of them on TinyLlama, and the raw-prompt mode counted words without them.
  Measured here — a 32-word prompt prefilled 50 tokens (56% over) and a 128-word prompt prefilled
  146 (14% over). Both exceed the 10% tolerance, so the withholding rule the previous pass added
  would have correctly suppressed **every prefill ratio in the sweep**, and the one column this
  program's target turns on would have gone unpublished. The rule was right; the prompt was wrong.
  Fixed in this pass (see below). The criterion stays ticked, now for a harness that reaches parity
  rather than one that only refuses to lie about missing it.

**What shipped.**

| Item | Where |
|---|---|
| `--juno-warmup N` (default 2) discarded requests before the measured one | `compare-llama-cpp.sh` |
| `--juno-reps N` (default 3) measured cycles, median published, min/max recorded | `compare-llama-cpp.sh` |
| Recording scoped to the measured request, started and stopped by the harness | `compare-llama-cpp.sh` |
| Extraction of one named recording to one named output | `JfrMetricsCli` (new, `metrics`) |
| Prompt-length calibration against the real `prompt_tokens` | `compare-llama-cpp.sh` |
| Fixed per-model heap, with off-table results labelled `derived` | `compare-llama-cpp.sh` |
| CPU governor, turbo state, GPU clocks and active throttle reasons in run metadata | `compare-llama-cpp.sh` |
| Median-versus-spread column and clock state in the published index | `compare-llama-cpp.sh` |
| `--selftest` over the aggregation arithmetic | `compare-llama-cpp.sh` |
| `jcmd` and settings-file checks before a model is loaded | `compare-llama-cpp.sh` |

**Warmup is what forced the recording to change hands, and this is the load-bearing decision in the
pass.** The engine's `--jfr` recording runs for the whole process lifetime. Warmup requests have to
run in the same process as the measured one — that is the point of them, since what they buy is
compiled code — so under the engine's own recording the discarded requests would land in the same
window as the measured one. That is not a small contamination: `juno.TokenProduced.tps` is computed
across the span from the first token in the recording to the last, so two warmup requests plus a
measured one would divide the measured token count by a span that also contains the warmups and the
idle gaps between them. Every aggregate the published figures are read off has the same shape.

So the harness now owns the window: the engine is launched without `--jfr`, the warmup requests are
issued and allowed to return, `jcmd JFR.start` names `juno-perf.jfc`, the measured request runs,
`jcmd JFR.stop` dumps, and `JfrMetricsCli` extracts. The settings file is the same one every other
recording site names, so the instrumentation overhead is unchanged and results stay comparable with
the previous pass's reference. Verified on a real process rather than reasoned about: with two
warmups and an 8-token measured request the recording holds **8** `juno.TokenProduced` events and
one `juno.PrefillBatch`, not 24 and 3. `--jfr DURATION` keeps a meaning — it is now an upper bound
on the window, and a recording that hits the bound still dumps to the file named at start, so the
flag does not silently no-op.

`JfrMetricsCli` is a new class rather than another entry point on `MetricsMain`: the two existing
ones serve the running engine (scan the working directory, map against `models.json`, or extract
programmatically at shutdown) and both write to a fixed relative path, so a caller can only steer
the output by changing its working directory, and the scanning form additionally requires the model
to be listed in `models.json`. It fails on a missing or empty recording instead of writing a report
of zeroes — every metric is written on every run whether its event fired or not, so a zero is a real
reading, and a zero meaning "no recording" would read as a clean fast run.

**On "reuse `compare-lora.sh`'s existing `--reps`/median implementation rather than writing a second
one".** The two comparison scripts share no library — neither sources `perf-lib.sh` — so literal reuse
would have meant extracting one, which is a refactor of the script that carries the standing
performance gate and is not this pass's scope. The precondition's intent is that a second, different
median semantics must not appear, and that is honoured: the jq `median` definition is the same
definition `compare-lora.sh` uses, character for character. It is extended with min, max and the
per-rep value list, which `compare-lora.sh` does not record and which the README's noise-floor rule
requires. If a later tier extracts a shared aggregation helper, these are the two call sites.

**Two aggregation decisions worth recording.** A repetition is a whole engine cycle, not another
request in the same process, because model load, page-cache state and device residency all sit
inside a cycle and a repetition that reused them would re-measure only the cheapest part of the run;
this matches the repetition discipline `compare-lora.sh` already follows. And the collection-pause
maximum and the allocation-per-token figure are aggregated across reps by their **worst** reading
rather than their median: the re-run rule fires on any single pause over its threshold, so a median
would hide precisely the outlier the rule exists to catch. Both figures also keep a full median,
min, max and per-rep value list in the result JSON.

**The heap is fixed per sweep model** (`tinyllama` 4g, `qwen2.5-3b` 5g, `Phi-3.5-mini` 6g,
`mistral-7b` 9g, `llama-1-30b` 30g, `tinyllama` Q2_K 4g) — the values the size derivation produced
when it was replaced, so the table starts from what the published runs were already taken at. A
model outside the table keeps the derivation rather than failing, because an off-table model still
has to be runnable, but its result records `heap_source: derived` and a derived heap is not
comparable with a baseline taken at a fixed one. `COMPARE_HEAP` still wins and records `explicit`.

**Tests, written before the implementation.** `JfrMetricsCliTest`, 6 cases in the `metrics` module,
covering the named-output path, the `jdk.*` accounting surviving into it, stem derivation, and the
three fail-closed cases (missing recording, empty recording, missing arguments). All six failed to
compile against the original tree because the class did not exist, which is the right failure; 6 of
6 pass now. The script's own arithmetic is covered by `--selftest`, 24 checks, which needs only `jq`
and runs no model: median over odd and even rep counts, min/max recorded, null readings dropped
rather than read as zero, an all-null aggregate staying null, a single failed rep failing the
aggregate, the worst-reading rule for the noise indicators, and the three heap sources. It was
written first and failed on the missing function. One check failed on first run because the
expectation was written wrong (`6` against jq's `6.0` for a selected element), not the code.

**Verification commands and results.**

- `./scripts/performance-tests/compare-llama-cpp.sh --selftest`: 24 of 24 checks pass.
- `mvn -o test -pl metrics -Dtest=JfrMetricsCliTest`: 6 tests, 0 failures.
- A real CPU cycle end to end (`--cpu --models tinyllama --n-prompt 128 --n-gen 8 --juno-warmup 1
  --juno-reps 2 --reps 1 --no-publish`): failures=0. The calibration probe prefilled 146 tokens
  against `n_prompt` 128, the calibrated 110-word prompt prefilled **exactly 128**, deviation 0, and
  the prefill ratio is published rather than withheld — the first like-for-like prefill reading this
  harness can produce. Both reps recorded `jfr_scoped_to_measured_request: 1` and
  `heap_source: fixed`.
- The same on GPU, which also exercises the automatically-added tuned lane and therefore the
  aggregation through `run_tuned_lane` (`--gpu --models tinyllama --n-prompt 128 --n-gen 16
  --juno-warmup 1 --juno-reps 2 --reps 1 --no-publish`): failures=0, both lanes at deviation 0 with
  scoped recordings, published pp and tg ratios, and the spread column populated.
- `mvn -o test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,vision,metrics,juno-player`,
  read from the surefire reports rather than from console scrollback: `tokenizer` 89/2 skipped, `lora`
  116, `node` 597/41 skipped, `coordinator` 294/1, `sampler` 66, `kvcache` 78, `health` 23, `registry`
  93, `vision` 95 — **0 failures and 0 errors in all nine**. The skips are the GPU-, ROCm- and
  missing-model-gated tests that were already skipped; `node` is 597 where the previous pass recorded
  578, the difference being tests added by the out-of-tier commits recorded below. Consolidated across
  all eleven modules, including the two re-run below: **1607 tests, 0 failures, 0 errors, 46 skipped**,
  and no surefire report on disk carries a failure.
- **The pre-existing flaky `metrics` test halted that reactor**, so `metrics` and `juno-player` were
  re-run on their own: `mvn -o test -pl metrics,juno-player` → `metrics` **52 tests, 0 failures**
  (46 from the previous pass plus this pass's 6 `JfrMetricsCliTest` cases), with
  `JfrMetricsExtractorAttentionTest` passing 5 of 5, and `juno-player` **104 tests, 0 failures, 2
  skipped** — the same figures the previous pass recorded. BUILD SUCCESS.
- **A four-hour test run was recorded here and the cause was self-inflicted; the figure is wrong.**
  `mvn -o test -pl metrics,juno-player` reported `Total time: 04:09 h`, and this record first
  attributed that to the real-model LoRA probe tests being inherently slow. They are not: the final
  verification run of all eleven modules completed in **24:55 min**, with `juno-player` a small part
  of it. The 4-hour run had been started while an earlier full-reactor run was still going, because
  its exit code had been read from a shell pipeline whose last stage was a `grep` — so the exit code
  described the grep, not Maven. Two Maven reactors then forked test JVMs against the same twelve
  threads. The lesson worth keeping is the one about the exit code, not one about the test suite:
  read a Maven result from `BUILD SUCCESS` or from the surefire reports, never from the status of a
  pipeline it was piped into, and check that a previous run has actually exited before starting
  another. A `timeout` around the command is also not a safety net — the 1800-second one used did not
  end the run, because the reactor waited on its forked test JVM long past the signal.
  The failure in the first run was
  `JfrMetricsExtractorAttentionTest.mixedPrefillAndDecode_aggregatesIndependently`, asserting exact
  double equality of `total_ms` against `prefill_ms + decode_ms` over real measured durations —
  `expected: 0.0018050000000000002 but was: 0.001805`. This is the same test, the same assertion and
  the same failure mode Tier 00 recorded and deliberately left alone, where it failed 3 of 10 runs in
  a modified tree and 1 of 10 in an untouched one. Nothing in this pass touches `juno.Attention`
  events or that code path, and the test passed on re-run.
  **Worth a decision by the owner rather than another sighting:** floating-point summation is not
  associative, so the assertion is wrong rather than the code, and it intermittently stops the
  documented full-`mvn test` reactor before its last two modules. Either a tolerance on that one
  assertion or `-Dsurefire.rerunFailingTestsCount=2` in the build (the previous pass passed that flag
  by hand) would settle it. Left unchanged here because Tier 00 took that decision explicitly and
  this pass owns neither the test nor the build configuration.

- `mvn verify -pl juno-master` was **not** run this pass, and is not needed by it. The previous pass ran
  it because it changed the flag `ClusterHarness` builds for each forked node JVM; this pass touches
  neither `ClusterHarness` nor any launcher, and its only `src/main` change is a `metrics` entry point
  no cluster path calls. The tier's full-verify exit criterion is unticked regardless, and the
  residency work will need it.

These runs were taken with deliberately short settings (`--n-gen` 8 and 16, two reps, one warmup) to
exercise the harness, and are **not** baselines. None is published under `docs/perf-compare/` and no
number from them may be scored.

**Performance gate: not run, and not required for this pass.** Nothing here touches the forward
pass, MatVec, GPU residency, batching, KV or quantization. The only `src/main` change is a new
extraction entry point in the `metrics` module, which no inference path calls. The changes that
reach a measured process are the absence of the engine's own `--jfr` flag and the harness-owned
recording, which alter what is recorded and when, not what is computed — and both sides of that
change use the same settings file, so the instrumentation overhead is the same.

**This pass is itself a measurement boundary**, and a larger one than the previous pass. A Juno
reading taken after it is warm, is the median of three cycles, sits at a fixed heap, and prefills
the requested token count; a reading taken before it is none of those. Prefill ratios published
before it read as better than a like-for-like measurement supports, and generation ratios published
before it read as worse, because they were cold. No number in this tier may be scored across it.
`docs/performance.md`'s measurement-boundary section records both passes.

**A seventh parity asymmetry, found by taking the re-baseline: generation length.** The README's
preconditions list six asymmetries. There is a seventh, and the first re-baseline run is what exposed
it. `llama-bench` generates `n_gen` tokens whatever the model would rather do; Juno stops at a stop
token. Measured on the GPU sweep at `n_gen` 64:

| Model | Juno tokens generated | Reference | Finish reason |
|---|---|---|---|
| `Phi-3.5-mini` | 64 | 64 | `length` |
| `mistral-7b` | 49 | 64 | `stop` |
| `tinyllama` | 22 | 64 | `stop` |
| `qwen2.5-3b` | **0** | 64 | `stop` |

Two distinct problems, with different severities. `qwen2.5-3b` emits a stop token immediately when
handed the calibrated minimal-token prompt, on all three repetitions, so there is **no generation
reading for it at all** — and the harness published that as a ratio of `0`, which reads as "infinitely
slower than the reference" when it means "never measured". That is the same class of error as an
unparity-corrected prefill ratio, and worse, because a zero looks like data. The harness now withholds
the generation ratio when nothing was generated, states the reason and the finish reason, and prints
`withheld` in the index, with `generation_parity` in every comparison JSON and a generated-token
actual/requested column beside the prompt-token one. Covered by two `--selftest` cases.

The second problem is milder and is reported rather than withheld: `tinyllama` at 22 tokens and
`mistral-7b` at 49 are real readings, but they are averages over a shorter span of context than the
reference's 64, and decode gets slower as the sequence grows, so those ratios are mildly flattered.
The shortfall is now stated per model in the index. Withholding them would empty the column for three
of four models on the strength of an effect smaller than this host's measurement floor; stating them
is the proportionate response.

**Closed by `min_tokens`, approved by the owner and shipped in this pass.** True generation parity
means Juno generating `n_gen` tokens regardless of a stop token, the way the reference tool does. That
is a public API addition, so under this plan's own feature-complete definition it had to land in
`openapi.yaml`, `juno-api.yaml` and the sampler together, and it did:

| Surface | How the minimum arrives |
|---|---|
| Sampler | `MinTokenFloor` (new class) suppresses the end-of-sequence token below the minimum |
| Sampling parameters | `SamplingParams.minTokens`, validated against `maxTokens` |
| Single request | `GenerationLoop.generate`, including the speculative verify site |
| Static batch | `GenerationLoop.generateBatch`, one floor per request rather than per batch |
| Continuous | `ContinuousBatchEngine` slot state, one floor per slot |
| Chat completions | `min_tokens` |
| Native inference | `sampling.minTokens`, both blocking and streaming |
| Contracts | `openapi.yaml`, `juno-api.yaml` |
| Embedding facade | `JunoHttpClient.blockingInference`/`blockingOpenAiChat` overloads |
| Benchmark harness | `--juno-min-tokens`, defaulting to the requested generation length |

The floor masks rather than ignores: an ignored end-of-sequence token still has text, so every
generation path would have to decide separately not to emit it, and the model would keep proposing it
every step. Masking lets the model fall through to its next-best continuation. It yields in exactly one
case — where a grammar has already reduced the legal set to end-of-sequence alone, masking it too would
leave every logit at negative infinity and a softmax over that is not a distribution, so the sequence
is allowed to end below its minimum rather than becoming unsamplable. A minimum above the maximum is
rejected rather than clamped, on both surfaces.

Verified on the model that exposed the problem: `qwen2.5-3b` on GPU at `n_prompt` 128 and `n_gen` 32
now reports **32 of 32 tokens generated**, `finish_reason: length`, prompt deviation 0, and a published
generation ratio of 0.384 where before there was no reading at all.

**A pre-existing defect found while doing it.** The native inference surface did not map a rejected
sampling value to a bad-request response. `SamplingParams` validates its own ranges and throws, and
neither native route caught it, so an out-of-range temperature — or the new minimum above the maximum —
returned a server error, telling the caller Juno had broken rather than that the request was wrong. The
chat surface already mapped these to 400. Both native routes now do, covered by
`NativeSamplingRejectionTest` including the temperature case, so the fix is demonstrably not specific
to the field that exposed it.

**Not done in this pass**, and still blocking exit criteria 3 and 4:

- The corrected re-baseline run itself, and therefore the re-derivation of the program target's GPU
  prefill row. Every precondition it depends on is now in, and both `llama-bench` binaries are
  present, so it is runnable here. It was not started for two reasons: it is the long run this tier
  is scored against and it should be taken deliberately rather than at the end of a pass that
  changed the harness under it, and the prompt calibration above changes what every published
  prefill ratio will read, which the owner should see before ten tiers are scored against the
  result. The command is
  `./scripts/performance-tests/compare-llama-cpp.sh --gpu --reps 3 --juno-reps 3 --juno-warmup 2`
  followed by the same with `--cpu`, both publishing under `docs/perf-compare/`.

### 2026-09-25 — the parity-corrected re-baseline, and what taking it exposed

Published: [`../perf-compare/20260925T053847Z/`](../perf-compare/20260925T053847Z/) (GPU, eight lanes)
and [`../perf-compare/20260925T055156Z/`](../perf-compare/20260925T055156Z/) (CPU, four lanes). Both
`failures=0`. Taken on `c91f879` plus this tier's uncommitted harness and `min_tokens` work.

**Both parity columns are now exact.** Every lane in both sweeps reports `128/128` prompt tokens and
`64/64` generated tokens, `finish_reason: length`, and no withheld ratio. This is the first sweep in
the repository where the two engines demonstrably did the same amount of work in both directions.

| Model (GPU) | Juno/ref pp | Juno/ref tg | Juno tg median (min/max) |
|---|---|---|---|
| `tinyllama` | 0.0400 | 0.325 | 62.71 (60.86 / 63.21) |
| `tinyllama` tuned | 0.0433 | 0.324 | 62.45 (62.32 / 63.00) |
| `qwen2.5-3b` | 0.0458 | 0.395 | 28.30 (22.35 / 28.56) |
| `qwen2.5-3b` tuned | 0.0471 | 0.394 | 28.22 (28.14 / 28.38) |
| `Phi-3.5-mini` | 0.0369 | 0.205 | 12.38 (11.08 / 12.40) |
| `Phi-3.5-mini` tuned | 0.0371 | 0.204 | 12.30 (12.25 / 12.49) |
| `mistral-7b` | 0.0709 | **0.558** | 20.62 (20.47 / 20.72) |
| `mistral-7b` tuned | 0.0695 | 0.549 | 20.29 (20.19 / 20.38) |

| Model (CPU) | Juno/ref pp | Juno/ref tg | Scorable |
|---|---|---|---|
| `tinyllama` | 0.0827 | 0.0900 | **no — see the collection-pause finding below** |
| `qwen2.5-3b` | 0.0753 | 0.0853 | yes |
| `Phi-3.5-mini` | 0.0482 | 0.0827 | yes |
| `mistral-7b` | 0.0739 | 0.0844 | yes |

**These numbers are not yet the reference, and the GPU prefill row is not yet re-derived, because
taking the sweep exposed an eighth asymmetry that invalidates the generation column as a
like-for-like comparison.** The reference tool runs prefill and generation as two separate
benchmarks: its prefill row is `n_prompt=128, n_gen=0` and its generation row is **`n_prompt=0`,
`n_gen=64`** — generation from an empty context, positions 0 through 64. Juno is measured in one
request, so its generation runs immediately after the 128-token prefill, at positions 128 through
192. Decode cost grows with context depth, so Juno's generation figure is taken two to three times
deeper into the sequence than the figure it is divided by.

This is visible in the numbers and should not be mistaken for a regression. `Phi-3.5-mini` read
0.330x before prompt-token parity landed, when Juno prefilled 20 to 30 tokens and therefore decoded
at a context close to the reference's; it reads 0.205x now that Juno prefills 128. Nothing about the
forward pass changed between those two readings. **Fixing prompt-token parity is what created this
asymmetry** — the two parity requirements cannot both hold inside a single request, which is exactly
why the reference tool uses two.

The fix is to mirror that structure: measure Juno twice per model, once with the calibrated
128-token prompt for the prefill figure and once with a minimal prompt for the generation figure.
That is a harness change of the same size as the warmup work, plus a third sweep. It is not started
here, because the owner should decide whether to spend it now or carry the asymmetry as a documented
caveat, and because it changes every generation ratio this plan will be scored against.

**Collection-pause rule: now machine-applied, and it fires on one published row.** The README states
the rule and Tier 01 was supposed to make it readable, but nothing applied it — the pass that added
the GC column left the judgement to a reader noticing a large number. `write_pair_summary` now emits
a `noise` object and `INDEX.md` carries a `Scorable` column reading `NOISY` with the reason. On the
CPU sweep, `tinyllama` repetition 1 took a **631 ms** pause inside a 40,164 ms window, against a
ceiling of 200 ms, so that row is not scorable and is marked as such above.

Worth recording what the repetition discipline did here: the three readings were 3.278, 3.219 and
3.280 t/s, a 1.9% spread, and the contaminated repetition supplied the median. So the pause cost
about 2% on a figure whose own spread is 2% — the damage was bounded by taking three repetitions, not
by noticing the pause. That is the case for medians-of-three independently of the marker. The two
published runs predate the marker, so their `INDEX.md` files do not carry the column; the rows above
are the authoritative reading of it.

**The lock-and-park half of the noise rule is not usable as a gate, and is not applied.** The README
sets it at `jdk.JavaMonitorEnter.total_ms` plus `jdk.ThreadPark.total_ms` over 10% of wall time.
Measured on a clean GPU run: `qwen2.5-3b` park time is 11,458 ms against a 4,241 ms request, **about
2.7x wall time**, with monitor time at zero, on all three repetitions of a run whose readings agree
to within 1%. The park figure is a sum across every thread, so an idle worker pool parks for longer
than the run takes however healthy it is. As written the condition fires on every run and
discriminates nothing. The figures are published in every result JSON; the gate is the collection
pause only. The README is corrected accordingly.

### 2026-09-25 — the reference re-baseline, taken with two Juno measurement lanes

**This is the new reference.** Published:
[`../perf-compare/20260925T172231Z/`](../perf-compare/20260925T172231Z/) (GPU, eight lanes) and
[`../perf-compare/20260925T174146Z/`](../perf-compare/20260925T174146Z/) (CPU, four lanes). Both
`failures=0`, every row at `128/128` prompt tokens and `64/64` generated tokens.

The two earlier sweeps of the same day
([`20260925T053847Z`](../perf-compare/20260925T053847Z/), CPU
[`20260925T055156Z`](../perf-compare/20260925T055156Z/)) are kept and are **superseded**: they measured
prefill and generation in one request and therefore understated generation. They remain valid against
each other and are the evidence for the size of that error.

**The lane split, and how much it was worth.** The owner approved measuring Juno the way the reference
tool measures itself — prefill and generation as two runs, not one. On `Phi-3.5-mini` the generation
figure moved from 12.38 to 24.42 t/s, and its ratio from 0.205x to 0.423x. The single-request shape was
costing roughly **half** the generation reading, far more than the "few percent, below the measurement
floor" this record estimated before measuring it. Decode at positions 128 through 192 is much more
expensive than at 0 through 64 on this hardware than that estimate assumed. The estimate was wrong and
only the measurement settled it.

| GPU, scorable | Juno/ref pp | Juno/ref tg | Juno tg median (min/max) |
|---|---|---|---|
| `qwen2.5-3b` | 0.0495 | 0.431 | 27.83 (25.62 / 29.44) |
| `qwen2.5-3b` tuned | 0.0497 | 0.400 | 25.79 (21.00 / 29.12) |
| `Phi-3.5-mini` | 0.0377 | **0.423** | 24.42 (24.32 / 24.55) |
| `Phi-3.5-mini` tuned | 0.0362 | 0.408 | 23.56 (23.22 / 23.64) |
| `mistral-7b` | 0.0709 | **0.581** | 19.98 (19.88 / 20.27) |

| GPU, marked NOISY | Juno/ref pp | Juno/ref tg | GC max |
|---|---|---|---|
| `tinyllama` | 0.0394 | 0.325 | 634 ms |
| `tinyllama` tuned | 0.0401 | 0.330 | 633 ms |
| `mistral-7b` tuned | 0.0727 | 0.631 | 636 ms |

| CPU, all scorable | Juno/ref pp | Juno/ref tg |
|---|---|---|
| `tinyllama` | 0.0847 | 0.128 |
| `qwen2.5-3b` | 0.0752 | 0.0790 |
| `Phi-3.5-mini` | 0.0494 | 0.123 |
| `mistral-7b` | 0.0741 | 0.0849 |

**The three NOISY rows are scored anyway, and here is why** (the README allows this provided the reason
is stated rather than the marker left hanging). In each case one repetition of three carried a pause of
about 635 ms, and in each case the reading from that repetition agrees with its siblings: `tinyllama`
generation read 56.82, 56.42 and 56.82 t/s — a 0.7% spread — with the pause on the middle one, and
`mistral-7b` tuned read 21.67, 21.24 and 21.93 with the pause on the **fastest** of the three. A pause
that cost nothing cannot have contaminated the figure.

**This record first explained that as the pause falling outside the measured token span. That
explanation was checked afterwards and is wrong** — see the correction under "The pause counter does
not measure stopped time" below. The pauses are inside the span. What is wrong is the counter, not the
attribution.

### Program target: the GPU prefill row, re-derived (exit criterion 3)

The README set `GPU pp >= 0.15x` provisionally, reasoning from the best prefill number the project had
ever produced — about 0.028x — and calling the target a three- to four-fold improvement while noting the
figure was not like-for-like.

Measured like-for-like for the first time, the starting point is **better** than that: 0.036x to 0.073x
across the sweep, median about 0.045x, with `mistral-7b` best at 0.071x and `Phi-3.5-mini` worst at
0.036x.

**The target stays at 0.15x**, for the plain reason that it is now a less demanding target than when it
was written, not a more demanding one. Against 0.028x it asked for 5.4x. Against the real median of
0.045x it asks for **3.3x**, and against the best model **2.1x**. Lowering a target because the
measurement improved would be reading the improvement backwards.

Two qualifications a later tier must carry. The spread across models is wide enough that "every sweep
model" makes `Phi-3.5-mini` the binding constraint at 4.2x, while `mistral-7b` needs 2.1x; a tier that
reports "target met" on one model has not met this target. And the readings above are **not comparable
with the pre-parity figures the other target rows were written against** — GPU tg for `Phi-3.5-mini` now
reads 0.423x where the table records 0.330x, and `mistral-7b` 0.581x where the table records 0.513x, but
almost none of that is the engine getting faster. It is the measurement stopping its penalty of Juno.
Whoever scores a later tier against those rows is scoring across a measurement boundary unless they
re-read them from here.

`mistral-7b` tuned at 0.631x is the first reading in this repository above the program's 0.60x generation
target for that model. It sits in the NOISY table, its three repetitions span 21.24 to 21.93, and it is
one model — it is a milestone worth noting and not a program target met.

### 2026-09-26 — three harness and test defects closed

**1. The flaky `metrics` test is fixed.** `JfrMetricsExtractorAttentionTest`'s
`mixedPrefillAndDecode_aggregatesIndependently` asserted that the prefill and decode totals add up to
the overall total using exact `double` equality. Both sides sum the same three measured durations but
group them differently, and floating-point addition is not associative, so the two could differ in the
last bit — `expected: 0.0018050000000000002 but was: 0.001805`. It failed about one run in three and
halted the documented eleven-module reactor before its last two modules three times in this tier's
execution. The assertion now permits a difference of 1e-9, which against values around 0.002 ms is
about four parts in ten million — far tighter than any real defect, since a miscounted event moves
either side by roughly a third. Run twelve times consecutively: twelve passes. Nothing else in the
repository asserts exact equality on a differently-grouped sum; the two sibling assertions compare
against a literal zero, which is exact.

**2. The pause counter does not measure stopped time, so the gate no longer reads it.** The intended
sharpening was to attribute each pause to the token span the generation figure is measured over, on the
theory that the 635 ms pauses landed in model load or prefill. That was built — `GcPauseSpans` (new
class in `metrics`) records each pause interval and answers count, maximum and total overlapping a
window, and `JfrMetricsExtractor` now emits `jdk.GCPhasePause.in_token_span.*` alongside the
whole-recording figures. Then it was checked against the real recordings, and the theory was wrong: the
pauses are **inside** the span.

The measurement that settles what is actually happening, from `tinyllama` tuned's three generation
repetitions:

| Repetition | Reported pause | Token span | Tokens | Rate |
|---|---|---|---|---|
| 1 | **633.2 ms** | 1110 ms | 64 | 57.6 t/s |
| 2 | 5.2 ms | 1120 ms | 64 | 57.1 t/s |
| 3 | 4.4 ms | 1107 ms | 64 | 57.8 t/s |

The spans are within 1% of each other whether the reported pause is 633 ms or 4 ms. A 633 ms
stop-the-world inside a 1110 ms span would leave 477 ms to produce 64 tokens, a rate of 134 t/s — more
than twice what this model reaches on this card. The duration therefore cannot be time the application
was stopped. Three occurrences across the sweep, all 633 to 636 ms, all with unaffected throughput.

So the gate now asks the question it actually cares about, and asks it of the repetitions: **a row is
scorable when its own generation readings agree.** The tolerance is 15% of the median, anchored to the
same host variability the README's noise floor is anchored to. This gets both cases right where the
pause rule got both wrong — it clears the three rows whose readings agree to within 1%, and it flags
`qwen2.5-3b` tuned, whose readings span 21.00 to 29.12 t/s, 31% of their median, and which the pause
rule passed as clean. A collection pause that does cost time still shows up, as one slow repetition.
Both pause figures, whole-recording and in-span, stay in every published result as context; neither
gates. Covered by `GcPauseSpansTest` (9 cases) and six harness selftest cases, including the two real
shapes above.

**3. The harness no longer leaves a process behind per engine launch.** Keeping the console REPL alive
requires stdin never to reach end of file, and that was done with `< <(while true; do sleep 3600; done)`
— a process substitution whose subshell nothing reaped. Every engine launch left one behind, and each
respawns a fresh `sleep` every hour, so they persist indefinitely: a check for leftovers found orphans
two days old, from Tier 00's smoke script, still spawning. Enough of them accumulated during this tier
that `pgrep` checks for "is a sweep still running" returned false positives, which cost real confusion
while stopping a sweep. `compare-llama-cpp.sh` now holds a named pipe open on a descriptor it owns and
releases both with the engine. Verified live: a three-repetition GPU run left **zero** leftover shells
and **zero** leftover pipes, where the same run previously left six.

**The same one-line pattern is in eight sibling scripts** — `smoke-tools.sh`, `smoke-grammar.sh`,
`smoke-tier00-consistency.sh`, `compare-vision.sh`, `compare-parallel.sh`, `compare-schedule.sh`,
`compare-prefill-batch.sh` and `compare-mixed-prefill.sh` — and they are **not** fixed here. The change
is mechanical, but none of them source `perf-lib.sh` today, so the fix is either eight separate edits or
eight scripts newly sourcing a shared helper, and several cannot be exercised without a GPU, real models
and a long run. Editing a smoke script that is another tier's exit-criteria evidence without being able
to run it is the wrong trade. `perf-lib.sh` is a safe host for the helper when someone takes them on: it
defines functions only, and its one assignment is guarded.

### Out-of-tier changes (recorded per execution rule 9)

Two commits touching hot-path or launcher behaviour landed while this tier was in progress and
outside its scope. Neither was in any tier's plan; both are recorded here because the value of this
plan tree is that it knows what was measured, when, and against which build.

| Commit | What it changed | Measurement boundary? |
|---|---|---|
| `1f90b68` | Both launchers (`scripts/run.sh`, `scripts/run.bat`) derive the JVM heap from the model file size instead of a fixed 4 GB, so a large model no longer dies with an `OutOfMemoryError` naming a tensor. `--heap` and `HEAP` still win. | **Yes, for launcher-driven runs only.** `compare-llama-cpp.sh` builds its own `java_args` and does not shell the launcher, so no llama.cpp ratio moved. Every `./juno`-driven measurement did, including `compare-lora.sh` and every smoke script. Do not compare a launcher-driven run taken before this commit against one taken after. |
| `c91f879` | Retired a device KV mirror by closing it in place rather than unmapping it, so an empty replacement is no longer read as history; gated attention on a written-prefix watermark, per mirror instead of per handler. Touched `DeviceKvCache`, `LlamaTransformerHandler`, `CudaMatVec`, `DeviceScratchBudget`, `Q4KDequantScratch`, with `DeviceKvMirrorWatermarkTest` and `DeviceScratchBudgetTest`. Published three `compare-lora.sh` runs (`docs/perf-compare/20260924T184429Z-lora/`, `20260924T185248Z-lora/`, `20260924T201548Z-lora/`). | **Correctness fix on the GPU attention path; gated and published.** The three runs compare a working tree against its own `HEAD` (`1f90b68`), which is why both columns name the same commit. Later tiers reading those directories should know that is deliberate, not a harness bug. |

Two consequences for later tiers, neither of which changes this tier's scope:

- **`DeviceScratchBudget` now exists in `src/main`.** [Tier 04C](TIER-04C-packed-weight-matmul.md)
  item 0 was written while it was an uncommitted working-tree file and says "if this shipped during
  Tier 01, this tier verifies and tests it rather than re-implementing it." It shipped in `c91f879`,
  not as Tier 01 scope. The instruction stands — Tier 04C verifies and tests it — but it should not
  expect to find it described anywhere in this tier's plan text.
- **The GPU attention path changed under [Tier 01B](TIER-01B-prefill-throughput.md).** That tier's
  item 0 re-measures per-architecture GPU-attention gains and characterises FP16 KV mirror
  divergence. Its step 1 re-baseline must be taken on `c91f879` or later, and its divergence
  characterisation must not quote any figure measured before it.

## Exit criteria

- [x] `scripts/performance-tests/juno-perf.jfc` exists, all four JFR recording sites name it, and
      `JfrMetricsExtractor` emits the `jdk.*` metrics the README specifies, with `metrics` tests. Until
      this is checked, no later tier can satisfy the GC/allocation recording rule.
      *Six sites exist, not four; five now name the file and the sixth (the AWS deployment script) is
      deliberately excluded — see the execution record. `JdkEventBucket` plus
      `JfrMetricsExtractorJdkEventsTest` (8 cases).*
- [x] `RAW_PROMPT` defaults to `1`, result JSONs carry Juno's actual `prompt_tokens`, and the
      publish path refuses a ratio whose `prompt_tokens` differs from `n_prompt` by more than 10%.
      *Verified both ways with fixtures: published at 3.1% deviation, withheld with a stated reason
      at 84%. The generation ratio is unaffected. Extended in the second pass: the rule was correct
      but the prompt could not satisfy it, because the chat template adds about 19 tokens that the
      raw-prompt word count did not account for, so a 128-word prompt prefilled 146 tokens and every
      sweep ratio would have been withheld. The word count is now calibrated against the token count
      the engine reports, which lands on `n_prompt` exactly on TinyLlama.*
- [x] The program target's GPU pp row re-derived against the parity-corrected re-baseline and
      restated in this file, with the number and the reasoning.
      *Kept at 0.15x, restated against a real like-for-like starting point of 0.036x to 0.073x
      (median 0.045x): the target asked for 5.4x against the old 0.028x figure and asks for 3.3x
      against the median now, so it became more reachable rather than less, and lowering it would read
      the measurement improvement backwards. `Phi-3.5-mini` at 0.036x is the binding constraint for an
      "every sweep model" threshold. See "Program target: the GPU prefill row, re-derived" above; the
      README table now carries the parity-corrected column alongside the original.*
- [x] Benchmark-parity preconditions landed in `compare-llama-cpp.sh` and a corrected re-baseline
      published and labelled as the new reference, with the pre-correction runs kept and marked
      pre-parity-correction. No number in this tier is scored across that boundary.
      *Reference published: `docs/perf-compare/20260925T172231Z/` (GPU) and `20260925T174146Z/` (CPU),
      both `failures=0`, every row at 128/128 prompt tokens and 64/64 generated tokens. Preconditions
      landed across three passes: the single JFR configuration and the `jdk.*` metrics; prompt-token
      parity with per-rep calibration; warmup, repetitions with a published median and min/max spread,
      a fixed per-model heap, and captured governor, turbo and GPU clock state; then two asymmetries
      the sweeps themselves exposed — generation length, closed by `min_tokens`, and generation
      context depth, closed by measuring prefill and generation in separate runs. The collection-pause
      rule is now machine-applied with a `Scorable` column; its lock-and-park half is withdrawn as
      unusable, with the measurement that shows why. Two earlier sweeps of the same day are kept and
      marked superseded. The one precondition still absent is a thread-count control reaching the hot
      path, which belongs to the CPU hot-path tier and is recorded in every published index as a
      stated mismatch rather than a silent one.*
- [ ] Residency primitive implemented, unit-tested, and documented (what it is, where the
      materialization boundary is, which ops participate).
- [ ] RMSNorm + RoPE measured *faster* than CPU scalar (or at minimum, no longer the ~7x-slower
      finding from Phase B) with residency, on real GTX 1080 hardware, published in
      `docs/perf-compare/`. **Contingency, decided before Tiers 02/06/07 start**: this project has
      already shelved three closely-related bets on grounds that turned out to be exactly this kind
      of per-op dispatch overhead (`CudaRmsNorm`/`CudaGraphSession` itself, draft-model speculative
      decoding, `VectorQuantKernels.dot()`), so a fourth negative result is a real possibility, not a
      formality. If the measured result is still worse than CPU scalar after the residency primitive
      is correctly wired (not just "not yet wired right"), do not iterate indefinitely — document the
      measurement, what was tried, and why it still regresses, then downgrade Tier 01 to
      **partial-complete**: the residency primitive itself (item 1) and the correctness guarantees
      below still ship and close out, but the "no longer dormant scaffolding" bullet below is
      explicitly waived for this pass, and Tiers 02/06/07 proceed using today's op-at-a-time GPU path
      for any sub-item that isn't itself residency-dependent, with their own residency-specific
      sub-items marked `NEEDS-TIER-01-REVISIT` rather than blocked indefinitely. Escalate to the user
      at that point rather than silently re-scoping — this changes three other tiers' scope, not just
      this one's. **Tier 01B is in that set, and is its most affected member.** An earlier draft
      excused it on the grounds that prefill is dominated by large-batch GEMM and host-device staging
      rather than per-op dispatch overhead — but host-device staging is precisely what residency
      removes, and Tier 01B's largest scope item is built on this primitive. If this tier downgrades,
      Tier 01B's item 2 does not proceed on a substitute design; it escalates. Its other items (the
      `--gpu-attention` architecture coverage, the JFR breakdown, chunk sizing, residual attention)
      proceed unchanged on today's GPU path.
- [ ] No correctness regression: greedy decode output identical (CPU) or within tolerance (GPU)
      with the new path enabled vs. disabled, across all three cross-surface-listed models.
- [ ] Cluster (pipeline- and tensor-parallel) smoke tests confirm activations still correctly
      materialize at the process/AllReduce boundary — no stale or device-resident data crossing a
      gRPC call.
- [ ] LoRA train + playback smoke tests unaffected.
- [ ] `docs/agent-arch.txt`/`docs/performance.md`/`docs/howto.md` updated (Juno-native language).
      *Done for both harness passes and unticked only because the box also covers the residency work,
      which has not started. This pass added `JfrMetricsCli` to the `metrics` entry in
      `docs/agent-arch.txt`, its invocation to `docs/howto.md`, and the second measurement boundary
      (calibrated prompt length, warm and repeated readings, harness-owned recording window, fixed
      heap, recorded clock state) to `docs/performance.md`.*
- [ ] `CudaGraphSession`/`CudaRmsNorm` are no longer "dormant scaffolding" — either wired live
      (preferred, if the measurement confirms the fix), or the tier is explicitly marked
      **partial-complete** per the contingency above (not silently marked complete with the
      scaffolding still dormant and unexplained).
- [ ] Full `mvn test`/`mvn verify -pl juno-master` pass with zero regressions.
      *`mvn test` across all eleven modules passes as of this pass: **1646 tests, 0 failures, 0 errors,
      46 skipped** in 24:55 min (`tokenizer` 89/2 skipped, `lora` 116, `node` 597/41, `coordinator` 311,
      `sampler` 81, `kvcache` 78, `health` 23, `registry` 93, `vision` 95, `metrics` 52, `juno-player`
      111/2). `coordinator` is 311 where the previous pass recorded 294, `sampler` 81 where it was 66,
      and `juno-player` 111 where it was 104 — the differences are this pass's new cases. Run with
      `-Dsurefire.rerunFailingTestsCount=2` for the known-flaky `metrics` attention test, as the first
      pass in this tier also did; no flake was reported. Unticked because the box also covers
      `mvn verify -pl juno-master` and the residency work, neither of which this pass carries.*
- [ ] `CHANGELOG.md` entry added.
      *An entry covering this pass is in (Session 90). Unticked because the box covers the tier, whose
      residency work has not shipped.*
