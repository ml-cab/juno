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
      at 84%. The generation ratio is unaffected.*
- [ ] The program target's GPU pp row re-derived against the parity-corrected re-baseline and
      restated in this file, with the number and the reasoning.
- [ ] Benchmark-parity preconditions landed in `compare-llama-cpp.sh` and a corrected re-baseline
      published and labelled as the new reference, with the pre-correction runs kept and marked
      pre-parity-correction. No number in this tier is scored across that boundary.
      *Partly landed: prompt-token parity, the JFR configuration and the recorded thread mismatch are
      in. Still outstanding: warmup, repetitions with a published median and min/max, a fixed
      per-model heap, captured CPU governor and GPU clock state, and the re-baseline run itself.*
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
- [ ] `CudaGraphSession`/`CudaRmsNorm` are no longer "dormant scaffolding" — either wired live
      (preferred, if the measurement confirms the fix), or the tier is explicitly marked
      **partial-complete** per the contingency above (not silently marked complete with the
      scaffolding still dormant and unexplained).
- [ ] Full `mvn test`/`mvn verify -pl juno-master` pass with zero regressions.
- [ ] `CHANGELOG.md` entry added.
