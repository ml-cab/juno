# Tier 01: GPU activation-residency redesign

Status: **complete** (2026-09-27). The decode residency region (norm, Q/K/V projection, RoPE) is wired
behind `--gpu-residency`, default off by owner decision, +3.6% to +5.4% decode where it runs; the
primitive threshold closed under the contingency (decode width missed, prefill width at parity,
chaining met; handed to Tier 01B item 2); step 3b, `CudaGraphSession` and the flag default moved to
Tier 02 by the owner. Closed with three out-of-tier fixes - per-request device memory (a GPU prefill
measurement boundary), `-Pintegration`, the tensor-parallel load error - and the fast prefill
repetition handed to Tier 01B.
Next in the running order: Tier 01B.
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
   *2026-09-27, owner decision: `CudaRmsNorm` is wired live as the resident norm inside
   `ResidentQkvPath` (behind `--gpu-residency`). `CudaGraphSession` is **moved to Tier 02**
   ([Tier 02 scope item 5](TIER-02-attention-long-context.md#in-scope)), because graph replay only
   pays once a region issues many launches per wait, which is what attention inside the region
   creates; today's region has five launches per layer. See "2026-09-27 - owner decisions on the
   step 3a pass" in the execution record.*
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
- `CudaGraphSession` graph capture/replay — deferred to [Tier 02](TIER-02-attention-long-context.md)
  scope item 5 with its wire-or-delete decision rule (owner, 2026-09-27); the code and
  `CudaGraphSessionTest` stay in place until then.
- Step 3b (KV append and attention inside the residency region) — [Tier 02](TIER-02-attention-long-context.md)
  scope item 4, per the owner, 2026-09-27.
- Making `--gpu-residency` default-on — re-decided after step 3b is measured
  ([Tier 02](TIER-02-attention-long-context.md) scope item 6); it stays `off` until then (owner,
  2026-09-27).

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
  - **And the same chain must reach >= 1.0x the CPU scalar path at prefill width (batch 512, dim
    2048), measured and reported separately from the decode-width figure.** The decode-width number
    alone is the wrong evidence for the primitive's largest consumer:
    [Tier 01B](TIER-01B-prefill-throughput.md) item 2 is built on this primitive and runs at a
    512-token prefill window, where the cost being removed is roughly 8 MB of activation staged each
    way per matmul rather than a per-op launch. A pass at batch 1 does not license that, and a fail at
    batch 1 should not block it. So report both widths, and read the contingency below against both:
    if the two disagree, record both figures and let Tier 01B item 2 proceed on the prefill result,
    since that is the width it will actually run at. A tier downgrade on a decode-width failure while
    the prefill width passed would block the one item the primitive most clearly helps, on evidence
    that never tested it.
  - Chaining must beat op-at-a-time: the two-op resident chain must cost **<= 0.7x** the same two
    ops run through today's host-round-trip-between-ops path, since eliminating one round trip is
    the entire mechanism being tested. If the chain is no cheaper than the sum of its parts, the
    primitive is not doing what it was built to do, whatever the absolute numbers say.
  - No regression elsewhere: end-to-end tg within **0.95x** of the pre-tier baseline on every sweep
    model with residency enabled, and `compare-lora.sh` at train **>= 0.95x**, wall-clock playback
    tps **>= 0.80x**.
  - Device memory returns to its pre-request level after each request (no leak across repeated
    requests), asserted via `GpuBindings.memGetInfo` rather than inferred from not crashing.
    *Met 2026-09-27 on the default path as well as the residency path, after the per-request scratch
    fix (see the close-out section): `memGetInfo`-based tests in `CudaMatVecScratchLifetimeTest`,
    `CudaAttentionNormScratchLifetimeTest` and `ResidentQkvPathTest`; smoke, 8 requests: tinyllama 938
    MiB after every request with the region off and on, mistral-7b 4604 off / 4608 on, llama-1-30b (at the
    card's capacity) flat from request 7 off and request 4 on over 16 requests.*
- **Benchmark-parity preconditions (blocking, and they come first).** This tier carries the harness
  corrections described in [`README.md`](README.md)'s "Benchmark parity preconditions" — prompt-token
  parity, Juno warmup and repetitions, matched thread count, fixed heap, recorded clock state, one
  shared JFR configuration — because it is the first tier to publish a llama.cpp-relative ratio and
  every later tier's gate inherits whatever this one establishes. Land them and re-baseline *before*
  measuring any residency result, so the tier's own before/after comparison is taken on one side of
  the change rather than across it. The `--threads` control the parity work wants does not exist
  until Tier 10 item 5; until then, record Juno's effective parallelism in the run metadata and state
  the mismatch in the published INDEX rather than leaving it implicit.

  Three of these are bigger than they look and none is optional:

  - **Prompt-token parity is the one that moves the numbers.** `RAW_PROMPT` defaults to `0` today,
    so every published pp ratio compared llama-bench's `-p N` against Juno's hard-coded 20-to-30
    token sentence. Flipping the default and enforcing the 10% `prompt_tokens`-versus-`n_prompt`
    check is what makes the pp column mean anything. Expect the re-baselined pp ratios to move
    substantially, and in the unflattering direction.
  - **Pre-tokenizer parity (precondition 7) is a `src/main` change, and it is the other half of
    precondition 1.** `GgufTokenizer` never reads `tokenizer.ggml.pre` — a repo-wide grep finds no
    reference to that key anywhere — so it runs one BPE strategy over every file regardless of the
    pre-tokenizer type the file declares, while the reference tool applies the split the file asks
    for. Precondition 1 makes Juno prefill the *number* of tokens the reference was given; this makes
    them the *same* tokens. Without it, the calibrated prompt hits a token count that matches by
    construction while the tokenization underneath it may not, and the difference is absorbed into
    both the pp denominator and the tg context depth with nothing to detect it inside the 10%
    tolerance.

    Scope here is exactly [Tier 04B](TIER-04B-tokenizer-fidelity.md)'s items 1 and 3 and no more:
    read the key, dispatch to a pre-tokenizer implementation per declared type, keep today's
    behaviour as the no-key path (correct for the SentencePiece-era models that predate the key), and
    reject a file declaring an unimplemented pre-type at load with an error naming it — the same
    treatment `ForwardPassHandlerLoader` now gives an unrecognized architecture. Enumerate the types
    actually present by running `./juno gguf-info` across `models/` before writing any code, rather
    than implementing a guessed list, and record the enumeration in this file. Tier 04B keeps its
    items 2 and 4 (the per-family split implementations beyond what the sweep models need, and the
    cross-engine token-ID parity corpus); those deepen the guarantee but do not gate a ratio, so they
    stay where they were.

    **This one can invalidate the reference re-baseline, and that is the point of landing it here
    rather than eight tiers later.** If dispatching on the declared pre-type changes any sweep model's
    `prompt_tokens` for the benchmark prompt, the reference sweep is re-taken on top of it and the
    earlier one is marked superseded, exactly as the two sweeps of 2026-09-25 were. Paying that once,
    now, is cheaper than paying it after five tiers have published gates against the old denominator
    — and cheapest of all if the enumeration turns out to show the sweep models' declared pre-types
    already match what `GgufTokenizer` does, in which case the finding is that the reference stands
    and the guarantee is now explicit instead of accidental. Record which of those two happened.

    The LoRA question Tier 04B raises (an adapter trained under the old tokenization may not compose
    with a base model tokenized under the new one — `tinyllama-1.1b-chat-v1.0.Q4_K_M.lora` on disk was
    trained under today's tokenizer) is **not** resolved here. If this precondition changes TinyLlama's
    tokenization, note the fact and keep the adapter's provenance recorded; the decision between
    re-train, version the `.lora` format, or accept-and-warn stays Tier 04B's, where its cross-surface
    rows 9 and 10 already own it.
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

**They are now owned.** [Tier 01B](TIER-01B-prefill-throughput.md) scope item 5 takes all eight, via a
shared helper in `perf-lib.sh` rather than eight copies of the edit, and carries the exit criterion. It
is the right owner because it runs three of them — `compare-vision.sh`, `compare-prefill-batch.sh` and
`compare-schedule.sh` — as required gates, so it can actually exercise the fix, which is the thing this
tier could not do.

### 2026-09-26 — benchmark-parity precondition 7: the pre-tokenizer split is read and dispatched on

Scope of this pass: exit criterion 5 only — [Tier 04B](TIER-04B-tokenizer-fidelity.md)'s items 1 and
3, hoisted here so that no tier publishes a prefill ratio whose denominator a later tier moves. No
residency code was written and no residency number was measured.

**Plan-versus-code drift found before starting.**

- HEAD is `d185756`. Tier 00's claims were re-verified rather than trusted, at this HEAD rather than
  at the one the previous passes checked: `generateBatch()` still carries the explicit "No
  cachePrefix call" comment where the write used to be, `ForwardPassHandlerLoader` still routes
  through `LlamaFamilyArchitectures.requireVerified`, a repo-wide grep finds no `com.hazelcast` or
  `RegistryService` symbol in any source or pom, the grep for internal tier numbers across every
  `src/main` tree is clean, `CLAUDE.md` lists `vision` and `metrics` in its test command,
  `TensorShardContext` is referenced only by its own test and a `ClusterHarness` comment, and
  `FaultTolerantPipeline` is reachable only through `HealthReactor`. All hold.
- The claim this pass turns on also holds: a repo-wide grep for `tokenizer.ggml.pre` over every
  `.java`, `.sh`, `.yaml` and `.proto` file outside `docs/` returned nothing. The key was read
  nowhere.
- **[`INVENTORY.md`](INVENTORY.md) understates what is on disk, again.** `phi-2.Q4_K_M.gguf`
  (`general.architecture=phi2`) is present and was not listed. The inventory's gap table asks for
  "a plain Phi-2 GGUF" for [Tier 01B](TIER-01B-prefill-throughput.md)'s per-architecture
  GPU-attention measurement and [Tier 06](TIER-06-speculative-decoding.md)'s `forwardVerify`
  coverage, and Tier 01B had pre-authorised shipping that architecture's default resolved off for
  want of the file. Corrected in the inventory and in both tiers, in this pass, per the rule the
  inventory itself carries about resolving a row everywhere it is cited.

**The enumeration, taken before writing any code**, by dumping every model file's metadata with
`GgufInfoMain` (what `./juno gguf-info` runs) and reading `tokenizer.ggml.model` and
`tokenizer.ggml.pre` off each. Sixteen files:

| Declared `tokenizer.ggml.pre` | Vocabulary | Files |
|---|---|---|
| *(absent)* | SentencePiece | `tinyllama-1.1b-chat-v1.0.Q4_K_M`, `tinyllama-1.1b-chat-v1.0.Q2_K` |
| *(absent)* | SentencePiece-style (`tokenizer.ggml.model=gemma4`) | `gemma-4-E4B-it-qat-UD-Q4_K_XL` |
| *(absent)* | GPT-2 BPE | `phi-2.Q4_K_M`, `moondream2-q5_k.llamafile` |
| `default` | SentencePiece | `Phi-3.5-mini-instruct-Q4_K_M`, `mistral-7b-instruct-v0.1-q4_k_m`, `mistral-7b-instruct-v0.2.Q2_K.llamafile`, `llama-1-30b.Q4_K_M` |
| `qwen2` | GPT-2 BPE | `qwen2.5-3b-instruct-q4_k_m`, `Qwen3-1.7B-Q4_K_M`, `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M` |
| `llama-bpe` | GPT-2 BPE | `Meta-Llama-3.2-1B-Instruct-Q8_0.llamafile` |
| `tekken` | GPT-2 BPE | `Devstral-Small-2-24B-Instruct-2512-UD-IQ1_S` |
| `minimax-m2` | GPT-2 BPE | `minimax-m2.5-tiny-24e-iq4_nl-imat` |
| `qwen35` | GPT-2 BPE | `Qwen3.5-0.8B.Q4_K_M` |

Two things follow from the table that the tier text did not anticipate. The declared value governs
**BPE vocabularies only** — a SentencePiece vocabulary takes no pre-tokenizer split, its word
boundaries coming from the `▁` prefix — so three of the four sweep models (`tinyllama`,
`Phi-3.5-mini`, `mistral-7b`) cannot be affected by this work at all, whatever they declare. And the
one sweep model that can be, `qwen2.5-3b`, declares `qwen2`. So the implemented set is `qwen2` plus
`llama-bpe`: the second is not a sweep model, but `Meta-Llama-3.2-1B-Instruct-Q8_0.llamafile`
declares it, is a verified `llama` architecture, and loads today — failing it closed to keep the
implemented set minimal would have removed a working model rather than protected one.

**What shipped.**

| Item | Where |
|---|---|
| The declared splits, the dispatch and the fail-closed set | `BpePreTokenizer` (new, `tokenizer`) |
| Reading the key, and not reading it for a SentencePiece vocabulary | `GgufTokenizer.resolvePreTokenizer` |
| Merging each pre-token on its own instead of the whole segment | `GgufTokenizer.mergePreTokens` |
| Today's whole-run path, unchanged, for a file that declares nothing | `GgufTokenizer.mergeWholeRuns` |
| The active split named in the tokenizer's load line | `GgufTokenizer.load` |

**Why the split matters at all, since nothing threw without it.** A BPE vocabulary is trained on
text that was first cut into pre-tokens, and merges are only ever learned inside one pre-token.
Merging a whole run in one pass admits pairs the training never produced. Measured on a 33-line
probe corpus against a second engine's tokenizer on the same strings:

| Model | Divergent lines before | After |
|---|---|---|
| `qwen2.5-3b` (`qwen2`) | 5 of 33 | **0 of 33** |
| `Qwen3-1.7B` (`qwen2`) | 5 of 33 | **0 of 33** |
| `Meta-Llama-3.2-1B` (`llama-bpe`) | 8 of 33 | **0 of 33** |

Two divergence classes, and both are silent. A run of whitespace before a word: `"a  b"` tokenized
as `"a"` + `"  "` + `"b"` where the training splits it `"a"` + `" "` + `" b"`, because the last space
of a run belongs to the word after it. And groups of digits, which only `llama-bpe` exposed here:
`"3.14159265"` merged into whatever the merge table allowed rather than into groups of at most
three, so a date or a version string tokenized differently from the way the model was trained to
read one. Most of the corpus already agreed — the merge table cannot contain a pair the split never
produced, so many boundaries are enforced implicitly — which is exactly why this was not visible
without measuring it.

**Files that declare nothing are byte-identical, verified rather than argued.** The same corpus was
encoded before and after the change on all five: `tinyllama`, `Phi-3.5-mini` and `mistral-7b`
(SentencePiece) and `phi-2` and `moondream2` (GPT-2 BPE, no key). All five produced identical token
IDs on all 33 lines. The no-key path is not a new code path — `mergeWholeRuns` is the previous
implementation moved into a method, merging one list of symbols across all segments exactly as
before — and a declared `default` takes it too, since an absent key and a declared `default` mean
the same thing.

**The fail-closed path, on real files.** `Qwen3.5-0.8B` (`qwen35`), `minimax-m2.5-tiny`
(`minimax-m2`) and `Devstral-Small-2-24B` (`tekken`) are now rejected at load with an error naming
the declared type and listing what is implemented. All three were already rejected for their
architecture, so no file that loaded before this pass fails after it. One behaviour change worth
recording: in local mode the tokenizer loads before the handler, so those three now report the
pre-tokenizer rejection rather than the architecture rejection. Both are correct refusals; the
message a user sees for those three files changed. Cluster mode is unaffected — the handler loads
first there, so it still reports the architecture. Verified end to end through the launcher:
`./juno local --model-path models/Qwen3.5-0.8B.Q4_K_M.gguf --cpu` exits 1 naming `'qwen35'`.

**The reference re-baseline stands, and here is the check rather than the argument.** The criterion
says that if any sweep model's `prompt_tokens` for the benchmark prompt changed, the reference
sweeps of 2026-09-25 are superseded and re-taken. They did not change, for two reasons that both
had to hold. Three of the four sweep models are SentencePiece and are untouched by construction. The
fourth, `qwen2.5-3b`, is affected in general but not by this prompt: the harness's raw prompt is
`"x x x … x"`, single letters separated by single spaces, which contains none of the constructs the
split moves — no whitespace run of two or more, no digits, no contractions. Confirmed by
measurement, not by reading the regex: see the harness run under "Verification commands" below,
which reproduces `128/128` prompt tokens at deviation 0 with the same calibrated word count.

**The LoRA-adapter question does not fire, and that is a finding rather than a deferral.**
[Tier 04B](TIER-04B-tokenizer-fidelity.md)'s cross-surface rows 9 and 10 ask whether an adapter
trained under the old tokenization still composes with a base model tokenized under the new one.
`tinyllama-1.1b-chat-v1.0.Q4_K_M.lora` was trained against `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`,
which is SentencePiece with no declared key, and its tokenization is byte-identical across this
change. So that adapter is **not** on the far side of a tokenization boundary, and Tier 04B's
decision (re-train, version the `.lora` format, or accept and warn) is still open but no longer has
a live instance on disk to resolve. It becomes live only if Tier 04B changes a tokenization an
adapter was trained under.

**Tests, written before the implementation.** `BpePreTokenizerTest`, 12 cases in the `tokenizer`
module, covering both splits (whitespace runs, digit grouping, contractions, line breaks,
punctuation), a losslessness property over a 17-string corpus including non-Latin text and emoji,
the dispatch, and the fail-closed set. They failed to compile against the original tree because the
class did not exist, which is the right failure. `PreTokenizerParityLiveTest`, 7 model-gated cases,
pins the token IDs a second engine produces for the same strings, plus the byte-identity of the
three no-declared-type files. Run against the original tree with the unit test set aside so the
module would compile: **3 failures of 7**, exactly the two whitespace cases and the digit-grouping
case, with the four cases that already agreed passing. That is the divergence reproduced as a test
before any fix, and the same seven pass now.

One expectation in each file was wrong when written and was corrected from the reference rather than
from reasoning: a punctuation run absorbs the line break that follows it, so `"end.\nNext"` is three
pieces and not four; and one live expectation had been copied from the wrong model's reading.

**Verification commands and results.**

- `mvn -o test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,vision,metrics,juno-player`:
  BUILD SUCCESS in 23:32 min, all eleven modules, **1674 tests, 0 failures, 0 errors, 46 skipped**
  (`registry` 93, `lora` 116, `kvcache` 78, `health` 23, `node` 597/41 skipped, `tokenizer` 108/2,
  `sampler` 81, `coordinator` 311/1, `vision` 95, `metrics` 61, `juno-player` 111/2). 1674 is 1646
  plus this pass's 19 `tokenizer` cases and the 9 `metrics` cases from the previous pass, which was
  recorded before they were counted. The skip count is unchanged at 46, so none of the seven new
  model-gated cases skipped — every file they need is on disk. Run without
  `-Dsurefire.rerunFailingTestsCount=2`: the flaky attention assertion is fixed, and the reactor
  reached its last two modules unaided for the first time in this tier.
- `mvn -o clean verify -pl juno-master`: BUILD SUCCESS, **20 tests, 0 failures, 0 errors** —
  `ThreeNodeClusterIT` 8, `InProcessClusterIT` 6, `TensorParallelClusterIT` 5,
  `UnsupportedArchitectureClusterIT` 1. Run because `CoordinatorMain` loads the tokenizer and this
  pass gave that load a new way to fail; the last of those four confirms the cluster path still
  reports the architecture refusal, since it loads handlers before the tokenizer.
- `mvn -o clean install -DskipTests` before both of the above, per the stale-artifact trap the first
  pass in this tier recorded.
- **The benchmark-prompt check, which is what decides whether the reference stands.** A GPU cycle on
  the one sweep model that can be affected
  (`--gpu --models qwen2.5-3b --n-prompt 128 --n-gen 8 --juno-warmup 1 --juno-reps 1 --reps 1
  --no-publish --no-tuned-lane`): `failures=0`, and the prefill lane reports
  `calibrated_prompt_words: 120`, `prompt_tokens: 128`, `n_prompt: 128`, `prompt_token_deviation: 0`.
  The published reference run [`20260925T172231Z`](../perf-compare/20260925T172231Z/) reports
  `calibrated_prompt_words: 120` and the same 128/128 on all three of its repetitions. **The same
  word count produces the same token count on both sides of the change**, which is the direct
  evidence that this is not a measurement boundary. Not published and not a baseline: one repetition,
  eight generated tokens, taken to read the parity fields.
- Three real files rejected by name at load (`Qwen3.5-0.8B` → `'qwen35'`, `minimax-m2.5-tiny` →
  `'minimax-m2'`, `Devstral-Small-2-24B` → `'tekken'`), and the first of those also through the
  launcher: `./juno local --model-path models/Qwen3.5-0.8B.Q4_K_M.gguf --cpu` exits 1 with the
  message, rather than the refusal being swallowed into a generic load error.
- The 33-line probe corpus encoded through Juno and through a second engine's tokenizer on every
  file where both can read it, before and after: the parity and byte-identity tables above.

**Performance gate: not run, and not required for this pass.** Nothing here touches the forward
pass, MatVec, GPU residency, batching, KV or quantization. The change is to how a prompt is cut into
tokens, not to what is computed per token, and the benchmark prompt's token count is unchanged, so
no published figure moves. This pass is **not** a measurement boundary — the first pass in this tier
that is not.

**Cross-surface reading for this pass** (the tier's own checklist is about residency, which has not
started): tokenization is backend-agnostic, so CPU and GPU see the same token IDs by construction and
the `--cpu` launcher run above exercises the refusal path; static and continuous schedules and both
REST surfaces tokenize through the same `GgufTokenizer.encode` with no shape change; the cluster
rows are covered by the four ITs, which also confirm nodes do not re-tokenize; LoRA train and
playback are covered by `juno-player`'s 111 tests including `LoraTrainingSequencesTest`, whose
completion-only loss masks are computed from token counts on a SentencePiece base that is
byte-identical here; vision is covered by the `vision` module's 95 tests plus moondream2's
byte-identical corpus, since it declares no pre-type and its `<image>` splice is unchanged; the CLI
gains no flag, and `./juno gguf-info` already reports the declared value because it dumps every
metadata key, which is how the enumeration above was taken.

**Not done in this pass**, and still owned by [Tier 04B](TIER-04B-tokenizer-fidelity.md):

- The splits for `tekken`, `minimax-m2` and `qwen35` — its item 2. Those three files are rejected
  rather than mistokenized, which is the safe failure and still a failure.
- The split the reference applies to a BPE vocabulary that declares `default` or nothing. Juno keeps
  its whole-run merge there, which is what `phi-2` and `moondream2` already ran under and what this
  tier's scope line ("keep today's behaviour as the no-key path") asks for. It remains a divergence
  from the reference for those two files, now explicit and logged at load instead of unstated. Also
  Tier 04B item 2.
- The cross-engine parity corpus as a committed artefact — its item 4. The 33-line corpus used here
  lives in this pass's scratch directory and its findings are the table above; Tier 04B is what
  turns that into a maintained fixture.

### 2026-09-26 — implementation step 1: the round-trip cost re-measured before any residency code

Implementation step 1 asks for the correctness/perf harness *first*, "so the 'regresses decode'
claim can be re-measured objectively before any residency code exists". That claim is the premise of
this whole tier and it was being carried as two remembered numbers in prose — `CudaRmsNorm`'s class
javadoc saying roughly 11x on a live decode A/B, and `docs/performance.md`'s Phase B checkpoint
saying 1.56x to 2.18x slower at prefill batch scale. Neither was reproducible on demand. Both now
are.

**What shipped.** `RmsNormRoundTripMicrobench` (`node`, package-private neighbours so it can call
both the real scalar path and the real GPU path, not copies of them) runs two lanes at each width:

- `cpu-scalar` — `LlamaTransformerHandler.rmsNormInto`, the path the handler actually uses today.
- `gpu-round-trip` — `CudaRmsNorm.normalizeBatch`, today's per-call host-to-device upload, kernel
  launch and device-to-host download.

Ratios are reported as CPU-scalar median over lane median, so a lane at least as fast as the path it
would replace reads at or above `1.00x`. That is deliberately the same unit this tier's threshold is
written in (">= 1.0x the CPU scalar path"), so the residency result can be read against the
threshold without a conversion step. The residency lane plugs in as a third lane at the same two
widths.

`scripts/performance-tests/rmsnorm-roundtrip-microbench.sh` drives it, captures JFR under the
tier's own `juno-perf.jfc`, and extracts GC and allocation figures through `JfrMetricsCli`.

**The measurement**, GTX 1080, dim 2048, 3 repetitions, published as
[`docs/perf-compare/20260926T060301Z-tier01-rmsnorm-roundtrip/`](../perf-compare/20260926T060301Z-tier01-rmsnorm-roundtrip/INDEX.md):

| width | batch | lane | median ms | spread % | vs CPU scalar | scorable |
|---|---:|---|---:|---:|---:|---|
| decode | 1 | cpu-scalar | 0.0035 | 0.6 | 1.00x | yes |
| decode | 1 | gpu-round-trip | 0.0371 | 2.3 | **0.09x** | yes |
| prefill | 512 | cpu-scalar | 1.3004 | 1.1 | 1.00x | yes |
| prefill | 512 | gpu-round-trip | 2.0859 | 1.5 | **0.62x** | yes |

**Both prose claims hold, and the second is now bounded rather than a range.** Decode is 10.6x
slower than scalar CPU, against the javadoc's "roughly 11x". Prefill is 1.60x slower, at the fast
end of Phase B's 1.56x-to-2.18x band — the band was three readings under different staging, this is
one number under one stated configuration.

**The finding that matters for what comes next: the gap narrows with width but does not close.** At
batch 1 a fixed per-call cost dominates and the GPU lane loses by an order of magnitude. At batch 512
that same fixed cost is amortised across 512 rows and the loss falls to 1.60x, while about 4 MB is
staged each way. So the two widths are failing for different reasons — launch overhead at decode,
staging bandwidth at prefill — and residency has a different job at each. This is precisely why the
threshold demands both widths separately and why a pass at one does not license the other. Neither
is at parity today, so both remain open for residency to close; there is no width at which the
current path is already good enough to leave alone.

**Correctness is checked on every run, not assumed.** The harness compares the GPU lane's output
against the scalar lane's and refuses to report timings if they diverge beyond `1e-4`: observed
`5.96e-07` at decode and `1.91e-06` at prefill. A harness that silently times a different
computation is worse than no harness.

**Noise.** One `jdk.GCPhasePause` of 2.71 ms across the whole run, so nothing needed re-running on
collection grounds. Row-level repetition dispersion is the binding rule (README's replacement for
the pause rule) and every row is inside it. Reaching that took a correction: at the harness's first
default of 500 ms warm-up per lane the decode GPU row's repetitions spanned **37.9%** of their
median and the row was correctly reported **unscorable**, so it was re-run rather than scored. The
defaults are now 3000 ms warm-up and an 800 ms measurement window, at which the same row spans 2.3%.
The dispersion rule caught a bad reading on its first real use.

**VRAM.** 65536 bytes not returned at process end — `CudaRmsNorm`'s per-thread device scratch,
allocated on first use and grown in place rather than freed per call, not a per-call leak. The
tier's device-memory exit criterion is about the residency primitive's own allocations and is not
claimed by this pass.

**One thing the harness learned the hard way, and now reports.** `memGetInfo` is device-wide, not
per-process. A stray surefire fork from an interrupted test run held 7.2 GB of the GPU alongside an
early run of this harness, and the retention line dutifully reported **4.1 GB "not returned"** —
memory belonging to another process — while the CPU lane also ran 17% slower on the contended host.
Nothing in the reading said so. The harness now carries device total alongside free bytes and states
plainly when retention exceeds what its own scratch could possibly be, so a contaminated reading
announces itself instead of being published as a leak. This matters beyond this harness: every later
tier that asserts "device memory returns to its pre-request level" is asserting it against the same
device-wide counter, and that assertion is only meaningful on an idle device.

**Tests.** `RmsNormRoundTripMicrobenchTest`, 19 cases, in the `node` module, GPU-free by design: the
scoring and reporting logic decides what this tier concludes, so it is covered without hardware,
while the measurement needs a device. Written first — it failed to compile against the tree as it
stood, which is the intended failure for a harness that does not exist yet. The cases pin median for
odd and even repetition counts, the 15%-of-median dispersion boundary in both directions, ratio
orientation (a slower lane must score below `1.00x`, not above it), rejection of a cell with no CPU
lane to score against, defensive copying of repetition arrays, and that both widths and the
unscorable marker survive into the rendered table.

**Full suite.** `mvn -o test` across all eleven test-bearing modules: **1697 tests, 0 failures,
0 errors, 46 skipped** in 23:41 min, and `mvn -o clean verify -pl juno-master` is **20 tests,
0 failures, 0 errors** (`InProcessClusterIT` 6, `ThreeNodeClusterIT` 8, `TensorParallelClusterIT` 5,
`UnsupportedArchitectureClusterIT` 1). That is the previous pass's 1674 plus this pass's 23 new cases,
with the skip count unchanged, so no new model-gated case started skipping. The tier's full-suite
exit criterion still covers the residency work and stays unticked.

**No perf gate applies to this pass, and that is a statement, not an omission.** The harness adds a
standalone class no handler references; the forward pass, MatVec, KV and batching paths are
unchanged, so `compare-lora.sh` and `compare-llama-cpp.sh` have nothing to detect. Both become
required when the residency path itself lands. No `compare-llama-cpp.sh` ratio is published here and
the 2026-09-25 reference sweeps are untouched.

**Still open in this tier** (implementation steps 2 to 5): the residency primitive itself, the
`RopeKernel` built on it from day one, wiring both through `LlamaTransformerHandler`'s decode path
behind an opt-in flag, and the two-op chain measurement the `<= 0.7x` chaining threshold is written
against. The chaining threshold cannot be read yet at all: it compares a resident two-op chain
against the same two ops run through today's host-round-trip path, and the second op does not exist
on the GPU today. Building `RopeKernel` is what makes that comparison possible, so the op-at-a-time
two-op baseline is taken in the same pass that builds it, not retrofitted afterwards.

**Four internal tier numbers removed from shipped docs, per execution rule 8.** The rule makes a
doc/claim audit repo-wide rather than scoped to whichever files a prior pass happened to look at.
Tier 00's equivalent grep was scoped to `src/main` and returned no hits, so the shipped docs were
never covered. A case-insensitive scan found `Infra Tier 11` in `docs/agent-arch.txt` and
`Tier 17`, `Tier 16` and `Tier 10` in `docs/performance.md` — all in files this pass was already
editing. Each now names the mechanism instead of the number (for example "GPU (batched-prefill
GEMM)" rather than "GPU (Tier 17 — batched-prefill GEMM)"). `docs/agent-arch.txt`, `docs/howto.md`,
`docs/performance.md` and `README.md` are now clean under that scan. **`CHANGELOG.md` still carries
15**, all inside historical release entries; rewriting shipped release history is a different
decision from correcting a current description, so it is left to Tier 14's audit rather than taken
unilaterally here. Worth noting that the first grep used for this check missed every one of them by
being case-sensitive on `Infra tier` and requiring a parenthesis — a check is only as good as its
pattern.

**Doc-maintenance gap noted, not fixed here.** `docs/perf-compare/README.md`'s run index stops at
`20260918T204959Z-lora`: the seven directories this tier's own precondition passes published on
2026-09-24 and 2026-09-25, including both reference re-baseline sweeps, were never added to it. This
pass added its own row and left the backlog alone rather than widening its scope. Tier 14's doc audit
owns closing it.

### 2026-09-27 — implementation step 2: the residency primitive, RoPE on the device, and the chain measured

Scope of this pass: implementation step 2 ("design and implement the minimal residency primitive,
scoped to RMSNorm + RoPE only"), its tests, and the op-at-a-time two-op baseline the step-1 record
committed to taking in the same pass that builds `RopeKernel`. Nothing is wired into a handler; that
is step 3. Dates in this section are UTC, matching the published run directory.

**Plan-versus-code drift found before starting.**

- HEAD is `cc94c53`. Tier 00's claims were re-verified at this HEAD rather than trusted:
  `generateBatch()` still carries the "No cachePrefix call" comment where the write used to be and
  calls neither `findLongestPrefix` nor `cachePrefix`; `ForwardPassHandlerLoader` still dispatches
  `phi2`/`phi3`/`qwen3`/`qwen3moe` and sends everything else through
  `LlamaFamilyArchitectures.requireVerified`; `Sampler`, `SamplingStep` and `RepetitionPenaltyStep`
  agree that `Sampler` alone states the order; `TensorShardContext` is referenced only by its test
  and `ClusterHarness`, and `FaultTolerantPipeline` only by `HealthReactor`; no `hazelcast` or
  `RegistryService` symbol survives in any source, proto or pom; a case-insensitive grep for
  internal tier numbers over every `src/main` tree is clean; `CLAUDE.md`'s test command lists
  `vision` and `metrics`. All hold. The session prompt that started this pass expected Tier 00 to be
  the active tier; it has been complete since 2026-09-23.
- **The step-1 record's claim that the production docs are clean of tier numbers is false.** It says
  `docs/agent-arch.txt`, `docs/howto.md`, `docs/performance.md` and `README.md` "are now clean under
  that scan". `docs/agent-arch.txt` carried four `PLAN-Infra-TierN.md` pointers and
  `docs/performance.md` carries 24 hits, among them two bare mentions in prose ("Tier 15 unblocked
  on gather tax", "Tier 5's domain") and a run of `PLAN-Infra-TierN.md` links. The two pointers in
  the `agent-arch.txt` entries this pass rewrote are gone; the other two there and all of
  `performance.md` are left to Tier 14's audit, which owns the repo-wide sweep, rather than widening
  this step.
- **README rule 4 is violated in `src/main` code comments, and nothing owns it before Tier 14.** Ten
  comments in `node` name the reference implementation (`GgufKQuantCodec`, `Qwen3Rope`,
  `Phi3RopeConfig`, `LlamaConfig` twice, `GgufReader` three times, `Phi2Rope`, `Phi3Rope`). Tier 00's
  item 9 covered tier numbers only. Recorded for Tier 14's grep, not changed here.
- **The no-emoji rule in `CLAUDE.md` is violated in CLI and log output, and no planned check looks for
  it.** `scripts/run.sh` prints `✔`/`⚠` from its `ok`/`warn` helpers, `scripts/aws/launcher.sh`
  prints `❌`, and `ConsoleMain` prints `✔` on loading a LoRA checkpoint; `docs/howto.md` and one
  historical `CHANGELOG.md` entry reproduce that output. Tier 14's consistency script greps competitor
  names and tier numbers only; adding pictographs to it would close this. Not changed here.
- `DeviceActivationBatch` is still the LoRA-training host-packing helper, as this file warned; the
  primitive is named `ResidentChain`/`ResidentActivation`. `LlamaTransformerHandler` is still the only
  handler holding the fused attention kernel (`gqaGpu`), and still leaves `rmsNormGpu` null.

**What shipped.**

| Item | Where |
|---|---|
| Residency region: one stream, the buffers on it, close frees all | `ResidentChain` (new, `node`) |
| Device activation buffer with the materialization boundary | `ResidentActivation` (new, `node`) |
| Launch parameter block, allocation-free, `invokeExact` | `KernelParams` (new, `node`) |
| RoPE kernel source and PTX | `node/src/main/cuda/rope.cu`, `rope.ptx` (new) |
| RoPE kernel loader and launch | `RopeKernel` (new, `node`) |
| RoPE on a resident activation, table uploaded once | `CudaRope` (new, `node`) |
| Resident RMS norm, weight uploaded once, never in place | `CudaRmsNorm.normalizeResident` |
| Allocation-free RMS-norm launch for the resident path | `RmsNormKernel.launchResident` |
| Four-lane chain measurement | `ResidentChainMicrobench` (new), `scripts/performance-tests/resident-chain-microbench.sh` (new) |

**Design, and why each choice.**

- *Where the boundary is.* Data crosses host and device only at `ResidentActivation.upload` (async,
  on the chain's stream) and `materialize` (the one point the host waits and sees device results).
  Operations between them touch device memory only; the host input array is untouched after
  upload, which a test asserts. A consumer that is not device-resident takes a materialized copy.
  Which operations participate today: RMS norm and RoPE.
- *One stream per region, owned by the region.* Ordering comes from the stream, so a chain of
  operations needs no host wait between them. `ResidentChain.close` synchronizes, frees every
  activation still open and destroys the stream, so a region cannot leak by forgetting a buffer.
  The buffers and stream go through `GpuBindings` and are vendor-neutral; the two kernels are
  CUDA-only, and `CudaRope.tryCreate` returns null off CUDA, the same contract as
  `CudaRmsNorm.tryCreate`.
- *Dispatch that does not undo the saving.* Every existing kernel launch builds a confined arena,
  allocates a slot per argument and calls the driver through `invokeWithArguments`, which boxes each
  argument. That is noise beside a synchronous round trip and is not beside a bare launch, so the
  resident path uses `KernelParams` (built once per thread, rewritten in place) and `invokeExact`
  for launches, copies and synchronizations. The old launch is kept unchanged on purpose: it is the
  round-trip path the step-1 baseline measured, and changing it would move the before-side of this
  tier's comparison.
- *RoPE precision.* The CPU path computes a double angle and double sine and cosine, then rounds;
  a single-precision angle would be off by about 2e-3 radians at position 30000. The kernel follows
  the CPU step by step: a double inverse-frequency table computed on the host with the CPU path's own
  expression and uploaded once, `sincos` in double, and the rotation written with
  `__fmul_rn`/`__fadd_rn`/`__fsub_rn` so the compiler cannot contract it into fused multiply-adds
  (the PTX was checked: `mul.rn.f32`/`sub.rn.f32`/`add.rn.f32`, no float `fma`). Result: bit-identical
  to the CPU rotation in all five parity cases, including position 30000, 128-wide heads and base
  1e6.
- *Adjacent pairs only.* The kernel implements the pairing `LlamaTransformerHandler` uses. Split-half
  (NeoX) pairing, which `Phi2Rope`/`Phi3Rope` use on the CPU, is not built, because nothing on the
  device path would call it; any later handler that wants the GPU RoPE needs it first.
- *No CUDA graph.* `CudaGraphSession` stays unwired. A captured graph fixes its kernel arguments,
  and the RoPE position is an argument that changes every decode step; replay would need the
  position read from device memory or an exec-node parameter update. Whether launch overhead is
  worth that is a step-4 question the measurement below starts to answer.

**Tests, written first.** `RopeKernelParityTest` (6 cases), `ResidentActivationTest` (8),
`ResidentChainMicrobenchTest` (7, GPU-free) and one new `CudaRmsNormTest` case (the resident path at
batch 1, 8 and 512 against the scalar reference and, exactly, against the round-trip path — same
kernel, same inputs, so the bits must not differ; they do not). 22 new cases in `node`. Against the tree as it stood, `mvn -pl node
test-compile` failed with `cannot find symbol` for every new class and nothing else, which is the
right failure for code that does not exist yet.

**One test had no teeth, and it was caught.** The test for two uploads in a row — both copy through
one pinned host buffer, so the second can overwrite the first while its transfer is still reading —
passed three runs out of three with the guard deliberately disabled. The transfer from pinned memory
simply outran the host's overwrite. It now queues milliseconds of device work ahead of the first
transfer, so the transfer is still waiting when the host starts the second copy: with the guard
disabled it fails three runs out of three (the device normalized the second rows in place of the
first), with the guard it passes three out of three. The guard is an epoch check, one synchronization
only when an upload follows another with no wait between.

**The measurement.** Published as
[`docs/perf-compare/20260927T025430Z-tier01-resident-chain/`](../perf-compare/20260927T025430Z-tier01-resident-chain/INDEX.md).
GTX 1080, dim 2048 as 32 heads of 64, base 10000, decode row at position 512, prefill window at
positions 0 to 511, 3 repetitions. All GPU lanes use device-resident weights, so op-at-a-time and the
resident chain differ by the host round trip alone.

| width | lane | median ms | min / max | vs CPU scalar | cost vs op-at-a-time |
|---|---|---:|---|---:|---:|
| decode | cpu-scalar | 0.0670 | 0.0669 / 0.0677 | 1.00x | — |
| decode | gpu-op-at-a-time | 0.0359 | 0.0332 / 0.0364 | 1.86x | 1.00 |
| decode | gpu-resident-chain | 0.0183 | 0.0182 / 0.0188 | **3.65x** | **0.51** |
| decode | gpu-device-only | 0.0112 | 0.0112 / 0.0114 | 5.98x | 0.31 |
| prefill | cpu-scalar | 34.06 | 33.91 / 34.49 | 1.00x | — |
| prefill | gpu-op-at-a-time | 3.88 | 3.86 / 3.90 | 8.77x | 1.00 |
| prefill | gpu-resident-chain | 2.15 | 2.10 / 2.16 | **15.86x** | **0.55** |
| prefill | gpu-device-only | 0.158 | 0.156 / 0.159 | 215.5x | 0.04 |

Every row scorable (largest spread 9.0%); one 3.7 ms collection pause over the whole run; 56 MB
allocated in total, all setup, none per call; 0 bytes of device memory not returned; largest
divergence from the CPU chain 2.4e-6, all of it the norm's summation order.

**Read against this tier's Threshold block, as written:**

- Resident chain `>= 1.0x` CPU scalar at decode width: **3.65x**. At prefill width: **15.86x**.
- Resident chain `<= 0.7x` op-at-a-time: **0.51** at decode, **0.55** at prefill.
- Device memory back to its starting level: **exactly**, per allocate-and-close cycle in the unit
  tests and over the whole harness run. Per request is a step-3 assertion.
- End-to-end tg, `compare-lora.sh`: not applicable until the path is wired.

**And the reading that has to go next to it, because it changes what those numbers mean.** About 95%
of the CPU chain is RoPE: the CPU norm alone is 0.0035 ms at decode and 1.3004 ms at prefill on this
host ([`20260926T060301Z-tier01-rmsnorm-roundtrip`](../perf-compare/20260926T060301Z-tier01-rmsnorm-roundtrip/INDEX.md)),
and execution samples put 1004 of this run's 1663 in `LlamaTransformerHandler.rope`. The scalar
`rope` recomputes a `Math.pow`, a `Math.cos` and a `Math.sin` for every rotated pair on every call.
So the `>= 1.0x` result is mostly a statement about the CPU RoPE's cost, not about residency. The
chaining result (0.51 / 0.55) is the one that isolates residency, and it holds. Against the CPU norm
alone, the resident chain is still about five times slower at decode width: **a two-operation region
that pays its own entry and exit does not beat an elementwise operation that is cheap on the CPU at
decode width.** Estimated, not measured: against a CPU RoPE that computes each position's angles once,
the decode-width chain would read about 0.3x and the prefill-width chain about parity, while the
device-only lane (0.158 ms against roughly 2 ms) would still clear it by an order of magnitude at
prefill. That is consistent with the step-1 finding that decode fails on fixed per-call cost and
prefill on staged bytes, and it says the decode-width win has to come from a region spanning many
operations per wait, not from norm and RoPE alone.

**The CPU RoPE cost is large in the real forward pass, measured, and nothing in this plan owns it.**
From the `juno.Rope` and `juno.ForwardPass` spans of the parity-corrected GPU reference sweep
([`20260925T172231Z`](../perf-compare/20260925T172231Z/INDEX.md), repetition 2, default lane), where
every matrix product already runs on the GPU:

| model | lane | `juno.Rope` | `juno.ForwardPass` | share |
|---|---|---:|---:|---:|
| tinyllama | generation, 64 tokens | 111.95 ms | 1126.73 ms | 9.9% |
| mistral-7b | generation, 64 tokens | 349.34 ms | 3240.58 ms | 10.8% |
| qwen2.5-3b | generation, 64 tokens | 192.23 ms | 2394.09 ms | 8.0% |
| tinyllama | prefill, 128 tokens | 216.07 ms | 918.47 ms | 23.5% |
| mistral-7b | prefill, 128 tokens | 682.16 ms | 2917.92 ms | 23.4% |

The angle depends only on position and pair index, yet it is evaluated for every head of every layer:
792 times per token on TinyLlama. Two ways to remove it: the GPU kernel this pass built (step 3), or
computing each position's angles once per forward pass on the CPU, which can be bit-identical because
the cached values would be the same expressions. The second is small, touches the forward pass of
every Llama-family model on every backend, and would **move the reference sweep's figures** (a
measurement boundary under README rule 9) **and the CPU baseline this tier's decode threshold is read
against**. That makes its placement a decision for the owner, not for this pass: see "Open for the
owner" below.

**Open for the owner — decisions this pass surfaced and did not take.**

1. *The CPU RoPE angle recomputation* (about 10% of GPU decode and 23% of GPU prefill forward-pass
   time above). Land it before step 3 as a recorded out-of-tier change and re-take this chain
   measurement against the corrected CPU path, or leave the CPU path alone and let step 3's GPU RoPE
   take it. Taking it first gives step 3 an honest baseline; it would also likely turn the
   decode-width `>= 1.0x` reading into a miss, which this tier's contingency already classes as a
   partial result for Tier 01B, not a downgrade.
2. *Whether `qwen2` and `qwen3`/`qwen3moe` use the right RoPE pairing.* `LlamaTransformerHandler`
   (for `qwen2`) and `Qwen3Rope` apply adjacent-pair rotation, and nothing in `src/main` permutes
   their Q/K rows. The reference implementation applies split-half (NeoX) rotation to those
   architectures, and its converter permutes Q/K rows only for `llama`. Juno already uses split-half
   for `phi2` and `phi3`, with a comment saying why adjacent is wrong there. A greedy run of
   `qwen2.5-3b` on this HEAD answers "The capital of France is Paris." correctly, which does not
   settle it: any consistent pairing keeps relative position, so a short factual prompt can survive
   the wrong one. `Qwen2LiveForwardTest` only asserts that the first token is not end-of-turn.
   Settling it needs a perplexity comparison of the two pairings on a real text, or logits against
   the reference tool, on `qwen2.5-3b` and `Qwen3-1.7B`. If confirmed, it is a forward-pass
   correctness fix on a sweep model and needs a home.
3. *Step 3's scope, as written, would regress decode.* In a transformer layer the Q/K/V projection
   sits between the norm and RoPE, and `MatVec` is host-in, host-out. Wiring norm and RoPE alone
   would add round trips: materialize the norm for the host projection, then upload Q and K for RoPE.
   Step 3 therefore needs a device-in, device-out entry point on at least the projection path the
   decode loop uses (the K-quant MMQ path already runs device pointer to device pointer inside
   `CudaMatVec`, between its own upload and download), and a rule for ordering `CudaMatVec`'s
   per-thread stream with the chain's. That is the smallest design that can meet step 3's intent;
   it is more than "norm and RoPE only", so it is flagged here rather than assumed.
4. *`min_tokens` no longer holds against every stop token.* Found while explaining this pass's test
   counts: the turn-marker change that shipped inside `cc94c53` (see Out-of-tier changes) merges the
   vocabulary's turn-marker ids into every request's stop set, and `MinTokenFloor` holds back only
   the end-of-sequence id, so a request can end below its minimum on a turn marker - by id, or as
   text through `EosOutputFilter`, which also does not consult the minimum. The benchmark prompt does
   not trigger it on any of the four sweep models (checked at this HEAD, including `tinyllama`, whose
   role headers are plain text rather than control tokens), so no published figure moved. Masking every stop-token id below the minimum, rather
   than only the end-of-sequence id, would restore what `min_tokens` promises; it is a sampler change
   outside this tier's scope, so it is the owner's to place.

**Verification commands and results.**

- `mvn -o -pl node test -Dtest='RopeKernelParityTest,ResidentActivationTest,CudaRmsNormTest,ResidentChainMicrobenchTest,RmsNormKernelParityTest,CudaGraphSessionTest,RmsNormRoundTripMicrobenchTest'`:
  54 tests, 0 failures, 0 skipped, so every GPU case ran on the GTX 1080. RoPE parity printed
  `max|diff| = 0` in all five rotation cases.
- The upload-race case with its guard disabled, three runs: 3 failures (row 0 normalized from the
  second upload); guard restored, three runs: 3 passes. The source was compared byte for byte with a
  copy taken before the experiment.
- `mvn -o clean install -DskipTests`, then
  `mvn -o test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,vision,metrics,juno-player`:
  BUILD SUCCESS in 22:46 min, **1731 tests, 0 failures, 0 errors, 46 skipped** (`registry` 93, `lora`
  116, `kvcache` 78, `health` 23, `node` 642/41 skipped, `tokenizer` 109/2, `sampler` 81,
  `coordinator` 322/1, `vision` 95, `metrics` 61, `juno-player` 111/2). Against the step-1 pass's
  1697: plus this pass's 22 `node` cases, plus 12 cases the turn-marker change in `cc94c53` added
  (1 `tokenizer`, 11 `coordinator`; see Out-of-tier changes). Skip count unchanged at 46.
- `mvn -o clean verify -pl juno-master`: BUILD SUCCESS, **20 tests, 0 failures, 0 errors**
  (`InProcessClusterIT` 6, `ThreeNodeClusterIT` 8, `TensorParallelClusterIT` 5,
  `UnsupportedArchitectureClusterIT` 1); no node JVM left behind.
- `./scripts/performance-tests/resident-chain-microbench.sh` on an idle device: the published run.
- Two short, unpublished harness runs to check whether `cc94c53` moved the reference sweep's parity
  (`compare-llama-cpp.sh --gpu --models qwen2.5-3b,Phi-3.5-mini --n-gen 64 --juno-reps 1
  --juno-warmup 1 --reps 1 --no-publish --no-tuned-lane`, then the same with
  `--models tinyllama,mistral-7b`): `failures=0` both times, every model - including both tinyllama
  files - at 128/128 prompt tokens (deviation 0) and 64/64 generated tokens, `finish_reason:
  length`. The first run alone covered only the two models whose markers are control tokens; the
  second was added because the new role-header stop also works on decoded text, which is where
  tinyllama's markers live. One repetition each, not a baseline, not scored.

**Performance gate: not run, and not required for this pass.** No handler constructs
`ResidentChain`, `CudaRope` or `RopeKernel`, and `CudaRmsNorm` is still unconstructed by default, so
the forward pass, MatVec, KV, batching and quantization paths are unchanged and `compare-lora.sh` /
`compare-llama-cpp.sh` have nothing to detect. The existing round-trip launch was deliberately left
alone so the step-1 baseline still describes the code. Both gates become required in step 3. This
pass is not a measurement boundary.

**Cross-surface reading for this pass.** Nothing reaches a product surface yet: CPU inference (row 1)
is untouched and remains the oracle both new kernels are tested against; CUDA (row 2) is where every
new test runs; ROCm (row 3) gets vendor-neutral buffers and stream handling through `GpuBindings` but
no kernels, and `CudaRope.tryCreate` returns null off CUDA — not exercised, no AMD hardware; the
schedules, cluster modes, LoRA, vision, REST and CLI rows are untouched because no handler or entry
point references the new classes. The tier's own checklist is resolved when step 3 wires the path.

**Docs updated.** `docs/agent-arch.txt` (entries for `ResidentChain`, `ResidentActivation`,
`KernelParams`, `RopeKernel`, `CudaRope`, `ResidentChainMicrobench`; `CudaRmsNorm` and
`CudaGraphSession` rewritten), `docs/howto.md` (the new harness), `docs/performance.md` (the chain
reading and the CPU RoPE share), `docs/perf-compare/README.md` (run index row), `CHANGELOG.md`
(Session 94). `README.md` needs nothing: it describes no class at this level.

### 2026-09-27 — follow-up item 1: the Qwen RoPE pairing, diagnosed by perplexity

Scope: "Open for the owner" item 2 above, taken first in the order the owner approved on 2026-09-27
(diagnose the Qwen pairing; fix `min_tokens`; fix the pairing if confirmed; remove the CPU angle
recomputation and re-baseline once; then step 3a). Diagnosis only: **default behaviour unchanged**.
Out of this tier's scope; recorded below under Out-of-tier changes.

**What was built.** `RopePairing` (`ADJACENT`, `SPLIT_HALF`) in `node`;
`LlamaTransformerHandler.rope(..., RopePairing)` (split-half uses the same per-pair frequency and the
same per-step rounding as the adjacent body, which it delegates to); a `ropePairing` field on
`LlamaTransformerHandler` used at all three rotation sites (single decode, batched decode, prefill
window) and set by `ropePairingFor(cfg)`, which returns `ADJACENT` for every architecture; a
`pairing` component on `Qwen3RopeConfig` (default `ADJACENT`) that `Qwen3Rope.apply` dispatches on
for both the plain and the YaRN rotation; package-private test-only loads
`LlamaTransformerHandler.load(path, ctx, backend, RopePairing)` and
`Qwen3TransformerHandler.load(..., RopePairing)`. No system property, no CLI flag.

**Tests.** `RopePairingTest` (5): the adjacent overload is bit-identical to the legacy method;
split-half equals adjacent applied to the head with its halves interleaved, then de-interleaved, bit
for bit at positions 0 to 32767; the same identity through `Qwen3Rope`'s YaRN path; `Qwen3RopeConfig`
defaults to `ADJACENT`. Written first; did not compile against the tree before the change.
`RopePairingPerplexityLiveTest` (3, model-gated, CPU backend): teacher-forced perplexity over a fixed
text, the opening of the United States Declaration of Independence (public domain,
`node/src/test/resources/cab/ml/juno/node/rope-pairing-perplexity.txt`), tokenized raw with no chat
template. `node` cannot depend on `tokenizer` (the dependency runs the other way), so the ids are
checked in beside the text, produced once by `GgufTokenizer.encode`: 821 ids for TinyLlama's
SentencePiece vocabulary (leading BOS included, as the tokenizer emits it) and 701 for the Qwen
vocabulary, which the Qwen2.5 and Qwen3 files share and encode identically (checked byte for byte).

**Result — confirmed.**

| Model | Tokens | Adjacent | Split-half | Winner, margin |
|---|---|---|---|---|
| `tinyllama-1.1b-chat-v1.0.Q4_K_M` (control, `llama`) | 821 | **5.27** | 7617.88 | adjacent, 1447x |
| `qwen2.5-3b-instruct-q4_k_m` (`qwen2`) | 701 | 169.13 | **1.165** | split-half, 145x |
| `Qwen3-1.7B-Q4_K_M` (`qwen3`) | 701 | 525.76 | **4.054** | split-half, 130x |

The control discriminates: adjacent wins on the file whose converter permuted Q/K rows, by three
orders of magnitude, so the method can tell the layouts apart. Both Qwen files are two orders of
magnitude better under split-half. **Juno has served `qwen2`/`qwen2.5` and `qwen3` with the wrong
pair layout.** Short factual prompts survived it (relative position within a short window is still
partly consistent), which is why the "capital of France" check and `Qwen2LiveForwardTest` never
caught it. The Qwen2.5 split-half figure is very low because the text is heavily memorized; the
comparison between layouts, not the absolute value, is the reading. Wall time on this host: about
282 s per TinyLlama pass, 350 to 384 s per Qwen3-1.7B pass, 685 to 711 s per Qwen2.5-3B pass.

`qwen3moe` shares `Qwen3Rope` and the reference implementation assigns it the same layout; it was not
measured (the only file on disk is 30B) and follows the other two in item 3.

**Commands.** `mvn -o -pl node test -Dtest=RopePairingTest`: 5 tests, 0 failures.
`mvn -o -pl node test -Dtest='RopePairingPerplexityLiveTest#control_tinyllama_adjacent_wins'`: 1
test, 0 failures, 567 s. `mvn -o -pl node test -Dtest='RopePairingPerplexityLiveTest#qwen*'`: 2
tests, 0 failures, BUILD SUCCESS, 2146 s.

**Consequence.** Item 3 (the fix) proceeds. Throughput is unaffected (same arithmetic per pair);
generated tokens on both Qwen models change. The live test runs about 36 minutes on this host in its
diagnostic form, too long for every `node` run; item 3 turns it into a regression test that runs
only the production layout with a perplexity ceiling per model.

### 2026-09-27 — follow-up item 2: `min_tokens` holds against the model's own end signals

Scope: "Open for the owner" item 4 above. A sampler and generation-loop change, outside this tier's
scope; recorded below under Out-of-tier changes.

**The gap, as found.** `cc94c53` made generation stop on the vocabulary's chat turn markers by id
(merged into every request's `stopTokenIds` by `GenerationLoop.resolveSamplingParams`) and on role
headers in decoded text (`EosOutputFilter`), but `MinTokenFloor` held back only end-of-sequence and
neither `Sampler.isStopToken` nor the text filter consulted the minimum. The published contract said
"only the end-of-sequence token is held back"; the harness help promised the minimum held "before a
stop token may end the request". Neither matched the code, and they did not match each other.

**The rule now.** Below the minimum the model's own end signals are held back: end-of-sequence, the
vocabulary's turn-marker ids, and a role header or turn marker in the decoded text. A stop the caller
asked for (a stop string, a stop token id, or a stop string that encodes to a single token) still ends
the request below the minimum, as the contract has always said for stop strings; a turn-marker id the
caller names explicitly is therefore not held back. End-of-sequence is held back even if the caller
lists it, as before.

**What changed.** `MinTokenFloor(int eos, int[] alsoHeld, int minTokens)`: masks end-of-sequence and
the extra ids below the minimum, yields only when nothing but held ids survives (the grammar rule,
widened to the set); no allocation per step. `GenerationLoop.minTokenFloor(requested)` builds it from
the request's params as submitted, before `resolveSamplingParams` merges the turn markers in, so the
caller's own stops are known; used by the single-request path, the static batch and
`ContinuousBatchEngine`. `EosOutputFilter.accept(piece, mayStop)`: with `mayStop == false` a complete
marker is emitted as text, a trailing marker prefix is still held back, and a scan floor makes sure a
marker passed over below the minimum is never found later (without it, a rescan that found an
already-emitted marker would try to truncate text that had already been streamed). All three emit
paths pass `!floor.holdsOpen(generatedSoFar)`, including the speculative path through `emitToken`.
Contracts (`openapi.yaml`, `juno-api.yaml`), `SamplingParams` and `JunoHttpClient` javadoc,
`docs/howto.md`, `docs/performance.md` and the `compare-llama-cpp.sh --juno-min-tokens` help now
state the rule above.

**Tests first.** `GenerationLoopMinTokensTurnMarkerTest` (12: four cases on each of the single,
static and continuous paths) was run against the HEAD versions of `GenerationLoop`,
`ContinuousBatchEngine` and `EosOutputFilter`, restored temporarily from `git show HEAD:` and then put
back: **6 failures, all `expected: 5 but was: 0`** - the turn-marker id and the role-header text each
ended the request at zero tokens on all three paths; the caller-stop and no-minimum controls passed,
as they should. The new `MinTokenFloorTest` (5), `SamplerMinTokensTest` (1) and `EosOutputFilterTest`
(8) cases did not compile against the old API (no held-set constructor, no `accept(piece, mayStop)`).
After the change: `mvn -o -pl sampler,coordinator test` BUILD SUCCESS, `sampler` 87 tests (81 + 6),
`coordinator` 342 tests, 0 failures, 1 skipped (322 + 20).

**Verification.** `mvn -o clean install -DskipTests`, then the 11-module
`mvn -o test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,vision,metrics,juno-player
-Dtest='!RopePairingPerplexityLiveTest' -Dsurefire.failIfNoSpecifiedTests=false`: BUILD SUCCESS in
25:57 min, **1762 tests, 0 failures, 0 errors, 46 skipped** (`registry` 93, `lora` 116, `kvcache` 78,
`health` 23, `node` 647/41, `tokenizer` 109/2, `sampler` 87, `coordinator` 342/1, `vision` 95,
`metrics` 61, `juno-player` 111/2): the hand-off 1731 plus item 1's 5 `node` cases and this item's 26.
The one exclusion is item 1's diagnostic A/B, which ran on its own and takes 36 minutes. Quick parity
check on the jar that install built (`compare-llama-cpp.sh --gpu --models
tinyllama,qwen2.5-3b,Phi-3.5-mini,mistral-7b --n-gen 64 --juno-reps 1 --juno-warmup 1 --reps 1
--no-publish --no-tuned-lane`, `target/perf-compare/20260927T074422Z/`): `failures=0`, all five files
(both tinyllama quantizations) at 128/128 prompt tokens, deviation 0, and 64/64 generated tokens.
Unpublished, one repetition, not scored.

### 2026-09-27 — follow-up item 3: Qwen2 and Qwen3 moved to the split-half RoPE layout

Scope: the fix item 1 confirmed. A forward-pass correctness change on a sweep model (`qwen2.5-3b`),
outside this tier's scope; recorded below under Out-of-tier changes.

**What changed.** `LlamaTransformerHandler.ropePairingFor(cfg)` returns `SPLIT_HALF` for `qwen2` and
`qwen2.5` (the rest of the LLaMA family stays `ADJACENT`), used at all three rotation sites.
`Qwen3RopeConfig.PAIRING` is `SPLIT_HALF`, which covers `qwen3` and `qwen3moe` on the plain and YaRN
paths (`Qwen3TransformerHandler` at three sites, `Qwen3MoeTransformerHandler` at one,
`Qwen3LoraTrainableHandler` at three). LoRA: `LoraTrainableHandler` (which serves `qwen2` through
`Qwen2LoraTrainableHandler`) takes its layout from the same `ropePairingFor` at its five forward
sites and its backward; `LoraTrainingMath.ropeBackward(..., RopePairing)` is the split-half adjoint;
`Qwen3Rope.applyBackward` and `LoraTrainingMath.qwen3RopeBackward` follow the config's layout, YaRN
included. Every RoPE site in `src/main` was checked by grep: the Phi-2/Phi-3 paths are split-half on
their own, and the vision text backbones (`llama` for LLaVA, `phi2` for moondream) are unaffected.
**No Qwen adapter exists on this host** (every `.lora` file on disk is a TinyLlama adapter), so no
retraining is owed here; an adapter trained on a Qwen model elsewhere with an earlier build trained
against the wrong layout, which the CHANGELOG says.

**The GPU `RopeKernel` is adjacent-only**, so step 3a (item 5) must keep `qwen2`/`qwen3` on the CPU
rotation until the kernel gains a split-half mode; that fallback is item 5's to implement and announce.

**Tests first.** `RopePairingTest` grew to 9: the production layout per architecture (`llama`,
`mistral`, `tinyllama` adjacent; `qwen2`, `qwen2.5`, Qwen3 config split-half); the split-half backward
is the adjoint of the forward at positions 1, 250 and 30000; the adjacent backward overload is the
legacy one bit for bit; `Qwen3Rope.applyBackward` and `LoraTrainingMath.qwen3RopeBackward` are the
adjoints of `Qwen3Rope.apply` in both layouts with and without YaRN. With the two layout constants
temporarily set back to `ADJACENT`, the production-layout case failed (`[qwen2] expected: SPLIT_HALF
but was: ADJACENT`); restored, it passes. `Qwen2LoraTrainableHandlerTest` gained a zero-adapter
parity case over positions 0 to 4 - the existing one decodes only position 0, where the rotation
angle is zero and no layout difference can show. With the inference handler fixed and the LoRA
forward not yet, it **failed at `pos 1 logit[0]`**, the first position that rotates anything; it
passes after the LoRA change.

**The A/B became a regression test.** `RopePairingPerplexityLiveTest` now runs by default, whenever
the files are present, through the production loader (`ForwardPassHandlerLoader.load`) over the first
128 tokens of the text, against a ceiling per model; the whole-text two-layout comparison is kept
behind `-Djuno.test.ropeAb=true`, which enables a test and reaches no production code. The 128-token
readings, measured with a throwaway probe under both layouts before the ceilings were set:

| Model | Right layout | Wrong layout | Ceiling |
|---|---|---|---|
| `tinyllama-1.1b-chat-v1.0.Q4_K_M` | 2.88 (adjacent) | 467.2 | 6.0 |
| `qwen2.5-3b-instruct-q4_k_m` | 1.29 (split-half) | 23.0 | 4.0 |
| `Qwen3-1.7B-Q4_K_M` | 2.50 (split-half) | 208.9 | 8.0 |

The production-loader run reproduces the right-layout column to four decimals (2.8836, 1.2869,
2.4977) in about 40, 122 and 60 s, so the regression adds roughly four minutes to a `node` run on
this host.

**Verification.** `mvn -o clean install -DskipTests`, then the 11-module `mvn -o test` (nothing
excluded): BUILD SUCCESS in 27:14 min, **1773 tests, 0 failures, 0 errors, 49 skipped** (`node`
658/44: item 2's 647 plus 4 `RopePairingTest` cases, 1 `Qwen2LoraTrainableHandlerTest` case, and the
6 live methods of which the 3 opt-in A/B ones skip by default; every other module as in item 2).
`mvn -o clean verify -pl juno-master`: BUILD SUCCESS, 20 tests, 0 failures (`InProcessClusterIT` 6,
`ThreeNodeClusterIT` 8, `TensorParallelClusterIT` 5, `UnsupportedArchitectureClusterIT` 1), no node
JVM left behind. Quick parity check on the rebuilt jar (`target/perf-compare/20260927T080434Z/`,
same command as item 2): `failures=0`, all five files at 128/128 prompt tokens, deviation 0, and 64/64
generated. One repetition with one warmup is not a throughput reading, but for the record: Juno tg
tinyllama 57.71, mistral-7b 18.66, Phi-3.5-mini 22.60, qwen2.5-3b 34.36 t/s, against the reference
sweep's 56.82, 19.98, 24.42 and 27.83 (qwen2.5-3b rep range there 25.62 to 29.44). The qwen2.5-3b
reading sits above that range although the change does the same arithmetic per pair; item 4's
three-repetition re-baseline measures it properly and is where it will be explained or retracted.
(Item 2's parity directory, `20260927T074422Z`, no longer exists: a root `mvn clean` removes
`target/perf-compare/`. Its figures are the ones recorded in the item 2 section.)

### 2026-09-27 — follow-up item 4: the CPU RoPE angles computed once, and the one re-baseline

Scope: "Open for the owner" item 1 above, placed before step 3 by the owner. A forward-pass change on
every backend, outside this tier's scope, and **a measurement boundary** (README rule 9).

**What changed.** New `RopeTable` (`node`): per `(headDim, base)`, cosine and sine per position,
computed with exactly the expressions `LlamaTransformerHandler.rope` used
(`1.0 / Math.pow(theta, (2.0 * i) / headDim)`, `pos * freq`, `(float) Math.cos`, `(float) Math.sin`),
filled lazily in blocks of 256 positions up to 32768, each block computed aside and published whole
through an `AtomicReferenceArray` (no lock on the read path; racing fillers compute identical values),
tables found through a copy-on-write array scanned without locking. `rope()` keeps its signature and
reads the table in both layouts; a position at or past 32768 is computed in place by the old body.
Covered: every caller of `LlamaTransformerHandler.rope` (Llama-family and Qwen2 inference, non-YaRN
Qwen3, the LoRA forward, the CPU lane of the chain microbench). Not taken (both optional in the plan
and not measured to matter): the YaRN path and `LoraTrainingMath.ropeBackward`.

**Tests first.** `RopeTableTest` (6) did not compile before `RopeTable` existed. It checks the adjacent
and split-half rotations bit for bit against verbatim copies of the pre-table code for 200 positions
in `[0, 32767]` (including 0, 1 and 32767) at every combination of head size 64/96/128 and base
10000/500000/1e6; positions 32768, 32769 and 100000 (computed in place, still identical); one table
per `(headDim, base)`; eight threads racing on cold blocks with zero mismatches; and fewer than 20000
bytes allocated by 20000 warm rotations. The allocation case was shown to fail with a planted
per-call allocation, then the plant was removed. `RopeKernelParityTest` still reads `max|diff| = 0`
against the CPU rotation.

**Verification.** `mvn -o clean install -DskipTests`, then the 11-module `mvn -o test`: BUILD SUCCESS in
24:47 min, **1779 tests, 0 failures, 0 errors, 49 skipped** (`node` 664/44, item 3's 658 plus these 6).

**RoPE share of the GPU forward pass, before and after** (`juno.Rope` over `juno.ForwardPass`,
repetition 2, default lane; before from `20260925T172231Z`, after from `20260927T091155Z`):

| Model | Lane | RoPE before | RoPE after | Share before | Share after |
|---|---|---:|---:|---:|---:|
| tinyllama | generation, 64 tokens | 111.95 ms | 4.07 ms | 9.9% | 0.5% |
| mistral-7b | generation, 64 tokens | 349.34 ms | 10.08 ms | 10.8% | 0.4% |
| qwen2.5-3b | generation, 64 tokens | 192.23 ms | 5.75 ms | 8.0% | 0.3% |
| tinyllama | prefill, 128 tokens | 216.07 ms | 6.57 ms | 23.5% | 0.9% |
| qwen2.5-3b | prefill, 128 tokens | 343.51 ms | 9.16 ms | 18.1% | 0.6% |
| mistral-7b | prefill, 128 tokens | 682.16 ms | 19.24 ms | 23.4% | 0.9% |

Phi-3.5-mini emits no `juno.Rope` span in either sweep (its rotation is `Phi3Rope`), as before.

**Perf gate: `compare-lora.sh --gpu --reps 3 --baseline cc94c53`** (`docs/perf-compare/20260927T090436Z-lora/`;
the harness builds the baseline in a temporary detached `git worktree`, removed afterwards; no commit,
branch or index change): train 41000 ms against 44000 ms, **ratio 0.932** (7% faster; the gate
`train >= 0.95x` in speed terms passes), playback wall-clock **1.075x** (gate `>= 0.80x`), recall
true. Against the last published baseline (`20260924T201548Z-lora`, 45000 ms, 12.15 t/s): 0.911 and
1.083. The baseline ref is `cc94c53` rather than that run's `1f90b68` so the comparison isolates this
session's working tree.

**The re-baseline** (`compare-llama-cpp.sh --gpu` and `--cpu`, `--reps 3 --juno-reps 3
--juno-warmup 2`, published). GPU `20260927T091155Z`, `failures=0`, every row 128/128 and 64/64. Four
rows spread over 15% of their median and were re-run in `20260927T093054Z` (tinyllama, qwen2.5-3b,
Phi-3.5-mini, both lanes): Phi-3.5-mini came back clean, three rows still exceed the rule there
(qwen2.5-3b default generation 23%; one repetition four to five times faster than the other two in
the tinyllama default and qwen2.5-3b tuned prefill lanes - 898 and then 819 t/s against about 168 on
the same tinyllama lane in both runs). The medians agree between the two runs (tinyllama pp 170.5 and
168.2, tg 70.1 and 71.5; qwen2.5-3b tg 30.8 and 31.1; qwen2.5-3b tuned pp 86.8 and 79.6) and the
median of three is robust to one such reading, so the rows are published with the spread stated
rather than re-run a third time; **the recurring fast prefill repetition is a harness question left
for the owner**. CPU `20260927T094414Z`, `failures=0`, every row within 1% spread.

| Model | GPU tg before -> after (t/s) | GPU pp before -> after | GPU tg ratio after | CPU tg before -> after | CPU tg ratio after |
|---|---|---|---:|---|---:|
| tinyllama | 56.82 -> 71.54 | 138.3 -> 168.2 | 0.375x | 3.42 -> 3.39 | 0.135x |
| qwen2.5-3b | 27.83 -> 31.08 | 66.1 -> 80.4 | 0.445x | 1.15 -> 1.14 | 0.094x |
| Phi-3.5-mini | 24.42 -> 24.57 | 43.2 -> 45.5 | 0.415x | 0.94 -> 0.93 | 0.095x |
| mistral-7b | 19.98 -> 23.40 | 43.9 -> 59.1 | 0.646x | 0.51 -> 0.53 | 0.090x |

(tinyllama, qwen2.5-3b and Phi-3.5-mini GPU from the re-run.) The README's "Program target" table
has a new column for these readings; the 2026-09-25 column stays. `20260925T172231Z` and
`20260925T174146Z` are marked superseded-as-reference in their INDEX files and in
`docs/perf-compare/README.md`, where neither had been listed until now. The qwen2.5-3b generation
reading item 3's single repetition flagged (34.4 t/s) does not reproduce as a three-repetition median
(31.08, within 1% of the first sweep's 30.83), so it was noise, not an effect of the layout change.

**The chain microbench re-read** (`docs/perf-compare/20260927T115107Z-tier01-resident-chain/`): the CPU
chain is 0.0049 ms at decode (from 0.0670) and 2.13 ms at prefill (from 34.06). Against it the
resident chain is **0.26x at decode width and 1.00x at prefill width** (median 1.002x, lanes' min-max
overlapping: parity within noise), and **0.58 / 0.55 of op-at-a-time**. Read against the Threshold
block: decode-width `>= 1.0x` **missed**; prefill-width `>= 1.0x` met only at parity; chaining
`<= 0.7x` **met** at both widths. Per this tier's contingency, a decode-width miss with the
prefill-width figure at or above 1.0x is **a partial result recorded and handed to
[Tier 01B](TIER-01B-prefill-throughput.md) item 2, not a downgrade**. The case for residency now rests
on the device-only lane (13.6x the CPU at prefill, 0.45x at decode): a region pays only when it spans
enough operations to amortise its one entry and exit, which is what step 3a is built to test.

### 2026-09-27 — implementation step 3a (follow-up item 5): the decode region wired, and step 4

Scope: this tier's own step 3 as the owner re-scoped it on 2026-09-27 - "3a", one residency region
per layer at decode covering norm, the Q/K/V projection and RoPE - then step 4's measurement. Step 3b
(KV append and attention inside the region) was **not** taken: the plan assigns attention residency to
Tier 02 and this prompt says to ask before taking it into Tier 01. It is the owner's call.

**What was built.**
- `ResidentQkvPath` (new, `node`): per layer, upload the residual row, `CudaRmsNorm.normalizeResident`
  with the layer's norm weight uploaded once at load, `Q4KMmqKernel.quantizeX` then `launchPacked` for
  W_q, W_k and W_v on the chain's stream with a **chain-owned** Q8_1 scratch
  (`ResidentChain.allocateScratch`, so no kernel on another stream can overwrite it mid-use),
  `CudaRope.applyResident` on q and k, and `ResidentActivation.materializeRows` for q, k and v with one
  wait. `unsupportedReason` names what it cannot run for a model (non-CUDA; split-half RoPE; Q/K/V
  biases; query width different from hidden size); `eligible(li)` is false for a layer without all
  three K-quant projections on the device. Issued under the context's serialization lock, as the
  matrix-vector path's work is.
- `LlamaTransformerHandler`: builds it at load when `--gpu-residency` is requested, after the device
  uploads; `transformerLayer` (single-sequence decode) takes q, k and v from the region when the
  layer is eligible, and otherwise from `normProjectRope`, which is the previous code moved into a
  method unchanged. Prefill windows and batched decode (`transformerLayerBatch`) are not wired, which
  the activation log line states.
- `GpuResidencyOptions` (new, public): `--gpu-residency on|off|auto` / `JUNO_GPU_RESIDENCY`, default
  **off**; `announceUnsupported` logs once per surface; `consoleNotice` prints a console warning at
  startup when the launch cannot use the region, because the console front end turns library logging
  off unless `--verbose` (found in this pass: without it, every declined surface would have been a
  silent no-op to a normal console or API user). Surfaces that decline: the CPU backend, `qwen2` /
  `qwen2.5` (split-half, biases), `phi2`, `phi3`, `qwen3`, `qwen3moe` (other handlers,
  `ForwardPassHandlerLoader`), LoRA training and `--lora-play` (`LoraTrainingHandlerFactory`). ROCm:
  `unsupportedReason` declines any non-CUDA context - **NEEDS-AMD-HARDWARE** to exercise.
- CLI: `ConsoleMain --gpu-residency` (validated at parse time), `scripts/run.sh` `local` and `cluster`,
  `ClusterHarness` forwards the property to forked nodes, `compare-llama-cpp.sh --gpu-residency`
  pass-through (help text, JSON metadata).

**A per-thread design that would have leaked, caught by the smoke test.** The first version kept one
device region per calling thread (`ThreadLocal`), as the plan text suggested. The request scheduler
starts **one new virtual thread per request** (`RequestScheduler`: `Thread.ofVirtual().name("gen-" +
requestId)`), so every request would have opened fresh regions on every in-process node and never
freed the old ones. The smoke test showed it (about 1 MiB more per request with the region on than
off); `ResidentQkvPathTest.shortLivedThreadsDoNotAccumulateRegions` then failed on it (51 regions
after 50 short-lived threads) before the fix: device regions now come from a pool bounded by
concurrent calls, and only the small host result arrays are per thread.

**Pre-existing, found by the same test, not fixed (the owner's to place).** With the region **off**,
the server's GPU memory grows on every request: about 23 MiB per request on tinyllama and 114 MiB on
mistral-7b in local mode (`nvidia-smi`, per process). The cause is the same pattern: `CudaMatVec`
keeps its device scratch - FP32/FP16 staging, the Q4_K dequant scratch, and a CUDA stream - in
`ThreadLocal`s, and each request runs on a new thread, so each request allocates a fresh set that is
never released. On an 8 GiB card that bounds how many requests a mistral-7b server can serve before
device memory runs out. It changes `CudaMatVec`, a hot path, so it needs its own tests and perf gate;
it is recorded here rather than fixed in this pass.

**Tests first.**
- `GpuResidencyOptionsTest` (6), `ResidentQkvPathTest` (8), `LlamaTransformerHandlerGpuResidencyTest`
  (2) did not compile before the classes existed.
- `ResidentQkvPathTest` pins the region **bit for bit** to the GPU op-at-a-time path (round-trip GPU
  norm, `sgemvSameX` over the same K-quant matrices, CPU rotation) at positions 0, 1, 17, 511 and 30000,
  and across three concurrent threads; checks the input is never written; declines a layer without
  device projections; allocates no device memory per call; returns device memory across 300
  create-run-close cycles (bound 4 MB: a deliberately planted leak of every region measured 48 MB over
  300 cycles, while the leak-free reading moves by at most 256 KB either way, not with the cycle count;
  the planted leak made the test fail, then was removed); and pools regions across short-lived threads.
- `LlamaTransformerHandlerGpuResidencyTest` on the real tinyllama file with a CUDA backend: the region
  active on 22 of 22 layers, **the same greedy token at all 24 decode positions**, largest logit
  difference **0.237** against the flag-off run (bound 0.5). Not bit-identical, and not expected to
  be: the handler's default decode normalizes on the CPU (`rmsNormGpu` is deliberately null), so the
  GPU norm's summation order moves a Q8_1 rounding of the projection input, compounded over 22 layers.
  On qwen2.5-3b the flag declines (split-half) and decode runs as before.

**Live, smoke and cluster checks.**
- `ModelLiveRunnerIT` with `JUNO_GPU_RESIDENCY=on` (forked nodes inherit it): **tinyllama passes**,
  pipeline and tensor. **mistral-7b fails at tensor-parallel shard loading** ("Tensor-parallel shard
  loading failed: UNKNOWN: Application error processing RPC") - and **fails identically with the flag
  unset**, so it is pre-existing: tensor mode currently loads the full model on each of three nodes
  (`TensorShardContext`: "geometry only"), three copies of a 4 GB model on an 8 GiB card. Separately,
  **the documented command `mvn verify -pl juno-master -Pintegration -DMODELS=...` runs no test at
  all**: the profile's empty `<excludes />` does not clear the default execution's exclusion of
  `ModelLiveRunnerIT` (Maven merges the element), so failsafe runs nothing and reports BUILD SUCCESS.
  Adding `-Dit.test=ModelLiveRunnerIT` makes it run. Pre-existing; not fixed here.
- `scripts/performance-tests/smoke-gpu-residency.sh` (new; the plan's name
  `smoke-tier01-gpu-residency.sh` would put a tier number into `docs/howto.md`, which `CLAUDE.md`
  forbids, so it was renamed). Final run on the final build
  (`target/gpu-residency-smoke/20260927T135113Z/`, 4 requests per mode, 32 generated tokens, local
  mode with three in-process nodes, idle device): **failures=0**.

  | Model | Region | Greedy output on vs off | GPU MiB after requests 1-4, off / on | Growth 2..4, on vs off |
  |---|---|---|---|---|
  | tinyllama | active; the node the smoke quotes: 8 of its 8 layers (22 of 22 in the single-handler test) | identical | 936 960 984 1006 / 936 960 984 1006 | 46 vs 46 |
  | mistral-7b | active; the node the smoke quotes: 11 of its 11 layers | identical | 4602 4716 4832 4946 / 4604 4718 4832 4946 | 228 vs 230 |
  | llama-1-30b | active on 20, 3 and 1 layers of the three nodes' 20 each (read from the logs of the first run of this smoke), the rest not on the device | identical | 7726 7762 7762 7774 / 7680 7758 7758 7764 | 6 vs 12 |

  The llama-1-30b row is the per-layer fallback working: the file does not fit the card, so the GPU
  layer policy places some layers on the device, the region runs on exactly those, and the host path
  runs the others, with identical greedy output. The growth column compares from the second request,
  after first-use buffers are sized and, at the card's capacity, placement has settled; the growth
  itself is the pre-existing per-request leak described above, the same with the region on or off.
  (The run directory under `target/` was removed by the root `mvn clean` of the verification build
  that followed; the figures above are the script's own summary output, kept in this session's
  record.) Cluster, tinyllama, region on: pipeline and tensor both answer with **exactly local mode's output**
  (activations cross the gRPC boundary as host arrays; nothing device-resident leaves a node), and no
  node JVM is left after either.
- Console notice verified by launch: `./juno local` on Phi-3.5-mini with `--gpu-residency on`, and
  `./juno lora` with `JUNO_GPU_RESIDENCY=on`, each print one yellow warning naming why the region does
  not run there; tinyllama prints none.

**Step 4 - measurement.** GPU sweep with `--gpu-residency on`, `--reps 3 --juno-reps 3
--juno-warmup 2`, published as `docs/perf-compare/20260927T131355Z/`, `failures=0`, every row 128/128 and
64/64; the tinyllama tuned (43%) and mistral-7b default (21%) rows spread over 15% and were re-run in
`20260927T133246Z/` (4% and 2%). Against the item 4 reference:

| Model | Region | tg reference | tg with region | Ratio | Threshold `>= 0.95x` |
|---|---|---:|---:|---:|---|
| tinyllama | active | 71.54 | 75.38 | **1.054** | pass |
| tinyllama tuned | active | 72.05 | 74.88 | 1.039 | pass |
| mistral-7b | active | 23.40 | 24.24 | **1.036** | pass |
| mistral-7b tuned | active | 23.11 | 23.70 | 1.025 | pass (first run; its re-run spread 19%) |
| qwen2.5-3b | declined | 31.08 | 30.77 | 0.990 | pass |
| Phi-3.5-mini | declined | 24.57 | 24.43 | 0.994 | pass |

JFR agrees on the mechanism: with the region on, decode emits no `juno.Rope` span, half as many
`juno.RmsNorm` spans (1408 against 2816 on tinyllama) and one `juno.MatVec` fewer per layer (4443
against 5851). The reference was taken earlier the same day rather than interleaved, so a few percent
of the difference could be host drift; the gate does not depend on it.
The chain microbench, re-run on the final code (unpublished; the primitive's code paths are those of
`20260927T115107Z-tier01-resident-chain`, with two additive methods): decode 0.26x the CPU chain and
0.52 of op-at-a-time, prefill 0.995x and 0.55 - the published reading holds, prefill at parity within
noise (1.002x there, 0.995x here). Device memory not returned: 0 bytes.
`compare-lora.sh --gpu --reps 3 --baseline cc94c53` with `JUNO_GPU_RESIDENCY=on`
(`docs/perf-compare/20260927T134122Z-lora/`): train **0.932** of the baseline's time, playback
**1.092x**, recall true - the LoRA handlers decline the region, and the console now says so.

**Verification.** `mvn -o clean install -DskipTests`, then the 11-module `mvn -o test`: BUILD SUCCESS
in 25:46 min, **1795 tests, 0 failures, 0 errors, 49 skipped** (`registry` 93, `lora` 116, `kvcache`
78, `health` 23, `node` 680/44, `tokenizer` 109/2, `sampler` 87, `coordinator` 342/1, `vision` 95,
`metrics` 61, `juno-player` 111/2): item 4's 1779 plus the 16 residency cases. `mvn -o clean verify
-pl juno-master`: BUILD SUCCESS, **20 tests, 0 failures** (`InProcessClusterIT` 6,
`ThreeNodeClusterIT` 8, `TensorParallelClusterIT` 5, `UnsupportedArchitectureClusterIT` 1), no node
JVM left.

**Open for the owner - decisions this pass surfaced and did not take.**

1. *Step 3b* - KV append and attention inside the region, downloading only the attention output. The
   bigger decode lever (the attention path's four synchronous uploads and the KV mirror's two copies
   per layer are still there); assigned to Tier 02 by the plan, so not taken without asking.
   *Decided 2026-09-27: not in Tier 01; Tier 02 owns it (its scope item 4).*
2. *The primitive-threshold box.* Decode width `>= 1.0x`: missed (0.26x against the cheaper CPU
   chain). Prefill width: parity within noise (1.002x, 0.995x). Chaining `<= 0.7x`: met at both widths.
   The contingency classes this as a partial result for Tier 01B item 2, not a downgrade; the wired
   region nonetheless makes decode 3.6% to 5.4% faster end to end, because it spans a projection as
   well. Ticking the box on that reading, or leaving it open, is the owner's call.
   *Decided 2026-09-27: ticked under the contingency, a partial result handed to Tier 01B item 2, not a
   downgrade.*
3. *`CudaGraphSession`* stays explained and not wired (nothing measured it worthwhile);
   `CudaRmsNorm` is now live on the resident path behind the flag. The "no longer dormant scaffolding"
   box turns on whether that satisfies it.
   *Decided 2026-09-27: re-scoped - `CudaGraphSession` moves to Tier 02 (scope item 5) with a
   wire-or-delete decision rule; the box is ticked on that Scope amendment.*
4. *Default of `--gpu-residency`.* Off, as planned until measured. Measured now: +3.6% to +5.4% where
   it runs, neutral elsewhere, identical greedy output on three models. Whether to make `auto` the
   default is a product decision.
   *Decided 2026-09-27: keep `off`; re-decided after 3b lands, on the larger re-measured gain (Tier 02
   scope item 6).*
5. *Pre-existing, found this pass, not fixed:* the per-request device-memory growth from
   `CudaMatVec`'s thread-local scratch under one-new-thread-per-request scheduling (about 114 MiB per
   request on mistral-7b); mistral-7b's tensor-parallel start failing on an 8 GiB card; the documented
   `-Pintegration` command running no test; the recurring fast prefill repetition in the comparison
   harness (item 4). Each needs a home.
   *Decided 2026-09-27: the memory growth is fixed in this tier as an out-of-tier change; the
   `-Pintegration` profile is fixed now; the tensor-parallel start failure gets a clearer error now and
   is recorded in Tier 09; the fast prefill repetition is handed to Tier 01B (implementation step 0).*

### 2026-09-27 - owner decisions on the step 3a pass

The owner decided the five items the step 3a pass left open:

| # | Question | Decision |
|---|---|---|
| 1 | Step 3b (KV append + attention inside the region) | **Not in Tier 01. Tier 02 owns it.** |
| 2 | Primitive-threshold exit box (decode width missed, prefill at parity, chaining met) | **Tick it**, with the contingency reading recorded: a partial result handed to Tier 01B item 2, not a downgrade. |
| 3 | "No longer dormant scaffolding" box | Owner delegated the choice between (a) "tick, CudaGraphSession explained-not-wired" and (c) "re-scope". **Decided: (c), re-scope to Tier 02.** |
| 4 | `--gpu-residency` default | **Keep `off`.** Re-decide after 3b lands, on the larger re-measured gain. |
| 5 | The four pre-existing problems | Memory leak fixed now; `-Pintegration` fixed now; tensor-parallel start failure recorded in Tier 09 with a clearer error now; fast prefill repetition handed to Tier 01B. |

Why (c) and not (a): the box offers exactly two outcomes - wired live, or the tier marked
partial-complete - and "explained, not wired" is neither, while decision 2 rules out partial-complete.
Execution rule 2 accepts "out of scope for this tier" only when the Scope section says so, and Scope item
2 named `CudaGraphSession`; so the Scope is amended (see In scope item 2 and Out of scope) and the class
moves to Tier 02, where a captured graph can actually pay (attention inside the region multiplies the
launches per wait). It is not deleted: Tier 02's decision rule decides wire-or-delete on a measurement.

### 2026-09-27 - close-out: the owner's decisions carried out

Scope: the five owner decisions above. Task order as the owner set it: plan-tree records, the
per-request device-memory fix, the `-Pintegration` profile, the tensor-parallel start error, the fast
prefill repetition handed to Tier 01B, then this close-out. Every code change here is outside this
tier's scope and is in the "Out-of-tier changes" table.

**Plan tree.** This file's Scope (item 2 note, three Out-of-scope bullets), both remaining exit boxes,
and the "Open for the owner" list of the step 3a section; [Tier 02](TIER-02-attention-long-context.md)
scope items 4 to 6 with implementation step 6, tests, thresholds and exit boxes;
[Tier 01B](TIER-01B-prefill-throughput.md) item 2's hand-off note and step 0's third item with its test
bullet; [Tier 09](TIER-09-tensor-parallelism-multi-gpu.md) scope item 1's known-limitation note.

**Per-request device memory.** The prompt named four `ThreadLocal`s in `CudaMatVec` and three in
`RocmMatVec`; `CudaGqaAttention.SCRATCH` (on by default through `--gpu-attention`) and
`CudaRmsNorm.SCRATCH` held device memory the same way. Design, confirmed by reading every use: all of
`CudaMatVec`'s scratch and stream use is inside `GpuContext.cublasSerializationLock()`, and each call
synchronizes its stream before leaving it, so the lock alone makes one scratch set per instance safe -
no pool needed there. The attention path and the round-trip norm run outside that lock (synchronous
copies on the default stream), so concurrent callers need separate entries: `DeviceScratchPool`, sized
by concurrent callers. `DeviceScratchBudget` reserves one dequantized matrix per backend; per-thread
scratch broke that assumption with every new thread and the instance-owned scratch restores it.
Tests first, each shown failing on the old code for the right reason: `CudaMatVecScratchLifetimeTest`
(scratch grew about 0.58 MB per thread, 2.9 to 31.9 MB over 50 threads; device-wide 44,040,192 bytes
over 60 threads) and `CudaAttentionNormScratchLifetimeTest` (+13.7 KB attention and +57 KB norm per
thread; device-wide 79,691,776 bytes over 60 threads, once the shapes were enlarged so the device-wide
reading could resolve it). Leak-free drift 0 in most runs, at most 2.4 MiB either way; bound 12 MiB.
Smoke, 8 requests, local mode, three in-process nodes, idle device: tinyllama 938 MiB after every
request with the region off and on (before: 936 -> 1006 over 4), mistral-7b 4604 off / 4608 on (before:
4602 -> 4946), cluster pipeline and tensor pass. llama-1-30b, which fills the card, settles and holds
(16 requests: off 7768 from request 7 with one 12 MiB step at 16, on 7784 from request 4); its first
8-request run failed only the smoke's old "on grows no more than off from request 2" comparison while
the two modes settled at different paces, so the check was tightened as the prompt allowed: over the
second half of the requests each mode may grow at most 8 MiB per request and on no more than off
(`docs/howto.md` says so). Replayed on the recorded series it passes every post-fix run and fails the
pre-fix tinyllama (46 MiB against 16) and mistral-7b (230 against 16) series.

**The perf gate caught a regression in the first version, and its cause.** The first published sweep
(`20260927T214616Z`, now marked superseded) read GPU prefill 7% to 9% below the reference on tinyllama,
qwen2.5-3b and mistral-7b with generation unchanged. A same-hour A/B against a build without the change
confirmed it (prefill 0.84x to 0.93x); a bisect build with only `CudaMatVec` reverted was as fast as
the baseline; per-phase timing inside `sgemmQ4KBatchedGemm` put the time in the host loop that packs
the activation window to FP16 (17 to 38 ms per matmul instead of about 1 ms, with the upload call
itself at 0.1 ms), and the JIT log shows that method compiled and made not entrant on uncommon traps
repeatedly, in both builds. The single-thread microbenchmark never reproduced it. The fix does not
depend on which branch trips the trap: the loop is now `packFp16Rows`, compiled on its own, shared by
the three batched paths. A/B after it, same hour: prefill tinyllama 181 -> 229 t/s, qwen2.5-3b 83 -> 90,
mistral-7b 61 -> 65 - above the old code, which had been paying the same cost intermittently. The
exact trigger of the traps is inferred, not proven; the extracted loop removes the sensitivity either
way.

**Gates on the final build.** Quick parity check: 128/128 and 64/64 on every sweep model,
`failures=0`. GPU sweep `20260927T232837Z`, with the `qwen2.5-3b` tuned and both `Phi-3.5-mini` rows
from the re-run `20260927T234659Z`: every row passes `>= 0.95x`; pp 1.10x to 1.34x on tinyllama,
qwen2.5-3b and mistral-7b, 1.02x to 1.04x on Phi-3.5-mini; tg 0.962x (tinyllama, tight spread; decode
does not touch the changed loop) to 1.021x. The re-run's `qwen2.5-3b` default row spread 23.84 to 31.88
t/s in generation although its first-run row was clean; recorded, not re-run a third time.
`compare-lora.sh --gpu --reps 3 --baseline cc94c53` (`20260927T235655Z-lora`; the harness created and
removed its temporary detached `git worktree`, as last session): train 0.909 of the baseline's time,
playback 1.082x. A measurement boundary for GPU prefill; see the table row.

**`-Pintegration`.** Before: `mvn -o clean verify -pl juno-master -Pintegration -DMODELS=<tinyllama>`
printed BUILD SUCCESS and no `Running` line. After `combine.self="override"`: `Running
cab.ml.juno.master.ModelLiveRunnerIT`, `Tests run: 1, Failures: 0`. The default verify still runs
exactly 20 (6 + 1 + 8 + 5). The documented command in `CLAUDE.md` works verbatim; `CLAUDE.md` is
unchanged. The `gpu` profile had the same defect; fixed, it now runs `GpuForwardPassIT` for the first
time and **2 of its 4 fail** on an absolute 0.03 tolerance against GPU values around 134 (worst 0.29,
about 0.2%): a stale tolerance for today's FP16 and K-quant paths, not introduced here. **Owner
decision 2026-09-27: keep the profile fix and record the failing test as open**; the owner then chose
to rebuild the test now (option B, see the table row): it judges agreement by relative L2 error,
cosine similarity, top-1 and top-5 logits, and a 16-token greedy decode over the whole model, each
bound calibrated between the measured value and a planted fault. It passes 5 of 5 on tinyllama and
mistral-7b; greedy output is identical GPU against CPU on both.

**Tensor-parallel start error.** Verbose node logs showed the real cause: each forked node dies with
`java.lang.OutOfMemoryError: Java heap space` in `GgufReader.tensorRaw` while constructing
`LlamaTransformerHandler` (forked nodes get `-Xmx4g`), an `Error` that `loadShard`'s `catch (Exception)`
did not see, so gRPC turned it into `UNKNOWN`. Tests first: the node-side case failed with exactly
`UNKNOWN: Application error processing RPC`, the client-side one because the message named no node.
Now `ModelLiveRunnerIT` on mistral-7b fails with: "Tensor-parallel node 0 did not load its shard: Model
load failed on node node-1 (layers 0-32, model mistral-7b-instruct-v0.1-q4_k_m.gguf):
java.lang.OutOfMemoryError: Java heap space" - the expected outcome until Tier 09.

**Verification** (final build, `mvn -o clean install -DskipTests` first). The 11-module `mvn -o test`:
BUILD SUCCESS in 25:34 min, **1806 tests, 0 failures, 0 errors, 49 skipped** (`registry` 93, `lora`
116, `kvcache` 78, `health` 23, `node` 690/44, `tokenizer` 109/2, `sampler` 87, `coordinator` 342/1,
`vision` 95, `metrics` 61, `juno-player` 112/2): the hand-off's 1795 plus this pass's 11 (10 `node`, 1
`juno-player`), skip count unchanged. The first full run stopped at `metrics` on one failure of
`JfrMetricsExtractorJdkEventsTest.monitorContentionAndParkTimeAreSummed` ("a 120ms held monitor produced
no JavaMonitorEnter event"); `metrics` is untouched by this pass, the test passed three isolated re-runs
and the whole second run, so it is recorded as timing-sensitive rather than fixed. `mvn -o clean verify
-pl juno-master`: **20 tests, 0 failures** (6 + 1 + 8 + 5). `mvn -o clean verify -pl juno-master
-Pintegration -DMODELS=<tinyllama>`: `ModelLiveRunnerIT` **1 test, 0 failures**; no node JVM left.

**Housekeeping.** The keepalive subshells leaked by this pass's smoke runs (Tier 01B scope item 5) were
killed; those left by the previous session's runs were not touched. The A/B builds lived under the
session scratchpad, not in the tree.

### Out-of-tier changes (recorded per execution rule 9)

Three changes touching hot-path, generation or launcher behaviour landed while this tier was in
progress and outside its scope. None was in any tier's plan; all are recorded here because the value
of this plan tree is that it knows what was measured, when, and against which build. The third
shipped inside a commit that was otherwise this tier's own, which is why the pass that made that
commit did not record it and the next pass did. The rows after those three are the follow-up items
the owner approved on 2026-09-27 (see the "follow-up item" sections above); they were planned, but
not as this tier's scope, so they are recorded here as well.

| Commit | What it changed | Measurement boundary? |
|---|---|---|
| `1f90b68` | Both launchers (`scripts/run.sh`, `scripts/run.bat`) derive the JVM heap from the model file size instead of a fixed 4 GB, so a large model no longer dies with an `OutOfMemoryError` naming a tensor. `--heap` and `HEAP` still win. | **Yes, for launcher-driven runs only.** `compare-llama-cpp.sh` builds its own `java_args` and does not shell the launcher, so no llama.cpp ratio moved. Every `./juno`-driven measurement did, including `compare-lora.sh` and every smoke script. Do not compare a launcher-driven run taken before this commit against one taken after. |
| `c91f879` | Retired a device KV mirror by closing it in place rather than unmapping it, so an empty replacement is no longer read as history; gated attention on a written-prefix watermark, per mirror instead of per handler. Touched `DeviceKvCache`, `LlamaTransformerHandler`, `CudaMatVec`, `DeviceScratchBudget`, `Q4KDequantScratch`, with `DeviceKvMirrorWatermarkTest` and `DeviceScratchBudgetTest`. Published three `compare-lora.sh` runs (`docs/perf-compare/20260924T184429Z-lora/`, `20260924T185248Z-lora/`, `20260924T201548Z-lora/`). | **Correctness fix on the GPU attention path; gated and published.** The three runs compare a working tree against its own `HEAD` (`1f90b68`), which is why both columns name the same commit. Later tiers reading those directories should know that is deliberate, not a harness bug. |
| `cc94c53` (the part that is not the step-1 harness) | Generation now stops on a chat template's role headers and turn markers, by decoded text and by token id: `ChatTurnMarkers` (new, `tokenizer`), `Tokenizer`/`GgufTokenizer.chatTurnTokenIds`, `EosOutputFilter`, `OpenAiAdapter.mergeStopTokenIds`, and `GenerationLoop.resolveSamplingParams`, which merges the turn-marker ids into every request's stop set on all three generation paths; `Phi3TokenizerLiveTest`, `EosOutputFilterTest`, `GenerationLoopEosPieceTest`. CHANGELOG Session 93. It shipped inside the step-1 commit and the step-1 record does not mention it; recorded here in the step-2 pass, the first to find it. | **Not a measurement boundary, checked rather than assumed**: two unpublished harness runs at this HEAD reproduce the reference sweep's parity on all four sweep models - `qwen2.5-3b` and `Phi-3.5-mini`, whose markers are control tokens, and `tinyllama` (both files) and `mistral-7b`, where the risk is the new text-level role-header stop (128/128 prompt tokens, 64/64 generated, `finish_reason: length`). **It opened a gap in `min_tokens`**: `MinTokenFloor` held back only the end-of-sequence id and `Sampler.isStopToken` did not consult the minimum, so a request could end below its `min_tokens` on a turn-marker id or a role header in text. Not seen on the benchmark prompt. **Closed 2026-09-27** by follow-up item 2 (see its section and the row below). |
| Working tree, 2026-09-27 (follow-up item 1) | `RopePairing`, a pair-layout parameter on `LlamaTransformerHandler.rope` and `Qwen3RopeConfig`, test-only forced-layout loads, `RopePairingTest`, `RopePairingPerplexityLiveTest` with its text and id fixtures. Every architecture still gets `ADJACENT`. | **Diagnosis only; default unchanged; not a measurement boundary.** Every production call resolves to the same arithmetic as before (the adjacent overload delegates to the old body, bit-identical by test). It found that `qwen2` and `qwen3` are served with the wrong layout (see "follow-up item 1"); the fix is item 3. |
| Working tree, 2026-09-27 (follow-up item 2) | `MinTokenFloor` holds back the vocabulary's turn-marker ids with end-of-sequence, minus the caller's own stops; `EosOutputFilter` does not end below the minimum and emits the held text; `GenerationLoop.minTokenFloor` on all three generation paths; contracts, javadoc, `docs/howto.md`, `docs/performance.md`, harness help; `GenerationLoopMinTokensTurnMarkerTest` and new cases in `MinTokenFloorTest`, `SamplerMinTokensTest`, `EosOutputFilterTest`. | **Not a measurement boundary.** It changes output only for a request with `min_tokens > 0` whose model proposes a turn marker below the minimum; the benchmark prompt never did on any sweep model (checked at `cc94c53`, and re-checked after this change by the quick parity run recorded in the item 2 section). No baseline invalidated. |
| Working tree, 2026-09-27 (follow-up item 3) | `qwen2`/`qwen2.5` (`LlamaTransformerHandler.ropePairingFor`) and `qwen3`/`qwen3moe` (`Qwen3RopeConfig.PAIRING`) moved to split-half RoPE, forward and LoRA backward, plain and YaRN; `LoraTrainingMath.ropeBackward(..., RopePairing)`; `RopePairingPerplexityLiveTest` as a regression with ceilings; new `RopePairingTest` and `Qwen2LoraTrainableHandlerTest` cases. | **Correctness fix, not a throughput boundary.** Same arithmetic per pair, so no throughput figure moves; generated tokens change on every Qwen model, including `qwen2.5-3b` in the sweep (its generated text in any earlier run is not comparable, its t/s is). Quick parity check after the change: see the item 3 section. |
| Working tree, 2026-09-27 (follow-up item 4) | `RopeTable`; `LlamaTransformerHandler.rope` (both layouts) reads it; `RopeTableTest`. Bit-identical output. | **Yes - a measurement boundary.** GPU throughput rose 11% to 23% (tg) and 22% to 35% (pp) on tinyllama, qwen2.5-3b and mistral-7b; Phi-3.5-mini and every CPU figure held. **Invalidates as reference**: `20260925T172231Z` (GPU) and `20260925T174146Z` (CPU), superseded by `20260927T091155Z` + `20260927T093054Z` (GPU) and `20260927T094414Z` (CPU); `20260924T201548Z-lora` as the LoRA baseline, superseded by `20260927T090436Z-lora`; `20260927T025430Z-tier01-resident-chain` as the chain's CPU-relative reading, superseded by `20260927T115107Z-tier01-resident-chain` (its op-at-a-time ratio still stands). Do not score across this boundary. |
| Working tree, 2026-09-27 (owner decision 5: per-request device memory) | `CudaMatVec` and `RocmMatVec` keep their device scratch (FP32/FP16 staging, `Q4KDequantScratch`, pinned host staging) and their stream per **instance** instead of per thread: every use was already under `GpuContext.cublasSerializationLock()` and each call synchronizes its stream before releasing it, so one set serves every caller (`releaseScratch`, `scratchDeviceBytes`). `CudaGqaAttention` (on by default with `--gpu-attention`) and `CudaRmsNorm.normalizeBatch` run outside that lock, so their scratch comes from the new `DeviceScratchPool`, sized by concurrent callers; `LlamaTransformerHandler.releaseGpuResources` closes the attention pool. The batched FP16 pack loop moved into its own method (`packFp16Rows`), found necessary by the perf gate (see the close-out section). Tests: `CudaMatVecScratchLifetimeTest` (5), `CudaAttentionNormScratchLifetimeTest` (4), one `RocmMatVecTest` case (NEEDS-AMD-HARDWARE). | **Yes - a measurement boundary for GPU prefill.** pp rose 10% to 34% on tinyllama, qwen2.5-3b and mistral-7b (the batched K-quant path), 2% to 4% on Phi-3.5-mini; tg within the gate (0.962x to 1.021x); per-request GPU memory flat. **Invalidates as reference**: `20260927T091155Z` + `20260927T093054Z` (GPU), superseded by `20260927T232837Z` + `20260927T234659Z`; `20260927T090436Z-lora` as the LoRA baseline, superseded by `20260927T235655Z-lora` (train 0.909, playback 1.082x). `20260927T214616Z` and `20260927T213908Z-lora` were taken on an intermediate build and are marked superseded. The CPU reference `20260927T094414Z` is untouched (CPU path unchanged). |
| Working tree, 2026-09-27 (owner decision 5: `-Pintegration`) | `juno-master/pom.xml`: the `integration` and `gpu` profiles' empty `<excludes />` became `<excludes combine.self="override" />`, so they no longer inherit the default execution's exclusion of the very test they exist to run; the profile comment's `-pl integration` corrected to `-pl juno-master`. | **Not a measurement boundary** (build configuration only). The documented `mvn verify -pl juno-master -Pintegration -DMODELS=...` now runs `ModelLiveRunnerIT`; the default verify still runs exactly the 20 stub ITs. Exposed: `-Pgpu` now runs `GpuForwardPassIT`, which failed 2 of 4 on an absolute 0.03 tolerance; rebuilt, see the next row. |
| Working tree, 2026-09-27 (owner decision after close-out: `GpuForwardPassIT` rebuilt) | `juno-master/.../GpuForwardPassIT.java`, test only. The two element-wise tests (absolute `within(0.03)`, TinyLlama shapes hard-coded) replaced by agreement measures with calibrated bounds: hidden state after the first half of the layers, relative L2 at most 0.005 (measured 0.00222 tinyllama / 0.00036 mistral-7b; planted 1% gain 0.00785, one head zeroed 0.0127) and cosine at least 0.99995 (measured 0.9999998 / 1.0; one head zeroed 0.9999212); logits of the second half, the same top-1, top-5 overlap at least 4 (measured 5 / 5) and relative L2 at most 0.025 (measured 0.00904 / 0.00751; 5% gain 0.0515); and a new 16-token greedy decode over the whole model that must match the CPU token for token (identical on both). Shapes read from the GGUF, the prompt from its tokenizer; every GPU handler released; stale javadoc corrected. 5 of 5 pass on tinyllama and mistral-7b. | **Not a measurement boundary** (test only). It is the GPU-against-CPU oracle; [Tier 01B](TIER-01B-prefill-throughput.md) item 0 extends it per architecture. |
| Working tree, 2026-09-27 (owner decision 5: tensor-parallel start error) | `EmbeddedNodeServer.loadShard` reports an `Error` during load (not only an `Exception`) as a failed load naming the node, the layers, the model file and the cause, instead of letting it escape as gRPC `UNKNOWN`; a package-private `ShardLoader` seam for the test. `TensorParallelPipelineClient.loadShards` names the node index and address when a node answers with a bare error status. Tests: one `EmbeddedNodeServerLoadFailureTest` case, one `TensorParallelPipelineClientTest` case. | **Not a measurement boundary** (load-failure reporting only). Placement and pre-flight are unchanged; the underlying limitation is Tier 09's (see its scope item 1 note). |

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
      marked superseded. The one precondition still absent **of the six this criterion covered** is a
      thread-count control reaching the hot path, which belongs to the CPU hot-path tier and is
      recorded in every published index as a stated mismatch rather than a silent one. A seventh
      precondition — pre-tokenizer parity — was added to the list after this box was ticked and carries
      its own box below rather than re-opening this one.*
- [x] **Precondition 7 (pre-tokenizer parity) landed**: `tokenizer.ggml.pre` is read and dispatched
      on, the enumeration of declared pre-types across `models/` is recorded in this file, a file
      declaring an unimplemented pre-type is rejected at load with an error naming it, and a file with
      no such key still loads bit-identically to today. If any sweep model's `prompt_tokens` for the
      benchmark prompt changed, the reference re-baseline is re-taken on top of it and the 2026-09-25
      sweeps are marked superseded; if none changed, that finding is recorded and the reference stands.
      Tests first, in the `tokenizer` module, and `-pl tokenizer` is in this tier's test command —
      which the documented `mvn test -pl ...` line does cover, unlike `vision` and `metrics`.
      *Hoisted from [Tier 04B](TIER-04B-tokenizer-fidelity.md) items 1 and 3 so that no tier publishes
      a pp ratio whose denominator a later tier moves. That tier keeps its items 2 and 4.*
      *Landed 2026-09-26 as `BpePreTokenizer` plus `GgufTokenizer`'s dispatch: the sixteen files on
      disk declare six distinct values and the enumeration is the table in the execution record above.
      `qwen2` and `llama-bpe` are implemented and reach full token-ID parity with a second engine
      across a 33-line probe corpus on all three files that declare them (`qwen2.5-3b` and
      `Qwen3-1.7B` from five divergent lines to zero, `Meta-Llama-3.2-1B` from eight to zero);
      `tekken`, `minimax-m2` and `qwen35` are rejected at load by name, verified on the three real
      files and end to end through the launcher. The five files declaring nothing produce identical
      token IDs before and after on every line of the same corpus. **No sweep model's benchmark
      `prompt_tokens` changed, so the 2026-09-25 reference sweeps stand** — three of the four are
      SentencePiece and cannot be affected, and the fourth's benchmark prompt contains none of the
      constructs the split moves; confirmed by a harness run at `128/128`, deviation 0, at the same
      calibrated word count. Tests first: `BpePreTokenizerTest` (12 cases) did not compile against the
      original tree, and `PreTokenizerParityLiveTest` failed 3 of 7 on it — the two whitespace cases
      and the digit-grouping case — before the fix.*
- [x] Residency primitive implemented, unit-tested, and documented (what it is, where the
      materialization boundary is, which ops participate).
      *Landed 2026-09-27 as `ResidentChain` (the region: one stream and the buffers on it; closing
      it frees them all) and `ResidentActivation` (the buffer; host and device meet only at `upload`
      and `materialize`), with RMS norm (`CudaRmsNorm.normalizeResident`) and RoPE (`CudaRope`, over
      the new `rope.cu` kernel) as the operations that run on it. `ResidentActivationTest` (8 cases)
      asserts the two-operation chain against the scalar path, that host arrays are never written
      through, and an exact return of device memory per allocate-and-close cycle, together with the
      same query seeing the allocation, so the no-leak check can fail; its upload-race case was shown
      to fail without the guard it tests. `RopeKernelParityTest` (6) is bit-identical to the CPU
      rotation; one new `CudaRmsNormTest` case pins the resident path to the round-trip path bit for
      bit. `node` is 642 tests, 0 failures, 41 skipped (620 plus these 22, skip count unchanged).
      Documented in `docs/agent-arch.txt`. As this box's earlier note warned, `DeviceActivationBatch`
      is unrelated. Nothing is wired into a handler yet; that is implementation step 3.*
- [x] RMSNorm + RoPE measured *faster* than CPU scalar (or at minimum, no longer the ~7x-slower
      finding from Phase B) with residency, on real GTX 1080 hardware, **at both decode width (batch 1)
      and prefill width (batch 512), reported separately**, published in `docs/perf-compare/`. **Contingency, decided before Tiers 02/06/07 start**: this project has
      already shelved three closely-related bets on grounds that turned out to be exactly this kind
      of per-op dispatch overhead (`CudaRmsNorm`/`CudaGraphSession` itself, draft-model speculative
      decoding, `VectorQuantKernels.dot()`), so a fourth negative result is a real possibility, not a
      formality. **Both widths in the threshold are read before this contingency fires**: a decode-width
      miss with a prefill-width pass is a partial result to record and hand to Tier 01B item 2, not a
      tier downgrade. If the measured result is still worse than CPU scalar after the residency primitive
      is correctly wired (not just "not yet wired right"), do not iterate indefinitely — document the
      measurement, what was tried, and why it still regresses, then downgrade Tier 01 to
      **partial-complete**: the residency primitive itself (item 1) and the correctness guarantees
      below still ship and close out, but the "no longer dormant scaffolding" bullet below is
      explicitly waived for this pass, and Tiers 02/06/07 proceed using today's op-at-a-time GPU path
      for any sub-item that isn't itself residency-dependent, with their own residency-specific
      sub-items marked `NEEDS-TIER-01-REVISIT` rather than blocked indefinitely. Escalate to the user
      at that point rather than silently re-scoping — this changes three other tiers' scope, not just
      this one's.
      *The **before** side of this comparison now exists and is published:
      `docs/perf-compare/20260926T060301Z-tier01-rmsnorm-roundtrip/`, taken by
      `RmsNormRoundTripMicrobench` at both widths on the GTX 1080 — decode 0.09x (10.6x slower than
      scalar CPU), prefill 0.62x (1.60x slower), every row scorable. Both figures confirm the prose
      claims this tier was built on. The box stays unticked because it asks for the residency result,
      which needs implementation steps 2 to 5. The `<= 0.7x` chaining threshold in this tier's
      Threshold block cannot be read until `RopeKernel` exists, since there is no second GPU op to
      chain today; its op-at-a-time baseline is taken in the same pass that builds it.*
      *Primitive-level reading taken 2026-09-27 and published
      (`docs/perf-compare/20260927T025430Z-tier01-resident-chain/`), the op-at-a-time baseline with
      it: the resident chain is 3.65x the scalar CPU chain at decode width and 15.86x at prefill
      width, and costs 0.51 and 0.55 of op-at-a-time, so every number in the Threshold block that can
      be read before wiring passes. The box stays unticked for two reasons. Implementation step 4 is
      where this tier re-reads it, on the wired path and beside the end-to-end gate. And the
      decode-width `>= 1.0x` leans on the scalar RoPE recomputing its angles on every call - about
      95% of the CPU chain - so it would probably not survive that CPU cost being removed, while the
      chaining result, which isolates residency, would. See "Open for the owner" in the execution
      record: whether that CPU cost is removed before step 3 is the owner's call, because it moves
      the baseline this box is read against.* **Tier 01B is in that set, and is its most affected member.** An earlier draft
      excused it on the grounds that prefill is dominated by large-batch GEMM and host-device staging
      rather than per-op dispatch overhead — but host-device staging is precisely what residency
      removes, and Tier 01B's largest scope item is built on this primitive. If this tier downgrades,
      Tier 01B's item 2 does not proceed on a substitute design; it escalates. Its other items (the
      `--gpu-attention` architecture coverage, the JFR breakdown, chunk sizing, residual attention)
      proceed unchanged on today's GPU path.
      *Re-read 2026-09-27 after the CPU RoPE table and on the wired path
      (`20260927T115107Z-tier01-resident-chain`, re-run on the final code): decode width 0.26x
      (missed), prefill width 1.002x / 0.995x (parity within noise), chaining 0.52 to 0.58 and 0.55
      (met). Per the contingency above, a partial result recorded and handed to Tier 01B item 2, not a
      downgrade; the wired region itself is 3.6% to 5.4% faster end to end. Left unticked for the
      owner's reading - see the step 3a section's "Open for the owner".*
      *Ticked by the owner on 2026-09-27 under the contingency: decode width missed, prefill width at
      parity, chaining met; a partial result handed to [Tier 01B](TIER-01B-prefill-throughput.md)
      item 2 (see that tier), not a downgrade. The wired region is +3.6% to +5.4% end to end.*
- [x] No correctness regression: greedy decode output identical (CPU) or within tolerance (GPU)
      with the new path enabled vs. disabled, across all three cross-surface-listed models.
      *2026-09-27, step 3a: greedy output **identical** over 32 generated tokens with the region on and
      off on tinyllama, mistral-7b and llama-1-30b (the last with the region on only its device
      layers), `smoke-gpu-residency.sh`, failures=0. Handler level: the same greedy token at all 24
      positions on tinyllama, largest logit difference 0.237, the GPU norm's summation order; the
      region itself is bit-identical to the GPU op-at-a-time path. CPU path untouched by construction
      (the region needs a CUDA backend and declines otherwise).*
- [x] Cluster (pipeline- and tensor-parallel) smoke tests confirm activations still correctly
      materialize at the process/AllReduce boundary — no stale or device-resident data crossing a
      gRPC call.
      *2026-09-27: with the region on, tinyllama pipeline and tensor clusters answer with exactly
      local mode's output and leave no node JVM; `ModelLiveRunnerIT` passes on tinyllama with
      `JUNO_GPU_RESIDENCY=on`. mistral-7b's tensor-parallel start fails on this 8 GiB card with the flag
      on and off alike (pre-existing; recorded in the step 3a section).*
- [x] LoRA train + playback smoke tests unaffected.
      *2026-09-27: `compare-lora.sh --reps 3` with `JUNO_GPU_RESIDENCY=on`: train 0.932 of the
      baseline's time, playback 1.092x, recall true (`20260927T134122Z-lora`); the LoRA handlers
      decline the region and the console says so.*
- [x] `docs/agent-arch.txt`/`docs/performance.md`/`docs/howto.md` updated (Juno-native language).
      *2026-09-27: `ResidentQkvPath`, `GpuResidencyOptions`, the chain's scratch and multi-row
      materialize in `docs/agent-arch.txt`; the flag and the smoke test in `docs/howto.md`; the wired
      region's measurement in `docs/performance.md`. No competitor names or tier numbers added.*
      *Done for all three passes so far, and unticked only because the box also covers the residency
      work, which has not started. The harness passes added `JfrMetricsCli` to the `metrics` entry in
      `docs/agent-arch.txt`, its invocation to `docs/howto.md`, and both measurement boundaries to
      `docs/performance.md`. The precondition-7 pass added a `BpePreTokenizer` entry to
      `docs/agent-arch.txt` and a "Pre-tokenizer splits" section to `docs/howto.md` stating what each
      declared value does and that an unimplemented one is refused; `docs/performance.md` needed
      nothing, since no published figure moved. `README.md` needed nothing either — it names the
      `tokenizer` module in one table row and links model support to the external documentation site,
      neither of which this changes. The step-1 pass added `RmsNormRoundTripMicrobench` to
      `docs/agent-arch.txt`'s `node` entry and the harness's invocation to `docs/howto.md`, and
      recorded the re-measured round-trip cost in `docs/performance.md` next to the Phase B
      checkpoint whose claim it reproduces. The step-2 pass added the residency primitive, the
      materialization boundary and the two participating operations to `docs/agent-arch.txt`
      (`ResidentChain`, `ResidentActivation`, `KernelParams`, `RopeKernel`, `CudaRope`,
      `ResidentChainMicrobench`, with `CudaRmsNorm` and `CudaGraphSession` rewritten and their two
      planning-file pointers removed), the chain harness to `docs/howto.md`, and the chain reading
      plus the scalar RoPE's measured share of the forward pass to `docs/performance.md`. The box
      stays unticked until the wired path is documented as well.*
- [x] `CudaGraphSession`/`CudaRmsNorm` are no longer "dormant scaffolding" — either wired live
      (preferred, if the measurement confirms the fix), or the tier is explicitly marked
      **partial-complete** per the contingency above (not silently marked complete with the
      scaffolding still dormant and unexplained).
      *Ticked 2026-09-27. `CudaRmsNorm` is wired live: the resident norm in `ResidentQkvPath`, behind
      `--gpu-residency` (default off by owner decision). `CudaGraphSession` is explicitly re-scoped to
      [Tier 02](TIER-02-attention-long-context.md) (scope item 5) in this tier's Scope section by the
      owner (execution rule 2), with a wire-or-delete decision rule there - so neither is dormant and
      unexplained.*
- [x] Full `mvn test`/`mvn verify -pl juno-master` pass with zero regressions.
      *2026-09-27, close-out, final build: 1806 tests, 0 failures, 0 errors, 49 skipped; `juno-master` 20
      tests, 0 failures; `-Pintegration` on tinyllama 1 test, 0 failures (see the close-out section,
      including one timing-sensitive `metrics` failure in a first run that did not recur).*
      *2026-09-27, after step 3a, from a clean install: 1795 tests, 0 failures, 0 errors, 49 skipped;
      `juno-master` 20 tests, 0 failures.*
      *Both halves pass as of the precondition-7 pass: `mvn test` across all eleven modules is
      **1674 tests, 0 failures, 0 errors, 46 skipped** in 23:32 min, and `mvn -o clean verify -pl
      juno-master` is **20 tests, 0 failures, 0 errors**. The count is 1646 plus 19 new `tokenizer`
      cases and the 9 `metrics` cases the previous pass added before counting them; the skip count is
      unchanged, so no new model-gated case skipped. No rerun flag was needed — the flaky attention
      assertion is fixed. Unticked because the box covers the tier, whose residency work has not
      shipped and will need both commands re-run.*
      *Re-run after the step-1 pass: `mvn -o test` is **1697 tests, 0 failures, 0 errors, 46
      skipped** (the same 1674 plus 23 new `node` cases, skip count unchanged) and
      `mvn -o clean verify -pl juno-master` is **20 tests, 0 failures, 0 errors**.*
      *Re-run after the step-2 pass, from a clean install: `mvn -o test` is **1731 tests, 0 failures,
      0 errors, 46 skipped** in 22:46 min (1697 plus this pass's 22 `node` cases plus 12 cases from
      the turn-marker change recorded under Out-of-tier changes) and `mvn -o clean verify -pl
      juno-master` is **20 tests, 0 failures, 0 errors**. Still unticked: the box covers the wired
      path, which will need both commands again.*
- [x] `CHANGELOG.md` entry added.
      *2026-09-27, close-out: Session 97 (per-request device memory, the prefill packing, the
      `-Pintegration` profile, the tensor-parallel load error), alongside Sessions 95 and 96.*
      *2026-09-27: Sessions 95 (the follow-up items) and 96 (the wired region).*
      *Entries covering the passes so far are in (Sessions 88, 90, 91, 92 and 94; the earlier note
      listed only the first three, omitting Session 92, the step-1 harness). Unticked because the box
      covers the tier, whose residency work is not yet wired into a handler.*
