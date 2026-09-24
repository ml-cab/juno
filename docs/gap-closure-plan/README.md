# Juno gap-closure plan

Source material: [`../llama-cpp-gap-analysis.md`](../llama-cpp-gap-analysis.md) (2026-09-18 snapshot,
branch `67-inference`, HEAD `0c519f1`). That document is raw analysis; this tree turns it into a
sequenced, testable implementation plan. Re-verify any cited line number against current source
before acting on it — both documents are snapshots, not ground truth that stays accurate forever.

## Execution rules (binding for every tier)

1. **One tier at a time, to feature-complete.** Do not start the next tier's work until the current
   tier's exit criteria (bottom of its file) are all checked off. "Next" means the next row of the
   tier index below, not the next integer — the index is the running order. Three rows break integer
   order: Tier 01B sits between 01 and 02, Tier 04B between 04 and 05, and Tier 08 runs before Tier
   06. Partial, half-wired features are not acceptable stopping points between tiers.
2. **No surface left aside.** A tier is not complete until its change has been carried through
   every product surface it touches — see "Cross-surface compatibility checklist" below. If a
   surface can't reasonably support the new feature yet (e.g. continuous schedule doesn't support
   per-request LoRA), the tier must make that an explicit, documented, fail-closed rejection for
   that surface, not a silent gap. "Out of scope for this tier" is only acceptable when stated
   explicitly in the tier's Scope section and the surface fails closed rather than silently
   degrading.
3. **Tests first, and they must cover the real matrix.** For every tier: write or extend the unit
   and integration tests *before* writing the implementation. Before marking any tier complete, run
   the full smoke matrix (CPU, CUDA GPU, LoRA train+play, vision, static schedule, continuous
   schedule, single-node local, pipeline-parallel cluster, tensor-parallel cluster) across every
   architecture family Juno claims to support, using real model files, and confirm zero regressions
   against the prior tier's baseline. See "Test infrastructure" below for the concrete mechanics.
   When a tier needs a model file that isn't present under `models/`, stop and ask — see
   [`INVENTORY.md`](INVENTORY.md) for what's already on disk and what's missing.
4. **No competitor product names in Juno-facing docs.** Never write `llama.cpp`, `vLLM`,
   `llama-server`, or any other competitor product name into anything outside this
   `docs/gap-closure-plan/` tree — not `docs/howto.md`, `docs/agent-arch.txt`, `README.md`,
   `CHANGELOG.md`, `docs/performance.md`, code, comments, CLI help text, or error messages. This
   plan tree itself may name them (it exists to compare against them). Filenames/script names that
   already exist under `docs/perf-compare/` and `scripts/performance-tests/` (e.g.
   `compare-llama-cpp.sh`) are pre-existing and out of scope for renaming; don't add new ones.
5. **The plan tree is self-contained.** When a tier's work requires updating production docs
   (`docs/howto.md`, `docs/agent-arch.txt`, `README.md`, `docs/performance.md`), those docs must
   read standalone — do not add "see `docs/gap-closure-plan/TIER-NN-...`" pointers into them. Only
   `docs/perf-compare/` entries may point back into `docs/performance.md` (existing convention);
   nothing may point into this plan tree from outside it.
6. **Cross-feature / product-surface compatibility.** Every tier's exit criteria include a
   cross-feature smoke table (same format `docs/performance.md` already uses for prior tiers) that
   exercises the new feature *combined with* the other major surfaces (LoRA play, vision, grammar,
   tools, cluster/TP, static/continuous), not just in isolation. A feature that only works alone is
   not feature-complete.
7. **Performance thresholds are numeric, and anchored to llama.cpp where the tier is performance-
   relevant.** Every tier's perf-gate bullet states a concrete pass/fail number for whatever new
   metric it introduces — "measured, published, no unexplained regression" alone is not enough for a
   metric with no prior baseline to be implicitly anchored to. See "Test infrastructure" below for
   the specific llama.cpp-relative gate this adds on top of the existing `compare-lora.sh` rule.
   **This rule is machine-checked, not trusted.** Tier 14's `smoke-tier14-doc-consistency.sh` greps
   this tree and fails if any tier file containing the string `Perf gate` does not also contain a
   `**Threshold` block carrying at least one numeral and a comparison operator. It matches on
   `Perf gate`, not `Perf gate (required)`, because three spellings of that heading are already in
   use and keying on the longest one would let the other two through. Tier 14's own file is excluded
   from the scan, since it names the string to describe the check. The rule was added once and
   immediately drifted — at the time this enforcement was written only six of seventeen tier files
   stated a threshold at all, and Tiers 11 and 12 both declared a required perf gate with no number.
8. **Doc/comment claim audits are repo-wide, not scoped to a named list of files.** When a tier's job
   is to fix a doc/code drift item (Tier 00's §2.x items, Tier 14's full audit), the fix is not
   complete until the same class of claim has been grepped for across `src/main` and `CHANGELOG.md`,
   not just in whichever file the gap analysis happened to cite. A rule stated once in `CLAUDE.md`
   (e.g. "no internal tier numbers in shipped code") is binding everywhere that rule applies, not
   just in the specific files a prior audit pass already looked at.

## Program target

This plan's stated purpose is closing the gap with llama.cpp. That purpose needs a number, or no
tier can fail on it and the final scorecard in Tier 14 can only report a direction of travel.

**Program exit target**, measured by `scripts/performance-tests/compare-llama-cpp.sh` on the standing
four-model sweep (`tinyllama-1.1b`, `qwen2.5-3b`, `Phi-3.5-mini`, `mistral-7b`, all Q4_K_M) on the
`docs/perf-compare/README.md` baseline host, under the benchmark-parity preconditions below:

| Metric | Target at end of plan | Reading when this plan was written |
|---|---|---|
| GPU tg, Phi-3.5-mini | >= **0.50x** | 0.330x |
| GPU tg, mistral-7b | >= **0.60x** | 0.513x |
| GPU pp, every sweep model | >= **0.15x** | see the caveat below — not 0.016x to 0.031x |
| CPU tg, every sweep model | >= **0.25x** | 0.106x to 0.147x |

**The pp row's starting reading is not yet known, and the target is provisional until it is.** The
0.016x-to-0.031x figures widely quoted from `docs/perf-compare/20260918T204809Z/` were taken with
`raw_prompt: 0`, meaning llama-bench prefilled 128 tokens while Juno prefilled 20 to 30 — they are
not a like-for-like prefill measurement (see "Benchmark parity preconditions" below). The only
parity-corrected prefill figures on record are worse and narrower: at a real 512-token prompt,
`20260915T043143Z` puts mistral-7b at 0.0166x and tinyllama at 0.0075x, and the best GPU prefill
number anywhere in this repository — TinyLlama at 119.56 t/s with `--gpu-attention on` and
`--prefill-batch 32`, `20260916T040113Z-prefill/` — is about **0.028x** of llama.cpp's 4225 t/s on
the same shape. Tier 01's re-baseline establishes the real starting reading under parity; **the
0.15x target is re-derived from that number in Tier 01's own file before any later tier is scored
against it.** A target anchored to a measurement that was never like-for-like is not a target.

**Intermediate milestones**, so the trend is checkable before Tier 14 rather than only at the end:

| After | Milestone |
|---|---|
| Tier 01B | GPU pp >= 0.10x on mistral-7b at `n_prompt=512`, and pp no longer degrades with prompt length |
| Tier 04 | GPU tg >= 0.40x on Phi-3.5-mini |
| Tier 10 | CPU tg >= 0.20x on every sweep model |

The GPU pp milestone is by far the least certain number in this table, and the parity correction
makes it harder to reach, not easier. Measured like-for-like, the best prefill
configuration ever recorded here sits at roughly 0.028x, so 0.10x is a **three- to four-fold
improvement over the best number this project has ever produced** — not a four-to-six-fold
improvement over a 0.0166x reading that was partly a measurement artefact. The other milestones
demand roughly one and a half to two and a half fold. It is stated at that level deliberately
because prefill is the dominant gap, not because it is known to be reachable. Missing it and
reporting the miss plainly is an acceptable outcome for this plan; not having a number to miss is
not.

**Relationship to `docs/infra-plan/`.** This plan tree replaces it. The infra tree grew to 36
documents across four numbering schemes (Infra tiers 1-20, phases P0-P3, standalone `PROMPT-*`
prompts, and a routing document that points between them), and tracking what owned what became more
expensive than the planning was worth. `docs/gap-closure-plan/` is the deliberate clean start: this
tree is the live plan, and no tier here defers to, routes into, or waits on anything in
`docs/infra-plan/`.

The infra tree stays on disk as **evidence, not governance**. It holds the root-cause JFR analysis,
the dated record of what was measured and when, and the reasoning behind numbers this plan starts
from — all of which is worth citing. Cite it for measurements and history; do not cite it for
ownership, sequencing, or gates. In particular, the table above is this plan's own target and does
not need reconciling with that tree's P0 gate; it happens to adopt the same Phi-3.5 figure because
that figure was derived from real measurement on this host, not out of deference.

Two mechanical consequences. Tier numbers are not shared between the trees, so cite an infra
document by filename and never by its tier number. And the measurement thresholds this plan relies
on (the `compare-lora.sh` train/playback ratios, the `compare-vision.sh` latency/tps ratios, the
rule that a new CLI flag is added to `compare-llama-cpp.sh`'s pass-through set in the same change
that ships it) are restated where this plan uses them rather than incorporated by reference, so no
tier here has to go read the infra tree to know what gate it is being held to.

## Tier index and ordering rationale

Ordering: correctness-and-consistency first, then the shared architectural root cause behind three
independent measured regressions, then the largest measured gap in the repository (prompt
processing), then outward through the feature surface in the order a request actually flows
(attention/context → KV → quantization → tokenization → sampling → model coverage → speculative
decoding → scheduling → parallelism → backend breadth → vision → LoRA → server/cluster surface),
closing with a documentation hardening pass.

Two rows do not sit where their integer would put them, and the table below — not the numbering —
is the running order (execution rule 1). **Tier 01B** sits between 01 and 02, and **Tier 04B**
between 04 and 05. **Tier 08 runs before Tier 06**: Tier 06 adds a `forwardVerify` override per
architecture, so running it first would mean adding verify support to four handlers and then having
Tier 08 introduce four more that either need the same work again or silently lack it. Tier 08 first
means Tier 06 covers every handler in one pass. Tier 08's own file previously argued the opposite
ordering while sitting after 06; that rationale is corrected there.

| Tier | Title | Gap analysis refs |
|---|---|---|
| [00](TIER-00-correctness-and-consistency.md) | Correctness & consistency audit | §2.1, §2.4, §2.5, §2.6, §2.7, §2.9 |
| [01](TIER-01-gpu-activation-residency.md) | GPU activation-residency redesign | §1.6, §2.8 |
| [01B](TIER-01B-prefill-throughput.md) | Prefill throughput | none — see that file's "Why this tier, why now" |
| [02](TIER-02-attention-long-context.md) | Attention & long context | §1.2 |
| [03](TIER-03-kv-cache-maturity.md) | KV cache maturity | §1.3 |
| [04](TIER-04-quantization-coverage.md) | Quantization coverage (mapped weight loading first) | §1.1 |
| [04B](TIER-04B-tokenizer-fidelity.md) | Tokenizer fidelity | none — see that file's "Why this tier, why now" |
| [04C](TIER-04C-packed-weight-matmul.md) | Packed-weight matmul (dequantize-to-FP16 elimination) | none — adjacent to §1.1; see that file's "Why this tier, why now" |
| [05](TIER-05-sampling-grammar.md) | Sampling & grammar completeness | §1.4 |
| [08](TIER-08-model-architecture-breadth.md) | Model architecture breadth | §1.9, real files in `models/` |
| [06](TIER-06-speculative-decoding.md) | Speculative decoding expansion | §1.5 |
| [07](TIER-07-continuous-batching.md) | Continuous batching maturity | §1.3 (scheduling half) |
| [09](TIER-09-tensor-parallelism-multi-gpu.md) | Tensor parallelism & multi-GPU | §1.8, §2.2 |
| [10](TIER-10-gpu-backend-breadth-cpu-simd.md) | GPU backend breadth & CPU hot path (SIMD, allocation, threading) | §1.7 |
| [11](TIER-11-vision.md) | Vision | §1.10 |
| [12](TIER-12-lora.md) | LoRA | §1.11 |
| [13](TIER-13-server-surface-clustering.md) | Server surface & clustering | §1.12, §2.3, §2.6 |
| [14](TIER-14-documentation-hardening.md) | Documentation hardening | §3 |

Also see [`INVENTORY.md`](INVENTORY.md) for the model/hardware inventory referenced by every tier.

## Cross-surface compatibility checklist (the rubric every tier applies)

Every tier's exit criteria must state, for each row below, either **PASS** (verified, cite the
smoke run), **N/A** (the tier's change genuinely cannot touch this surface — state why), or
**FAIL-CLOSED** (the surface correctly rejects the new capability with an explicit error, and that
rejection is itself tested).

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference (scalar) | `--gpu-layers 0` |
| 2 | CUDA GPU inference | GTX 1080 available locally |
| 3 | ROCm GPU inference | no AMD hardware available locally — see [`INVENTORY.md`](INVENTORY.md) for how this is gated |
| 4 | Static schedule | default schedule, `--parallel` micro-batching |
| 5 | Continuous schedule | `--schedule continuous`, local mode only today |
| 6 | Single-node local mode | `./juno local` |
| 7 | Pipeline-parallel cluster | `./juno cluster --pType pipeline` |
| 8 | Tensor-parallel cluster | `./juno cluster --pType tensor` |
| 9 | LoRA training | `./juno lora` |
| 10 | LoRA playback | `--lora-play` |
| 11 | Vision | `--mmproj-path`, `/v1/vision/chat`, local mode only today |
| 12 | OpenAI-compatible REST surface | `/v1/chat/completions`, streaming, tools, grammar |
| 13 | Native REST surface | `/v1/inference`, `/v1/inference/stream` |
| 14 | CLI direct usage | `./juno local`/`cluster`/`lora`/`merge`/`lora-import`/`gguf-info`/`test` |

**Row 1 caveat (CPU inference) before Tier 10 lands:** every tier from 00 through 09, Tier 01B
included, uses row 1 as a *correctness* oracle only — `CpuMatVec`'s scalar path, not a SIMD path,
since Tier 10 is what makes CPU inference actually fast. A tier's row-1 "PASS" before Tier 10 means
"correct," not "final-performance-path verified." Tier 10 re-verifies, rather than assumes, that its
CPU changes preserve every earlier tier's row-1 correctness result — vectorized float accumulation
can legitimately reorder floating-point sums vs. the scalar path, and so can a different thread count
or work-splitting strategy, which Tier 10 now also changes — see that tier's own exit criteria.

## Test infrastructure

Three layers, all of which get extended (never replaced) tier over tier:

1. **Unit tests** in the owning module (`mvn test -pl <module>`). Each tier's file names the exact
   test classes to add cases to or create. Note that `CLAUDE.md`'s own quick-reference `mvn test -pl
   tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,juno-player` command omits two
   real modules with their own test suites — `vision` (relevant from Tier 08 onward, primary from
   Tier 11) and `metrics` (the JFR-extractor tests every perf-gate change ultimately depends on).
   Any tier touching either module must explicitly add `-pl vision` / `-pl metrics` to its own test
   commands rather than assuming the documented command covers them; Tier 00 corrects the documented
   command itself.
2. **Integration tests**: `ModelLiveRunnerIT` (`juno-master/src/test/java/cab/ml/juno/master/`,
   run via `./juno test --model-path ...`, 8 checks today — 6 pipeline-parallel + 2 tensor-parallel)
   and the forked-JVM cluster ITs (`ThreeNodeClusterIT`, `TensorParallelClusterIT`, `mvn verify -pl
   juno-master`). Every tier that changes forward-pass, batching, or cluster behavior adds a new
   check to `ModelLiveRunnerIT` rather than only relying on unit tests — this is the one test class
   that runs against a real model file end to end.
3. **Bash smoke scripts** under `scripts/performance-tests/`, following the existing
   `smoke-grammar.sh` / `smoke-tools.sh` convention: one `smoke-tierNN-<short-name>.sh` per tier,
   written in bash, that drives `./juno local`/`cluster` against the real model matrix (see
   `INVENTORY.md`) and asserts on process exit code, HTTP status, and response shape — not just
   "does it crash," but "is the output the specific thing this tier promised." Each smoke script
   must be runnable standalone and re-run (unmodified) by every later tier as a regression check —
   later tiers only ever *add* a new `smoke-tierNN` script, never edit an earlier one except to fix
   a bug in the script itself.

Performance regression gates stay governed by the existing rule in the root `CLAUDE.md`
("Performance gates on hot-path changes"): any tier touching the forward pass, MatVec, GPU
residency, batching, or KV paths runs `scripts/performance-tests/compare-lora.sh` (and
`compare-vision.sh` if vision/CLIP is touched) against the last published baseline in
`docs/perf-compare/`, and publishes a new timestamped result directory there — that mechanism
already exists and this plan doesn't change it, it just says explicitly, per tier, when it applies.

**llama.cpp-relative gate (applies from Tier 01 onward).** `compare-lora.sh`/`compare-vision.sh`
only ever compare Juno against its own prior baseline — neither tells you whether the actual stated
goal of this plan (closing the gap with llama.cpp) is moving. Every tier whose scope includes the
forward pass, MatVec, GPU residency, batching, quantization, or KV paths (at minimum: Tiers 01, 01B,
02, 03, 04, 04B, 04C, 06, 07, 08, 09, 10) additionally re-runs
`scripts/performance-tests/compare-llama-cpp.sh` on the same host/model/quant/flags as the last
published run under `docs/perf-compare/`, and records the resulting Juno/llama.cpp tg and pp ratios
in that tier's own file (not just in `docs/performance.md`, so the trend across tiers is visible
from the plan tree itself), **against the program target and the intermediate milestone for that
tier** (see "Program target" above). The script itself is not a new mechanism — it already exists
with a multi-month history of published runs — it just has to actually be invoked per tier instead
of only opportunistically, and now has a number to be read against. Tier 14's exit criteria add a
final consolidated scorecard summarizing this trend across every tier (see that file).

**Benchmark parity preconditions (blocking — must land before any tier publishes a llama.cpp
ratio).** As `compare-llama-cpp.sh` stands today, the two engines are not measured comparably, and
every ratio this plan is about to collect would inherit the difference:

| Asymmetry, as of this writing | Effect on the ratio |
|---|---|
| **llama-bench prefills `-p ${N_PROMPT}` tokens; Juno prefills a fixed 57-character sentence unless `--raw-prompt` is passed, and `RAW_PROMPT` defaults to `0`** | **the largest asymmetry, and it lands on the metric this plan cares most about. In the headline run (`docs/perf-compare/20260918T204809Z/`, `raw_prompt: 0`) llama-bench prefilled 128 tokens while Juno prefilled 20, 20, 21 and 30. The published pp ratio is a batch-width mismatch plus a cold-start penalty as much as a throughput gap** |
| llama-bench runs `-r 3` (plus its own internal warmup); Juno is measured from a **single** `/v1/chat/completions` request of `n_gen` tokens | C2 compilation of the entire forward pass sits inside Juno's measurement window; the reported Juno number is a cold-JVM number |
| llama-bench gets `-t ${N_THREADS}`; Juno has **no** thread-count control that reaches the hot path (`-Djuno.simd.pool.size` is built but the kernels dispatch on `ForkJoinPool.commonPool()`) | the two engines run at different parallelism — commonPool defaults to one fewer than `availableProcessors()` |
| `COMPARE_HEAP` is derived automatically from model size | heap, and therefore GC behaviour, changes between models and between runs of the same model as code changes |
| Four JFR creation sites disagree: `run.sh`'s `cmd_test` uses `settings=profile`, `ClusterHarness` uses `settings=default`, and `ConsoleMain` builds two recordings programmatically from `Configuration.getConfiguration("default")` (`startLocalJfr` and the cluster-coordinator recording) | measurements carry different instrumentation overhead and are not directly comparable. Note which site matters: `compare-llama-cpp.sh` passes `--jfr` as an *app* argument, so **every published Juno ratio was taken under `ConsoleMain`'s programmatic `default`** — not under either of the two configurations named in a `settings=` string |
| no CPU governor, turbo, or GPU clock state is recorded | a thermally throttled run is indistinguishable from a regression |
| no JDK event is consumed anywhere in this repository — `JfrMetricsExtractor` declares only `juno.*` names, and `compare-llama-cpp.sh`'s `jfr_summary_json` reads only `juno.ForwardPass.*`/`juno.TokenProduced.*` | GC pauses, allocation rate, hot methods and lock/park time cannot be read off any gate run, so the noise-control rule below and Tier 10's cost breakdown are unenforceable until the extractor is widened |

Before Tier 01 publishes its gate, `compare-llama-cpp.sh` gains, in this order of importance:

1. **Prompt-token parity.** `RAW_PROMPT` defaults to `1`, so Juno prefills approximately the same
   token count llama-bench is given. Every result JSON records Juno's actual `prompt_tokens`
   alongside `n_prompt`, and a run whose Juno `prompt_tokens` differs from `n_prompt` by more than
   10% is **not publishable as a ratio** — publish it as a Juno-only measurement or re-run it. This
   is first because it is the only asymmetry that changes the metric the program target is written
   against, and because the two runs this plan quotes as evidence that prefill degrades with prompt
   length sit on opposite sides of it (see Tier 01B).
2. `--juno-warmup N` (default 2) discarded requests before the measured one.
3. `--juno-reps N` (default 3) with the median reported and min/max recorded. Reuse
   `compare-lora.sh`'s existing `--reps`/median implementation rather than writing a second one, and
   note that `compare-lora.sh` itself defaults to `REPS=1` — every gate run in this plan passes
   `--reps 3` explicitly to both scripts.
4. A Juno thread-count control that actually reaches the hot path, set equal to llama-bench's `-t`
   (this requires the hot-path threading work in Tier 10 item 5 to expose one — until it does,
   record the effective Juno parallelism in the run metadata and state the mismatch in every
   published INDEX rather than leaving it implicit).
5. A fixed per-model `COMPARE_HEAP` rather than a derived one.
6. CPU governor plus `nvidia-smi -q -d CLOCK` captured into the run metadata.

The JFR configuration mismatch is resolved in the same change, and the fix has to name all four
creation sites, not the two that carry a literal `settings=` string. Add
`scripts/performance-tests/juno-perf.jfc` — derived from `default`, enabling every `juno.*` event
plus `jdk.GCPhasePause`, `jdk.ObjectAllocationSample` (throttle 300/s),
`jdk.ThreadAllocationStatistics` (period 500 ms), `jdk.ExecutionSample` (period 10 ms),
`jdk.JavaMonitorEnter` and `jdk.ThreadPark` (threshold 10 ms) — and point all four at it:
`run.sh`'s `cmd_test`, `ClusterHarness`'s forked-node flag, and **both**
`Configuration.getConfiguration("default")` calls in `ConsoleMain` (`startLocalJfr` and the
cluster-coordinator recording). The last two are the ones that matter most: `compare-llama-cpp.sh`
launches Juno with `--jfr` as an app argument, so that is the configuration every published ratio
was actually taken under.

Because this changes the harness, it also breaks strict comparability with the runs already
published under `docs/perf-compare/`. That is accepted: the first corrected run is re-baselined
against itself and labelled as the new reference, the pre-correction numbers are kept and marked as
pre-parity-correction rather than deleted, and no tier's gate is scored against a baseline taken on
the other side of the change. Tier 01 carries this work as a precondition, not as scope creep — it
is the measurement that ten tiers' exit criteria depend on.

**Numeric thresholds, not "no unexplained regression."** Per execution rule 7 above, every tier's
perf-gate bullet under "Tests to write/upgrade before implementation" states a concrete pass/fail
number for whatever new metric that tier introduces (a memory-reduction percentage, a throughput
ratio, a latency ceiling) — the same way Tiers 01/03/06/07 already do for the specific historical
regression each one is re-measuring ("no longer ~7x slower," "gather tax at zero," "faster than
`--spec-type none`," "beats static"). Pick the number before implementation starts, the same way the
tests-before-implementation rule already asks for tests before code.

**Regression-noise control: GC pauses and allocation rate are tracked by default, not discovered ad
hoc.** `docs/performance.md`'s own history includes a real false-positive regression signal caused by
a single 622ms GC pause contaminating a short JFR-window `tps` measurement (the LoRA playback gate
incident that led to switching that one gate to wall-clock timing). Every perf-gate run from Tier 01
onward records `jdk.GCPhasePause` count/max and an allocation-rate figure alongside its primary
metric, and a run containing an outlier GC pause is re-run rather than scored — this generalizes the
fix already applied once to LoRA, instead of waiting to rediscover the same failure mode per tier.

**That rule needs tooling that does not exist yet, and Tier 01 builds it.** `JfrMetricsExtractor`
declares twenty `juno.*` event names and no `jdk.*` ones; a repo-wide grep for `GCPhasePause`,
`ObjectAllocationSample`, `ExecutionSample`, `JavaMonitorEnter`, `ThreadPark` and
`ThreadAllocationStatistics` across every `.java` and `.sh` file returns nothing outside `docs/`. So
as things stand, no gate run in this plan can report a GC pause or an allocation rate, and Tier 10's
cost breakdown has nothing to read. Alongside the `.jfc` above, Tier 01 extends
`JfrMetricsExtractor` with a `jdk.*` bucket emitting `jdk.GCPhasePause.count`, `.max_ms`,
`.total_ms`, `jdk.ThreadAllocationStatistics.bytes_total` (this, not the sampled
`jdk.ObjectAllocationSample`, is what a bytes-per-token ceiling is read from — the sampled event
gives attribution, not a rate), `jdk.ObjectAllocationSample.top_sites`,
`jdk.ExecutionSample.top_methods`, `jdk.JavaMonitorEnter.total_ms` and `jdk.ThreadPark.total_ms`,
with tests in the `metrics` module. `compare-llama-cpp.sh`'s `jfr_summary_json` surfaces the GC and
allocation figures into every published result JSON. Until this lands, "records GC pauses and
allocation rate" is an instruction nobody can follow.

**Noise floor: no gate may be stated tighter than the harness can resolve.** Two CPU sweeps taken
eight minutes apart on this host with identical flags — `docs/perf-compare/20260918T031702Z/` and
`20260918T032455Z/`, differing only in `juno_use_vector` — moved llama.cpp's own TinyLlama tg from
25.98 to 22.45 t/s (**-14%**) and its pp from 61.17 to 71.07 t/s (**+16%**), with llama-bench's `-r 3`
already applied. That is the measurement floor. No tier may state a pass/fail threshold inside
±15% unless it is a median of at least three runs with min/max published. Where a tier's threshold is
necessarily tighter than that (Tier 01B's "tg within 0.95x", Tier 04's "within 15% of the Q4_K MMQ
kernel"), the median-of-three discipline is mandatory, not optional, and the min/max spread is
published next to the median so a reader can see whether the result cleared the floor.

**No CI exists in this repository today** (confirmed: `.github/` holds only a `modernize/`
directory, no workflows) — tier-gating (execution rule 1: don't start Tier N+1 until Tier N's exit
criteria are all checked) is enforced procedurally, by whoever executes the plan re-reading the
checklist, not automatically. This is an accepted, explicit trade-off rather than a silent gap; if a
CI pipeline is added during this plan's execution, wiring `mvn test` plus the relevant
`compare-*.sh` gate into it per tier is in scope for whichever tier is active at that point.

**Owner and revisit trigger:** whoever executes the plan owns this decision, and it is re-examined
at Tier 07 or at the first point a tier's full smoke matrix exceeds thirty minutes of hands-on
execution, whichever comes first. Record the outcome of that re-examination in the then-active
tier's file, so "we decided not to" stays a decision with a date on it rather than an omission.

## Model and hardware inventory

See [`INVENTORY.md`](INVENTORY.md). Summary: one NVIDIA GTX 1080 (CUDA) is the only GPU available
in this environment; there is no ROCm/AMD hardware. Tiers that touch ROCm code paths are
implemented and unit-tested to the extent possible without hardware, and their exit criteria
include an explicit `FAIL-CLOSED` or `NEEDS-AMD-HARDWARE` marker rather than a false `PASS` for row
3 of the compatibility checklist — flag this to the user when a tier reaches that point rather than
guessing at real-hardware behavior.

## Feature-complete definition

A tier is feature-complete only when **all** of the following hold, not just "the happy path
works":

- Every row of the cross-surface compatibility checklist for that tier's feature is PASS, N/A (with
  reason), or FAIL-CLOSED (tested).
- All new/extended tests from layer 1-3 above pass, and all pre-existing tests still pass
  (`mvn test` across all unit-test-bearing modules, `mvn verify -pl juno-master`).
- Any hot-path change has a published `docs/perf-compare/` entry per the existing performance-gate
  rule, against a concrete numeric threshold stated in that tier's own file (execution rule 7 — not
  just "no unexplained regression"), and, for Tiers 01, 01B, 02, 03, 04, 04B, 04C, 06, 07, 08, 09, 10, an
  accompanying `compare-llama-cpp.sh` run recording the current Juno/llama.cpp ratio and reading it
  against the program target and that tier's intermediate milestone, if it has one.
- **The published API contract is updated in the same change as the code.**
  `api/src/main/resources/openapi.yaml`, `api/src/main/resources/juno-api.yaml` and
  `api/src/main/proto/inference.proto` are the contract, and no tier may add an endpoint, a request
  or response field, or an RPC without updating them. This is load-bearing for Tiers 02 (context-shift
  opt-in), 03 (session save/restore), 05 (`x_juno_samplers`, widened `json_schema`), 12
  (`x_juno_loras`) and 13 (`/v1/rerank`), and for Tiers 06 and 09 where the gRPC semantics change
  rather than the shape. Before this was written, the `api` module appeared in this plan exactly
  once — Tier 00 removing `RegistryService` — while six tiers added surface on top of it.
- `CHANGELOG.md` gets an entry describing what shipped, in the project's existing style.
- `docs/agent-arch.txt`, `docs/howto.md`, and `README.md` are updated if the change is user-facing
  or architectural (existing `CLAUDE.md` rule), using Juno-native language only (rule 4 above).
- No new dead/dormant/unwired scaffolding is left behind without an explicit tracking note in that
  tier's file explaining why it's dormant and what would need to happen to wire it in (the pattern
  flagged in gap-analysis §2.8 — don't repeat it silently).
