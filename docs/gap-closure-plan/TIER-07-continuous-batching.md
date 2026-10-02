# Tier 07: Continuous batching maturity

Status: not started
Gap analysis refs: §1.3 (scheduling half), corroborated by `docs/performance.md`'s own 0.86x
multi-session-throughput finding

## Objective

Move `continuous` scheduling from "structurally real but not yet a throughput win" to actually
beating `static` on the benchmarks that matter, by switching from slot-based to token-budget-based
step sizing, raising the practical concurrency ceiling past the current fixed `DEFAULT_RUNNING_SET
= 8`, and making a deliberate, documented decision about cluster support (either build it, or
formally close the "v1 scope" note with a concrete "why not" rather than an open-ended TODO).

## Why this tier, why now

This depends on Tier 03's KV cache work (true block-table attention, no gather tax) — token-budget
scheduling only pays off if admitting more concurrent requests doesn't multiply a per-request
gather cost that Tier 03 is responsible for eliminating. It also depends on Tier 01/02's residency
and attention work for the same reason underlying every batching improvement: more concurrent
sequences means more attention calls, and those need to be cheap per-call for a bigger batch to
actually help rather than hurt.

## Scope

### In scope

1. Token-budget scheduling: `ContinuousMixedStepPolicy` moves from treating "one decode request"
   and "one prefill chunk participant" as equal-cost slots to sizing each step by an actual token
   budget (configurable, analogous in purpose to `--prefill-batch` but governing the whole step,
   not just prefill chunks).
2. Raise (or make properly configurable and validated at higher values) the concurrency ceiling
   past 8, bounded by real measured memory/throughput limits rather than an arbitrary constant.
3. Re-run the exact multi-session throughput benchmark that produced the 0.86x finding
   (`docs/performance.md:177`) after every other change in this tier lands, as the tier's go/no-go
   checkpoint — continuous must beat static before this tier is called complete, or the tier
   documents precisely why not and what would be needed.
4. Cluster support decision: either implement `continuous` scheduling for cluster/tensor-parallel/
   pipeline-parallel topologies (a real feature, likely substantial given today's KV/paging code has
   no shard-routing logic at all per the gap analysis), or close the "v1 scope" note with a
   concrete technical explanation of what would be required and file it as an explicitly-scoped
   future tier rather than leaving it an open TODO indefinitely.
5. **The CI re-examination this plan defers to this tier.** [`README.md`](README.md) records that no CI
   exists (`.github/` holds only a `modernize/` directory, no workflows), that tier-gating is therefore
   enforced procedurally by whoever executes the plan re-reading the checklist, and that the decision is
   "re-examined at Tier 07 or at the first point a tier's full smoke matrix exceeds thirty minutes of
   hands-on execution, whichever comes first." That naming lived only in the README, so nothing in this
   tier's own scope or exit criteria would have told a reader the revisit was due — which is how an
   accepted trade-off quietly becomes an omission. It is now this tier's item.

   Re-examine and record the outcome here with a date. Two things have changed since that judgement was
   made and both bear on it: six tiers have landed, each adding a `smoke-tierNN-*.sh` script that every
   later tier is required to re-run unmodified as a regression check, so the procedural matrix is
   monotonically growing; and this tier is the one whose own gate is a concurrency benchmark that has to
   be re-run after every change in it. Time the full smoke matrix as part of this tier's exit work and
   record the figure, since the thirty-minute condition cannot be evaluated without it.

   Either outcome is acceptable and both must be written down. If CI is adopted: wire `mvn test` across
   every unit-test-bearing module (including `vision` and `metrics`, which the documented command
   omitted until Tier 00 fixed it), `mvn verify -pl juno-master`, `check-plan-thresholds.sh`, and the
   accumulated `smoke-tierNN-*.sh` scripts that can run without a GPU or a real model. The GPU-gated and
   real-model gates stay manual — this host's GTX 1080 is not a CI runner — so state plainly which
   fraction of the matrix CI actually covers rather than implying it covers the gate. If CI is declined:
   name what changed since the README's original judgement, or state that nothing did, and record the
   measured smoke-matrix duration that supports the call.

6. **Load-dependent prefill chunk sizing under `continuous` (added 2026-10-02, owner decision; handed
   over by [Tier 01B](TIER-01B-prefill-throughput.md) item 3).** `continuous` keeps the fixed
   `--prefill-batch` default of 32 because that is the decode-interleaving fairness unit: under mixed
   load (one 512-token prompt plus three short streaming requests, TinyLlama, GPU) 32 gives the short
   requests the lowest time to first token (898 ms, against 1,004 ms at 128 and 1,336 ms at 512). The
   cost is paid when nothing needs protecting: a 512-token prompt arriving alone takes 2,235 ms under
   `continuous` at 32, against 600 ms at 512 and 549 ms under `static` with whole-prompt sizing
   ([`20261002T194000Z-tier01b-item3-chunk-review`](../perf-compare/20261002T194000Z-tier01b-item3-chunk-review/INDEX.md)).
   A fixed number cannot be right for both cases, so the chunk becomes a function of the step's load,
   built on item 1's token budget: a prefill whose step has no decoding member takes the whole
   remaining prompt (bounded by the free-VRAM sizing `static` already uses), and one that shares the
   step with decoders takes what the step's budget leaves. The owner chose this over moving the fixed
   default to 128 (short-request TTFT +12% on TinyLlama, more on larger models) and over free-VRAM sizing
   for `continuous` (interleaving disappears). An explicit `--prefill-batch N` keeps meaning a fixed `N`
   on `continuous`. `PrefillChunkDefaults` (coordinator) is where the default lives today; its
   `continuous` reason and `docs/howto.md`'s `--prefill-batch` row change in the same change.

### Out of scope

- Any change to the `static` schedule itself, beyond what's needed to keep the Tier-00 prefix-cache
  fix correct as continuous's behavior changes around it.
- Elastic cluster membership (Tier 00 removed the unimplemented `RegistryService`; any replacement is Tier 13's scope).

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | token-budget scheduling and higher concurrency must work without a GPU (slower, but correct) |
| 2 | CUDA GPU inference | primary target for the throughput benchmark |
| 3 | ROCm GPU inference | same scheduling logic, NEEDS-AMD-HARDWARE only for the kernel-level dependencies (Tier 01-03's GPU-specific pieces), scheduling itself is backend-agnostic |
| 4 | Static schedule | must remain fully unaffected — this tier only changes `continuous` |
| 5 | Continuous schedule | primary target of this entire tier |
| 6 | Single-node local mode | primary dev/test surface |
| 7 | Pipeline-parallel cluster | either gets real continuous support this tier, or gets a documented, tested, still-correct fallback-to-static with an updated (not just "v1 scope") explanation |
| 8 | Tensor-parallel cluster | same |
| 9 | LoRA training | N/A |
| 10 | LoRA playback | confirm the still-unwired `x_juno_loras` per-request override (§1.11) continues to fail closed correctly under the new token-budget scheduler — don't accidentally let a scheduling refactor silently start accepting it half-wired |
| 11 | Vision | vision is local-mode only today; confirm token-budget scheduling doesn't assume text-only batch composition in a way that breaks vision requests sharing a batch with text requests, if that's a supported combination — verify explicitly |
| 12 | OpenAI REST surface | streaming (SSE) TTFT/TPOT metrics must be re-measured post-change, not just aggregate throughput |
| 13 | Native REST surface | same |
| 14 | CLI | `--parallel`'s meaning under `continuous` changes (from a slot cap to informing/interacting with the token budget) — document this clearly since it's a user-visible behavior change |

## Implementation steps

1. Write the throughput benchmark reproduction first (confirm the 0.86x baseline still holds on
   current HEAD, accounting for whatever Tiers 01-03 already changed) as the before-measurement.
2. Implement token-budget sizing in `ContinuousMixedStepPolicy`.
2a. On top of it, item 6: the load-dependent prefill chunk, measured on its own against fixed 32
    before step 3 changes the concurrency ceiling.
3. Raise/validate the concurrency ceiling.
4. Re-run the benchmark; iterate until continuous beats static or the ceiling of what's achievable
   without further architectural change is well-understood and documented.
5. Make and execute the cluster-support decision.

## Tests to write/upgrade before implementation

- **Reproduce the existing 0.86x benchmark** as an automated, re-runnable check (if it isn't
  already one) rather than a one-off manual measurement — this becomes the tier's primary
  regression/success gate.
- **Item 6 (load-dependent chunk)**: `ContinuousMixedStepPolicyTest` cases that a prefill with no
  decoding member in its step receives its whole remaining prompt (capped by the sizing bound), that one
  sharing a step with decoders receives no more than the budget leaves, and that an explicit
  `--prefill-batch N` still yields fixed `N`-token chunks; and a `PrefillChunkDefaults` test for the new
  `continuous` default.
- **`ContinuousMixedStepPolicyTest`**: token-budget sizing correctness — a step's actual token cost
  stays within budget across varied decode/prefill-chunk mixes.
- **New concurrency-ceiling test**: correctness (not just throughput) at the new higher ceiling —
  confirm no request starvation, no KV corruption, under real concurrent load.
- **`ModelLiveRunnerIT`**: add a continuous-schedule multi-session throughput check.
- **New bash smoke script**: `scripts/performance-tests/smoke-tier07-continuous-batching.sh` —
  drives N concurrent SSE streams under `continuous`, asserts correctness and captures TTFT/TPOT.
- **Perf gate (required)**: this is entirely a batching/scheduling hot-path change —
  `compare-schedule.sh` (already exists per `scripts/performance-tests/`) rerun, plus
  `compare-lora.sh`, plus `compare-llama-cpp.sh` for a llama.cpp-relative reading (per README's
  llama.cpp-relative gate); publish under `docs/perf-compare/`.

  **Threshold.**
  - Aggregate multi-session throughput under `continuous` must reach **>= 1.10x** `static` on the
    8x64 GPU benchmark that produced the 0.86x finding (`docs/performance.md`). Not 1.0x: parity is
    inside this host's noise floor and would not establish that continuous is actually better.
  - Streaming latency must not be traded away for aggregate throughput — p95 TTFT under `continuous`
    at the benchmark concurrency **<= 1.25x** `static`'s, and p95 TPOT **<= 1.10x**. A scheduler
    that wins on aggregate tokens by starving individual streams has not improved the product.
  - Correctness at the raised concurrency ceiling: no request starvation (every admitted request
    completes) and no KV cross-contamination, asserted by test rather than by throughput alone.
  - Item 6, a prompt alone: a 512-token prefill under `continuous` with no other request in flight takes
    **<= 1.25x** the time it takes under `static` on the same build (2,235 ms against 549 ms today on
    TinyLlama, 4.07x), median of three, TinyLlama and Mistral 7B.
  - Item 6, under mixed load: on `compare-mixed-prefill.sh --n-prompt 512` (three short streams), the
    short requests' mean TTFT **<= 1.10x** the fixed-32 reading and the long prompt's TTFT **<= 0.75x**
    it, from a same-hour interleaved A/B with pinned clocks (fixed `--prefill-batch 32` against the new
    default; README, "No-regression gates tighter than the floor are Juno-against-Juno"), TinyLlama and
    Mistral 7B.
  - `static` must be untouched: Juno tg and pp t/s **>= 0.98x** the pre-tier build, from a same-hour interleaved A/B with pinned clocks against the pre-tier build (README, "No-regression gates tighter than the floor are Juno-against-Juno"), since
    this tier is scoped to change only `continuous`. If the A/B's own spread on this host is wider than
    2%, record the spread and score against it rather than claiming a resolution the harness lacks.

## Models needed

Existing models are sufficient (this is a scheduling change, not architecture-dependent). Reuse the
same models as the original `20260911T194430Z-continuous` bake-off for a clean before/after
comparison.

## Exit criteria

- [ ] Continuous scheduling beats static on the multi-session throughput benchmark (or the tier
      documents precisely why not, with a concrete follow-up path, rather than shipping silently
      unchanged).
- [ ] Token-budget scheduling implemented and correctness-tested.
- [ ] Item 6: the `continuous` prefill chunk follows the step's load, an explicit `--prefill-batch N`
      still fixes it, and both item 6 thresholds are met (a prompt alone within 1.25x of `static`;
      under mixed load short-request TTFT within 1.10x of fixed 32 and the long prompt's at most 0.75x
      of it), with the run directory cited.
- [ ] Concurrency ceiling raised and validated, or kept at 8 with a documented, measured reason.
- [ ] Cluster-support decision made and executed (real support, or a closed, technically-grounded
      "not now" note replacing the open-ended "v1 scope" comment).
- [ ] **The CI re-examination is recorded in this file with a date and an outcome** — either a
      `.github/workflows/` pipeline covering `mvn test` across every unit-test-bearing module,
      `mvn verify -pl juno-master`, `check-plan-thresholds.sh` and the GPU-free smoke scripts, with the
      fraction of the matrix it actually covers stated; or a written decision not to, naming what
      changed since the README's original judgement or stating that nothing did. The measured duration
      of this tier's full smoke matrix is recorded either way, since the README's thirty-minute revisit
      condition cannot be evaluated without it. An unrecorded revisit is a missed exit criterion, not a
      deferral.
- [ ] Cross-surface checklist fully resolved.
- [ ] Perf gate published showing the throughput result against every threshold above, or the
      miss reported plainly with its number.
- [ ] Docs (`docs/howto.md`, `docs/performance.md`) updated with the new `--parallel` semantics
      under `continuous`.
- [ ] `CHANGELOG.md` entry added.
