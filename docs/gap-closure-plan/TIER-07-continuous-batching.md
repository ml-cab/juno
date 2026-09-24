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
3. Raise/validate the concurrency ceiling.
4. Re-run the benchmark; iterate until continuous beats static or the ceiling of what's achievable
   without further architectural change is well-understood and documented.
5. Make and execute the cluster-support decision.

## Tests to write/upgrade before implementation

- **Reproduce the existing 0.86x benchmark** as an automated, re-runnable check (if it isn't
  already one) rather than a one-off manual measurement — this becomes the tier's primary
  regression/success gate.
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
  - `static` must be untouched: tg and pp within **0.98x** of the pre-tier baseline, since this tier
    is scoped to change only `continuous`.

## Models needed

Existing models are sufficient (this is a scheduling change, not architecture-dependent). Reuse the
same models as the original `20260911T194430Z-continuous` bake-off for a clean before/after
comparison.

## Exit criteria

- [ ] Continuous scheduling beats static on the multi-session throughput benchmark (or the tier
      documents precisely why not, with a concrete follow-up path, rather than shipping silently
      unchanged).
- [ ] Token-budget scheduling implemented and correctness-tested.
- [ ] Concurrency ceiling raised and validated, or kept at 8 with a documented, measured reason.
- [ ] Cluster-support decision made and executed (real support, or a closed, technically-grounded
      "not now" note replacing the open-ended "v1 scope" comment).
- [ ] Cross-surface checklist fully resolved.
- [ ] Perf gate published showing the throughput result against every threshold above, or the
      miss reported plainly with its number.
- [ ] Docs (`docs/howto.md`, `docs/performance.md`) updated with the new `--parallel` semantics
      under `continuous`.
- [ ] `CHANGELOG.md` entry added.
