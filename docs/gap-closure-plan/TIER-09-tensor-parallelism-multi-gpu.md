# Tier 09: Tensor parallelism & multi-GPU

Status: not started
Gap analysis refs: §1.8, §2.2

## Objective

Build the real per-layer column-/row-parallel weight-sliced computation that `TensorShardContext`'s
javadoc has described since before this plan existed but that no handler actually implements today
(§2.2) — replace the current "broadcast full computation to every node, AllReduce a mathematically
inert sum" stub path with genuine tensor-sliced compute. Separately, evaluate and, if justified,
implement a single-process multi-GPU path (so one JVM can use more than one local GPU without a
gRPC round trip per exchange), addressing §1.8.

## Why this tier, why now

This is sequenced after model architecture breadth (Tier 08) so that real tensor-parallel slicing
is built against the final set of transformer handlers, not against handlers that are about to
change. It's a large, correctness-sensitive tier — slicing Q/K/V and FFN weights incorrectly
produces output that looks plausible but is subtly wrong, exactly the kind of bug the project's
fail-closed philosophy is designed to catch, so this tier leans hard on numerical-parity testing
against the existing (correct, if slow) single-node dense computation.

## Scope

### In scope

1. Real column-parallel Q/K/V projection and row-parallel output projection, per
   `TensorShardContext`'s existing (currently aspirational) design, wired into
   `LlamaTransformerHandler` first (the most-used handler), with per-layer AllReduce replacing
   today's whole-model-output AllReduce.
2. Replace `TensorParallelClusterIT`'s dummy-fixed-logit `CyclicForwardPassHandler` stub with a
   real multi-node test against an actual sliced model, numerically validated against the
   equivalent single-node dense run.
3. Extend real tensor-parallel slicing to the other transformer handlers (Phi-2, Phi-3, Qwen3,
   Qwen3-MoE, and whatever Tier 08 added) — per rule 2, this tier isn't complete until every
   currently-supported architecture has real tensor-parallel support, not just Llama-family.
4. Evaluate single-process multi-GPU: determine whether `GpuContext`'s existing per-device
   addressing (`deviceIndex`, `GpuBindings.deviceMalloc(deviceIndex, ...)`) is sufficient scaffolding
   to build a same-process multi-GPU tensor-parallel scheduler on top of, avoiding a gRPC round
   trip per cross-GPU exchange when all GPUs are local to one host. If justified (this environment
   only has one GPU, so this is a design-and-unit-test effort here, validated for real once
   multi-GPU hardware is available — flag that constraint explicitly), implement it.

### Out of scope

- NCCL/RCCL integration — the gap analysis notes this is a deliberate design choice (Java
  distributed tooling over NCCL/MPI), not treated as a gap to reverse in this plan unless the user
  says otherwise; the single-process multi-GPU path in scope above can use direct
  peer-to-peer/device-to-device CUDA calls without adopting NCCL specifically.
- Elastic N-node clustering beyond the existing fixed-3-node `ClusterHarness` topology (that's
  Tier 13's scope).

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | not directly affected — tensor parallelism is a multi-device concept — but the dense CPU path remains the numerical-parity oracle every sliced computation is checked against |
| 2 | CUDA GPU inference | primary target |
| 3 | ROCm GPU inference | tensor-parallel slicing logic (shard math, AllReduce) is backend-agnostic; GPU-kernel-level pieces NEEDS-AMD-HARDWARE |
| 4 | Static schedule | tensor-parallel + static batching interaction must be retested now that per-layer AllReduce replaces the old whole-output reduction |
| 5 | Continuous schedule | tensor-parallel + continuous is currently auto-fallback to static (Tier 07's territory) — confirm this tier's changes don't accidentally make that combination reachable in a half-working state; keep the fallback until Tier 07 (or a later revisit) explicitly re-evaluates it |
| 6 | Single-node local mode | single-process multi-GPU (if built) is exercised here, even with only one physical GPU available — design for N≥2 and validate what's testable with N=1 (i.e., the code path compiles/runs correctly in a single-GPU degenerate case) |
| 7 | Pipeline-parallel cluster | must remain unaffected — this tier only changes the tensor-parallel path |
| 8 | Tensor-parallel cluster | primary target of this entire tier |
| 9 | LoRA training | confirm LoRA training still works (or is still explicitly unsupported) for tensor-parallel-sharded models |
| 10 | LoRA playback | same |
| 11 | Vision | vision is local-mode only; N/A for cluster/tensor-parallel this tier |
| 12 | OpenAI REST surface | end-to-end correctness via chat completions against a real tensor-parallel-sharded model, numerically compared to the same model run single-node dense |
| 13 | Native REST surface | same |
| 14 | CLI | `./juno cluster --pType tensor` behavior changes from "works but does nothing useful" to "actually shards" — this is a significant behavior change, document it clearly |

## Implementation steps

1. Write the numerical-parity test harness first: run a small model single-node dense, then
   2-way-and-3-way tensor-parallel-sharded, assert outputs match within float tolerance — this test
   must fail against today's stub implementation (confirming it currently doesn't validate anything
   meaningful) before any new code is written, as proof the test is actually exercising real
   slicing once implemented.
2. Implement real column-/row-parallel slicing for `LlamaTransformerHandler`; get the parity test
   passing for 2-node and 3-node splits.
3. Replace `TensorParallelClusterIT`'s stub handler with the real sliced computation.
4. Extend to the remaining handlers, one at a time, each gated by its own parity test.
5. Evaluate and, if justified, implement single-process multi-GPU addressing.
6. Full cross-surface smoke matrix, with particular focus on the numerical-parity gate — this
   tier's exit bar is correctness, not just "doesn't crash."

## Tests to write/upgrade before implementation

- **New `TensorShardContextParityTest`** (or similarly named): single-node dense vs. 2-way/3-way
  tensor-parallel-sharded output comparison, for each supported architecture, within float
  tolerance.
- **`TensorParallelClusterIT`**: rewritten to use a real model and real slicing instead of the
  dummy-fixed-logit stub — this is the single most important test change in this tier, since it's
  the one the gap analysis specifically called out as currently not testing anything meaningful.
- **New single-process multi-GPU unit tests** (if that work proceeds): correct device-to-device
  data movement, correct results in the N=1 degenerate case, and a design-level test double
  simulating N≥2 if real multi-GPU hardware isn't available to validate against directly (flag this
  limitation explicitly rather than claiming full validation).
- **`ModelLiveRunnerIT`**: the existing 2 tensor-parallel checks get upgraded from stub-based to
  real-slicing-based; add checks for the other newly-sliced architectures.
- **New bash smoke script**: `scripts/performance-tests/smoke-tier09-tensor-parallel.sh` — runs
  the 3-node tensor-parallel cluster against a real model and diffs output against a single-node
  dense run of the same model/prompt/seed.
- **Perf gate (required)**: tensor parallelism is explicitly a forward-pass/GPU-residency change —
  full `compare-lora.sh`, plus a dedicated tensor-parallel-vs-single-node throughput/latency
  comparison; publish under `docs/perf-compare/`.

## Models needed

Existing dense models suffice for parity testing across 2-3 simulated shards on one physical
machine (the existing `ClusterHarness` forks JVM processes, not physical hosts, so no additional
hardware is strictly required to validate correctness — only the single-process multi-GPU
sub-scope needs a second physical GPU, which isn't available here; flag that limitation when this
tier reaches that point).

## Exit criteria

- [ ] Real column-/row-parallel slicing implemented for every currently-supported transformer
      handler, numerically validated against single-node dense output.
- [ ] `TensorParallelClusterIT` exercises real slicing, not a dummy-fixed-logit stub.
- [ ] Single-process multi-GPU evaluated; implemented if justified, with N=1 degenerate-case
      validation and an explicit note about what couldn't be validated without a second physical
      GPU.
- [ ] Cross-surface checklist fully resolved; continuous+tensor-parallel fallback-to-static
      behavior explicitly confirmed unchanged (not silently made reachable in a half-working state).
- [ ] Perf gate published.
- [ ] Docs (`docs/agent-arch.txt`, `docs/howto.md`) updated to state plainly that tensor parallelism
      now does real sliced computation, replacing whatever Tier 00 documented as its prior
      (non-functional) state.
- [ ] `CHANGELOG.md` entry added.
