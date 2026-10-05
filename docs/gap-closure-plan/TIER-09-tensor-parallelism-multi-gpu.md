# Tier 09: Tensor parallelism & multi-GPU

Status: not started
Gap analysis refs: §1.8, §2.2

## Objective

Build the real per-layer column-/row-parallel weight-sliced computation that `TensorShardContext`'s
javadoc has described since before this plan existed but that no handler actually implements today
(§2.2) — replace the current "broadcast full computation to every node, AllReduce a mathematically
inert sum" stub path with genuine tensor-sliced compute. Separately, evaluate and, if justified,
implement a single-process multi-GPU path (so one JVM can use more than one local GPU without a
gRPC round trip per exchange), addressing §1.8. And make prompt processing on every cluster path batched rather
than one gRPC round trip per prompt token (scope item 5, added 2026-10-02).

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
   *Known limitation found before this tier (2026-09-27).* mistral-7b cannot start in tensor mode
   on this host (one 8 GiB GTX 1080, three forked nodes) because every node loads the **whole**
   model (`TensorShardContext` is geometry only). Observed by `ModelLiveRunnerIT` on 2026-09-27 with
   and without `--gpu-residency`. The first reading was that three copies do not fit the card; the
   node's own error, once it was made to reach the coordinator (a Tier 01 out-of-tier change),
   shows the first wall is the **node heap**: `java.lang.OutOfMemoryError: Java heap space` in
   `GgufReader.tensorRaw` during `LlamaTransformerHandler` construction, at the forked node's default
   `-Xmx4g` (`juno.node.heap`). A larger heap would only move the failure to device or host memory for
   three full copies. Real weight slicing (item 1) is what fixes it; its parity test should include a
   model that only fits when sliced, and the heap each node is given should follow its slice (as the
   launchers already derive heap from the model file size) rather than a fixed 4 GB.
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

5. **Batched prefill on every cluster path (added 2026-10-02, owner decision; found by
   [Tier 01B](TIER-01B-prefill-throughput.md) item 3).** Neither gRPC pipeline client
   (`ProcessPipelineClient` for `--pType pipeline`, `TensorParallelPipelineClient` for `--pType tensor`)
   overrides `InferencePipeline.prefillBatch`, so a cluster prefill runs the interface default: one
   `ForwardPass` call per prompt token, whatever `--prefill-batch` says. Nothing single-node prefill has
   gained (batched GEMMs, the prefill-window device region, whole-window attention) reaches a cluster
   node. Measured on TinyLlama, three forked nodes on this host's one GPU, a 512-token prompt
   ([`20261002T194000Z-tier01b-item3-chunk-review`](../perf-compare/20261002T194000Z-tier01b-item3-chunk-review/INDEX.md)):
   local 0.55 s; pipeline 20.3 s at chunk 32 and 21.3 s at 512; tensor 28.8 s and 29.7 s. The scope is
   **both cluster modes and the standalone coordinator** (`juno-master`'s `CoordinatorMain` builds the
   same two clients), not tensor mode alone:
   - a window forward on the node contract: `api/src/main/proto/inference.proto` gains the window shape
     (token ids or a `W x hidden` activation, start position, window width) in the same change as the code,
     per the README's feature-complete rule. `ForwardRequest.batch_size` exists today but no node treats
     it as a prefill window; reuse it only if the node's semantics for it are made explicit and tested;
   - the node runs its handler's batched window path (`forwardBatch`) for a window request, so a GPU node
     takes the prefill-window device region exactly as a local run does;
   - `ProcessPipelineClient` passes a window's activations from shard to shard, one call per shard per
     window; `TensorParallelPipelineClient` exchanges and reduces once per layer per window rather than
     per token (with item 1's per-layer AllReduce once it lands);
   - both clients override `prefillBatch`; a node that does not support window requests fails the
     request with an error naming the node, rather than the coordinator silently falling back to
     per-token calls;
   - `--prefill-batch` then matters on cluster surfaces, so `PrefillChunkDefaults` (coordinator) re-reads
     the cluster default (it is 32 today because the value is inert there) and `docs/howto.md`'s
     `--prefill-batch` row changes with it.

### Out of scope

- NCCL/RCCL integration — the gap analysis notes this is a deliberate design choice (Java
  distributed tooling over NCCL/MPI), not treated as a gap to reverse in this plan unless the user
  says otherwise; the single-process multi-GPU path in scope above can use direct
  peer-to-peer/device-to-device CUDA calls without adopting NCCL specifically.
- Elastic N-node clustering — discovery, membership, the production launch path and fault tolerance
  are all Tier 13's scope. The one exception is implementation step 0 option (a): generalizing
  `ClusterHarness`'s node count so a 2-way parity split can be tested at all. That is a test-harness
  change, not elastic clustering, and it is the only part of Tier 13's territory this tier may take.

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | not directly affected — tensor parallelism is a multi-device concept — but the dense CPU path remains the numerical-parity oracle every sliced computation is checked against |
| 2 | CUDA GPU inference | primary target |
| 3 | ROCm GPU inference | tensor-parallel slicing logic (shard math, AllReduce) is backend-agnostic; GPU-kernel-level pieces NEEDS-AMD-HARDWARE |
| 4 | Static schedule | tensor-parallel + static batching interaction must be retested now that per-layer AllReduce replaces the old whole-output reduction |
| 5 | Continuous schedule | tensor-parallel + continuous is currently auto-fallback to static (Tier 07's territory) — confirm this tier's changes don't accidentally make that combination reachable in a half-working state; keep the fallback until Tier 07 (or a later revisit) explicitly re-evaluates it |
| 6 | Single-node local mode | single-process multi-GPU (if built) is exercised here, even with only one physical GPU available — design for N≥2 and validate what's testable with N=1 (i.e., the code path compiles/runs correctly in a single-GPU degenerate case) |
| 7 | Pipeline-parallel cluster | unaffected by the slicing work (items 1 to 4); **a primary target of item 5** (batched prefill), with greedy output identical to its per-token prefill and prefill time held to the item 5 threshold |
| 8 | Tensor-parallel cluster | primary target of this entire tier, item 5 included |
| 9 | LoRA training | confirm LoRA training still works (or is still explicitly unsupported) for tensor-parallel-sharded models |
| 10 | LoRA playback | same |
| 11 | Vision | vision is local-mode only; N/A for cluster/tensor-parallel this tier |
| 12 | OpenAI REST surface | end-to-end correctness via chat completions against a real tensor-parallel-sharded model, numerically compared to the same model run single-node dense |
| 13 | Native REST surface | same |
| 14 | CLI | `./juno cluster --pType tensor` behavior changes from "works but does nothing useful" to "actually shards" — this is a significant behavior change, document it clearly |

## Implementation steps

0. **Resolve the node-count dependency before writing the parity harness.** This tier's parity test
   wants 2-way *and* 3-way splits, but `ClusterHarness` is hard-coded to three nodes
   (`threeNodes()`/`tensorNodes()`, fixed ports, and an explicit `totalLayers must be >= 3 to split
   across 3 nodes` guard), and generalizing it to a configurable node count is Tier 13's scope —
   a later tier. Pick one and record the choice here:
   - **(a)** hoist just the node-count generalization of `ClusterHarness` into this tier, leaving
     the production launch path, discovery and fault tolerance to Tier 13 (preferred: it is a
     contained change and it makes the 2-way parity case possible); or
   - **(b)** restrict this tier's parity testing to 3-way splits and state plainly that 2-way is
     untested until Tier 13, rather than listing a 2-way case that cannot run.

   Do not leave this implicit — a parity test that silently only ever ran 3-way while the plan
   claims 2-and-3-way coverage is the same class of defect this tier exists to fix in
   `TensorParallelClusterIT`.

1. Write the numerical-parity test harness: run a small model single-node dense, then
   tensor-parallel-sharded at each split width step 0 made available, assert outputs match within
   float tolerance — this test must fail against today's stub implementation (confirming it
   currently doesn't validate anything meaningful) before any new code is written, as proof the test
   is actually exercising real slicing once implemented.
2. Implement real column-/row-parallel slicing for `LlamaTransformerHandler`; get the parity test
   passing at every split width step 0 made available.
3. Replace `TensorParallelClusterIT`'s stub handler with the real sliced computation.
3a. Item 5, batched cluster prefill, tests first. The contract change and the pipeline-mode and
    coordinator half do not depend on slicing and are measured on their own, as a separate change; the
    tensor-mode half lands after step 3 so it batches the real per-layer exchange rather than the stub.
4. Extend to the remaining handlers, one at a time, each gated by its own parity test.
5. Evaluate and, if justified, implement single-process multi-GPU addressing.
6. Full cross-surface smoke matrix, with particular focus on the numerical-parity gate — this
   tier's exit bar is correctness, not just "doesn't crash."

## Tests to write/upgrade before implementation

- **Plan check, first**: `scripts/performance-tests/check-plan-thresholds.sh` passes before any other
  test or code in this tier (README execution rule 7).
- **New `TensorShardContextParityTest`** (or similarly named): single-node dense vs.
  tensor-parallel-sharded output comparison at every split width step 0 made available (3-way at
  minimum, 2-way as well if step 0 chose option (a)), for each supported architecture, within float
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
- **Item 5 (batched cluster prefill)**: a node-level test that a window request equals the same tokens
  sent one per request (logits and KV within float tolerance, at window widths 1, 8, 9, 32 and 512); a
  client test counting `ForwardPass` calls per prefill (one per shard per window in pipeline mode, not
  one per token); a node that rejects window requests surfaces an error naming it; and a cluster IT
  (both `--pType` values, and `CoordinatorMain`) whose greedy output over 64 tokens equals the per-token
  prefill's on six prompts. `ThreeNodeClusterIT` and `TensorParallelClusterIT` keep passing.
- **New bash smoke script**: `scripts/performance-tests/smoke-tensor-parallel.sh` — runs
  the 3-node tensor-parallel cluster against a real model and diffs output against a single-node
  dense run of the same model/prompt/seed.
- **Standing CPU and allocation gate** (README, "Test infrastructure"): run against the pre-tier jar
  and score it before closing this tier.
- **Perf gate (required)**: tensor parallelism is explicitly a forward-pass/GPU-residency change —
  full `compare-lora.sh`, plus a dedicated tensor-parallel-vs-single-node throughput/latency
  comparison, plus `compare-llama-cpp.sh` for a llama.cpp-relative reading on the single-node
  configuration (per README's llama.cpp-relative gate — multi-node TP has no direct llama.cpp
  equivalent on this hardware, so the single-node ratio is the relevant anchor); publish under
  `docs/perf-compare/`.

  **Threshold — and read the hardware constraint before setting one.** An earlier draft required
  "2-node TP throughput must exceed single-node dense throughput by ≥1.3x." That is unreachable
  here and would have to be waived or faked. `ClusterHarness` forks node JVMs on `localhost`, and
  per [`INVENTORY.md`](INVENTORY.md) this environment has exactly one GTX 1080, so every shard
  contends for the same device: tensor parallelism on one GPU does the same total work as a single
  node and adds gRPC hops and AllReduce on top. It cannot be faster, however correct the slicing.

  So throughput is **recorded, not gated**, and correctness is the exit bar:
  - **Gate:** sliced output matches single-node dense within float tolerance on every supported
    architecture — this is the threshold, and it is a hard one.
  - **Gate:** per-layer AllReduce and gRPC overhead is published as a measured per-token cost, so a
    future multi-GPU host has a number to be held to.
  - **Gate:** single-node throughput must not regress — Juno tg and pp t/s **>= 0.95x** the pre-tier
    build, from a same-hour interleaved A/B with pinned clocks against the pre-tier build (README, "No-regression gates tighter than the floor are Juno-against-Juno"), since this tier rewrites the projection path that single-node inference
    also uses.
  - **Gate, item 5:** a 512-token prefill on TinyLlama through `./juno cluster` takes **<= 0.20x** the
    per-token reading in each mode (pipeline 20.3 s, tensor 28.8 s at
    `20261002T194000Z-tier01b-item3-chunk-review`, re-read on the pre-item-5 build before implementation),
    median of three, with gRPC `ForwardPass` calls per prefill **<= ceil(512 / chunk) x nodes** read off the
    run. Set well outside the noise floor on purpose: removing 511 of every 512 round trips per shard and
    reaching the device region should be worth far more than 5x, and anything less means the window path
    is not what runs.
  - **Recorded, no threshold:** 2- and 3-way TP throughput against single-node dense on this host.
    Expect it to be below 1.0x. Report the number and the reason plainly rather than omitting it.

## Models needed

Existing dense models suffice for parity testing across the simulated shard counts step 0 makes
available on one physical machine (the existing `ClusterHarness` forks JVM processes, not physical hosts, so no additional
hardware is strictly required to validate correctness — only the single-process multi-GPU
sub-scope needs a second physical GPU, which isn't available here; flag that limitation when this
tier reaches that point).

## Exit criteria

- [ ] Step 0's node-count decision recorded in this file, with the split widths actually tested
      named explicitly — no claim of 2-way coverage unless 2-way ran.
- [ ] Real column-/row-parallel slicing implemented for every currently-supported transformer
      handler, numerically validated against single-node dense output.
- [ ] `TensorParallelClusterIT` exercises real slicing, not a dummy-fixed-logit stub.
- [ ] Item 5: prefill on `--pType pipeline`, `--pType tensor` and the standalone coordinator is batched
      per window through a contract change recorded in `inference.proto`; greedy output equals the
      per-token prefill's; the item 5 gate is met (512-token TinyLlama prefill <= 0.20x the per-token
      reading in each mode, `ForwardPass` calls per prefill <= ceil(512 / chunk) x nodes), with the run
      directory cited; the cluster `--prefill-batch` default re-read and documented.
- [ ] Single-process multi-GPU evaluated; implemented if justified, with N=1 degenerate-case
      validation and an explicit note about what couldn't be validated without a second physical
      GPU.
- [ ] Standing CPU and allocation gate met (README, "Test infrastructure"): CPU tg and pp
      >= 0.95x the pre-tier build, allocation per token <= 1.10x, in-span GC pause total <= 1.25x.
- [ ] Cross-surface checklist fully resolved; continuous+tensor-parallel fallback-to-static
      behavior explicitly confirmed unchanged (not silently made reachable in a half-working state).
- [ ] Perf gate published: parity gate met, single-node Juno t/s >= 0.95x the pre-tier build (same-hour A/B), AllReduce/gRPC
      per-token overhead recorded, and multi-node throughput reported as measured — including if it
      is below 1.0x, which is the expected result on a single-GPU host.
- [ ] Docs (`docs/agent-arch.txt`, `docs/howto.md`) updated to state plainly that tensor parallelism
      now does real sliced computation, replacing whatever Tier 00 documented as its prior
      (non-functional) state.
- [ ] `CHANGELOG.md` entry added.
