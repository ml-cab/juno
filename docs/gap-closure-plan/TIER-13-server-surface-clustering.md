# Tier 13: Server surface & clustering

Status: not started
Gap analysis refs: §1.12, §2.3, §2.6 (execution half — the decision from Tier 00, this tier does
the implementation if "implement" was chosen)

## Objective

Close the remaining server-surface gaps (a reranking endpoint; genuinely broader tool/function
calling — more chat templates supported, or a path to model-native function calling rather than
prompt-injection-only; elastic N-node clustering beyond the fixed 3-node `ClusterHarness` topology),
and wire `FaultTolerantPipeline` into the production cluster launch path (§2.3) so a lost node has
an actual failover story.

## Why this tier, why now

This is sequenced last among the feature tiers because elastic clustering and fault tolerance are
most valuable once everything they'd be protecting (real tensor parallelism from Tier 09, mature
continuous batching from Tier 07, broad architecture support from Tier 08) actually exists — fault
tolerance for a cluster running an incomplete feature set is lower-value than fault tolerance for
the finished system.

## Scope

### In scope

1. **`FaultTolerantPipeline` wiring**: connect the existing circuit-breaker/retry component to
   `CoordinatorMain.buildPipeline`/`buildTensorPipeline`, so a node failure in `./juno cluster`
   actually triggers the existing (currently unwired) retry/circuit-breaker logic instead of an
   unhandled failure. Since pipeline/tensor-parallel nodes hold unique, non-interchangeable shards,
   "failover" here likely means fast, clear failure reporting plus (if elastic clustering below is
   built) reassignment to a replacement node — not naive retry-to-a-replica, which doesn't make
   sense for a sharded topology. Design this explicitly rather than assuming the existing
   replica-pool-oriented `FaultTolerantPipeline` design transfers unchanged.
2. **Elastic N-node clustering**: move `ClusterHarness` and the production cluster launch path
   beyond the hard-coded 3-node topology, to a configurable node count. Tier 00 removed the
   never-implemented `RegistryService` RPCs and the unused Hazelcast dependency, so any node
   discovery this needs is designed here from scratch rather than revived from that surface.

   **Check what Tier 09 already took.** Tier 09's implementation step 0 may have hoisted the
   `ClusterHarness` node-count generalization into itself so its 2-way parity split could run at
   all. If it did, that part is done and this tier's remaining work is the production launch path,
   discovery, membership and fault tolerance — read Tier 09's recorded decision before starting
   rather than re-doing it.
3. **Reranking endpoint**: add a `/v1/rerank`-style endpoint (Juno-native naming), reusing the
   existing embeddings infrastructure's pooling logic where applicable.
4. **Broader tool/function calling**: extend `ToolPrompt.SUPPORTED_MODEL_TYPES` beyond
   `{llama3, chatml, qwen3}` to cover the chat templates already supported for plain generation
   (Phi-3, Mistral, Gemma, TinyLlama) where a correct tool-call prompt/parse format can be
   established for each.

### Out of scope

- A legacy `/v1/completions` endpoint — the gap analysis notes this exists in some other systems
  but doesn't flag it as something Juno's actual users need; explicitly declined unless the user
  says otherwise when this tier starts.
- Audio/Whisper-style multimodal — no gap-analysis finding suggested this is needed; out of scope
  unless requested.
- True model-native function calling (a model architecture that natively emits structured
  tool-call tokens rather than free text) — this is a training/model-format concern beyond what
  broadening the prompt-injection approach can achieve; flagged as a possible future tier if the
  user wants to pursue it, not attempted here.

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | reranking and broadened tool calling must work without a GPU |
| 2 | CUDA GPU inference | elastic clustering and fault tolerance must work with GPU-resident nodes |
| 3 | ROCm GPU inference | same, NEEDS-AMD-HARDWARE for final validation of any GPU-specific interaction |
| 4 | Static schedule | reranking is a new request type — confirm it integrates with `RequestScheduler` correctly (or is explicitly out-of-band like embeddings currently are, per `docs/howto.md:622-625`'s existing embeddings exception — decide and document which model reranking follows) |
| 5 | Continuous schedule | same consideration |
| 6 | Single-node local mode | reranking and broadened tool calling must work in `--local` mode first |
| 7 | Pipeline-parallel cluster | elastic clustering and fault-tolerance wiring are primary targets here |
| 8 | Tensor-parallel cluster | same |
| 9 | LoRA training | N/A for reranking/clustering; broadened tool calling should compose with LoRA-played models using the newly-supported templates |
| 10 | LoRA playback | same |
| 11 | Vision | confirm `/v1/vision/chat`'s existing "does not honor `tools`" restriction is re-verified, not accidentally changed by the broadened tool-calling work |
| 12 | OpenAI REST surface | reranking endpoint and broadened tool calling both surface here |
| 13 | Native REST surface | same, native equivalents if applicable |
| 14 | CLI | elastic node count needs a `./juno cluster --nodes N`-style flag (replacing the implicit fixed-3 assumption); `--help` updated |

## Implementation steps

1. Design the failover semantics for a sharded (non-replica) topology explicitly, before wiring
   anything — this is a real design decision, not a mechanical wiring task.
2. Wire `FaultTolerantPipeline` (or a redesigned equivalent, if the design step concludes the
   existing replica-oriented class doesn't fit) into `CoordinatorMain`.
3. Generalize `ClusterHarness` and the production launch path to a configurable node count.
4. Decide how elastic nodes are discovered (Tier 00 removed the old `RegistryService`/Hazelcast
   surface, so there is nothing to revive); implement only what the elastic design needs.
5. Add the reranking endpoint.
6. Extend tool-calling template support one template at a time, each validated with real
   tool-call round-trip tests.

## Tests to write/upgrade before implementation

- **New `FaultTolerantPipeline`/`CoordinatorMain` integration test**: kill a node mid-request in a
  forked-JVM cluster test, confirm the documented failover/failure-reporting behavior actually
  triggers (this test must fail against today's unwired state, confirming it's testing something
  real).
- **New `ClusterHarness` test**: N-node cluster for N other than 3 (e.g. 2, 5), confirming correct
  shard assignment and operation.
- **Membership/discovery integration test** (if the elastic design adds a discovery mechanism):
  dynamic node registration/deregistration correctness.
- **New `RerankHandlerTest`**: standard reranking correctness cases.
- **`ToolPromptTest`**: new cases per newly-supported chat template, round-tripping a tool call
  correctly.
- **`ModelLiveRunnerIT`**: add checks for elastic cluster sizes, a node-failure scenario, reranking,
  and each newly-supported tool-calling template.
- **New bash smoke script**: `scripts/performance-tests/smoke-tier13-server-clustering.sh` —
  drives an elastic-sized cluster, kills a node mid-run, exercises reranking, and exercises tool
  calling on each newly-supported template.
- **Perf gate**: only required if the fault-tolerance wiring or elastic clustering changes the
  per-request hot path (it shouldn't for the happy path, but confirm); reranking is a new
  code path, not a hot-path *change*, so the existing rule's "optional" carve-out likely applies —
  confirm at implementation time rather than assuming.

  **Threshold, if the gate applies.** Per-request latency on the happy path within **0.95x** of the
  pre-tier baseline on both schedules, median of three per the README's noise-floor rule — the
  circuit-breaker and retry wrapper sits on every request once wired, so "it only costs something
  when a node dies" needs measuring rather than assuming. Reranking's own number: end-to-end latency
  for a 10-document rerank at **<= 1.5x** the latency of a single chat completion of equivalent
  total token count on the same model, since it reuses the embedding path and should not be paying
  a per-document full-generation cost. If the gate is judged not to apply, record that judgement and
  the reasoning here rather than omitting the section.

## Models needed

Existing dense models suffice for reranking and tool-calling template extension. Elastic clustering
and fault-tolerance testing use the existing forked-JVM harness and don't need new model files.

## Exit criteria

- [ ] `FaultTolerantPipeline` (or its redesigned equivalent) is wired into the production cluster
      launch path, with tested, documented failure/failover behavior for a sharded topology.
- [ ] Cluster node count is configurable, not hard-coded to 3.
- [ ] No ambiguous membership state: either the elastic design's discovery mechanism is implemented
      and tested, or membership stays fixed at launch and is documented as such (Tier 00 removed
      `RegistryService` and Hazelcast).
- [ ] Reranking endpoint implemented and tested, and declared in
      `api/src/main/resources/openapi.yaml` and `juno-api.yaml` alongside the code that serves it
      (README feature-complete rule). Any gRPC change from the elastic-clustering work is reflected
      in `api/src/main/proto/inference.proto` in the same change.
- [ ] Tool calling works on at least one additional chat template beyond
      `{llama3, chatml, qwen3}`, with the remaining unsupported templates still failing closed
      correctly.
- [ ] Cross-surface checklist fully resolved.
- [ ] Docs (`docs/howto.md`, `docs/agent-arch.txt`) updated, Juno-native language only.
- [ ] `CHANGELOG.md` entry added.
