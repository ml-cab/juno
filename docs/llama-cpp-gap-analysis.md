# Juno vs. llama.cpp: gap analysis and internal inconsistencies

Snapshot date: 2026-09-18, branch `67-inference`, HEAD at commit `0c519f1`.

This report is a code-verified comparison of Juno's current implementation against llama.cpp's
architecture, plus a catalog of internal inconsistencies found while reading the source. It does
**not** draw on `docs/infra-plan/` (excluded per instruction — that directory is known to be
unreliable). Every claim below is grounded in `node/`, `coordinator/`, `kvcache/`, `sampler/`,
`lora/`, `vision/`, `api/` source, plus `docs/agent-arch.txt`, `docs/howto.md`,
`docs/performance.md`, `CHANGELOG.md`, and `CLAUDE.md`. File:line citations are given where they
materially support a claim; treat this document itself as a snapshot to re-verify against source
before acting on it, the same way memory files are treated.

This is intended as raw material for a future planning pass, not a plan itself — it deliberately
does not prioritize or sequence the items below.

---

## 1. What Juno can take from llama.cpp

Organized by subsystem, roughly in the order a reader would hit them going from disk to output
token.

### 1.1 Quantization coverage

- **Missing formats entirely**: Q4_1, Q5_0, Q5_1, and the whole IQ1/IQ2/IQ3/IQ4 importance-quantized
  family. `GgufReader.loadTensor()`'s switch throws `UnsupportedOperationException` for any of
  these — a GGUF using them fails to load at all. Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, F16/BF16/F32
  are supported.
- **GPU-packed (MMQ) kernels exist only for Q4_K/Q5_K/Q6_K** (`Q4KMmqKernel.java`, CUDA-only,
  `q4k_gemv.cu`). Q2_K, Q3_K, Q8_0, and Q4_0 have no device-resident packed kernel — if offloaded at
  all, they go through a generic dequantize-to-FP32/FP16 path, losing the VRAM-compression benefit
  that is the whole point of shipping a small quant. llama.cpp's ggml CUDA backend has MMQ kernels
  for effectively all quant types.
- **ROCm has zero fused K-quant kernel** — `RocmMatVec.java` never references `DeviceQ4KMatrix`.
  Even the three types with a CUDA MMQ kernel fall back to plain dequantized matmul on AMD GPUs.
- **No `quantize` tool**: Juno can decode/load pre-quantized GGUF but has no CLI path to produce a
  quantized GGUF from an F32/F16 checkpoint. `merge` (LoRA-into-GGUF) explicitly avoids
  re-quantizing rather than implementing it (`docs/howto.md:363-364`). Anyone wanting to quantize a
  new model for Juno currently has to reach for another project's tool first.

### 1.2 Attention and long context

- **No FlashAttention-equivalent.** `gqa_attention.cu`'s own header comment calls itself "a
  straightforward parallel kernel, not a tiled/online-softmax FlashAttention design" — it fully
  materializes QK^T in a scratch buffer per block instead of streaming/recomputing, so it pays
  O(seq²) memory and misses the bandwidth savings FlashAttention-style kernels get on long contexts.
- **No sliding-window attention** (Mistral-style windowed causal mask) anywhere in `node/`.
- **No context-shifting.** Every KV backend (`DenseKvTensor`, `KvPageTable`, `PagedKvTensor`,
  `DeviceKvCache`) throws `IllegalStateException` once a sequence exceeds `MAX_SEQ_LEN` (32768).
  llama.cpp's context-shift (drop oldest N tokens, keep going) has no equivalent — long-running
  chats simply hard-fail once the cap is hit.
- YaRN/RoPE scaling *is* implemented (`Qwen3Rope.java`, `Phi3RopeConfig.java`) and is a genuine
  parity point, not a gap.

### 1.3 Batching, KV cache, and prefix reuse

- **"Paged" KV is a per-request chunked allocator, not PagedAttention.** `KvPageTable.gather()`
  always copies the full logical prefix out of pages into a contiguous scratch buffer before
  attention runs (the "gather tax", `docs/performance.md:134-149`); there is no block-table-aware
  attention kernel that reads pages in place, and pages are never shared across two different
  requests' page tables even for identical prompts. vLLM/llama.cpp-server's PagedAttention avoids
  both the copy and the duplication.
- **Continuous scheduling caps out around 8 concurrent slots** (`ServeSchedulePolicy
  .DEFAULT_RUNNING_SET = 8`) and schedules by slot count, not token budget — unlike vLLM's
  continuous batching, which sizes each step by a token budget so it can pack many more small
  requests per step. Juno's own bake-off shows continuous multi-session throughput at 0.86x static
  (`docs/performance.md:177`), i.e. it is not yet a win.
- **No disk-backed session/prompt-cache persistence.** KV state lives in RAM only, tied to a JVM
  process's lifetime — llama.cpp's `--prompt-cache` session files (save KV to disk, reload in a
  later process) have no Juno equivalent. `CHANGELOG.md:1758` documents this as a deliberate
  simplification from an earlier three-tier (GPU + off-heap + disk) design.
- **No KV cache defragmentation** — dense KV only grows, paged KV's block pool only tracks
  free/live pages with no compaction.
- **Cross-session prefix sharing doesn't exist by design** (`docs/howto.md:59`: "shared system
  prompts across different sessions do not skip prefill") — llama.cpp server implementations
  commonly share a prefix cache across otherwise-unrelated requests when the leading tokens match.
- KV quantization is limited to F16(really fp32)/Q8_0 — no Q4_0/Q4_1 KV quant tier for
  memory-constrained long-context serving.

### 1.4 Sampling and grammar

- **Missing sampler strategies**: min-p, typical-p (locally typical sampling), tail-free sampling,
  mirostat v1/v2, DRY (repeat-suppression), XTC. `SamplingParams` has no fields for any of these,
  and `Sampler.create()` hardcodes exactly seven fixed steps in a fixed order — there is no
  `--samplers`-equivalent to choose or reorder the chain, unlike llama.cpp's composable
  `llama_sampler_chain`.
- **OpenAI `frequency_penalty` is faked** by converting it into Juno's binary
  repetition-penalty step (`OpenAiAdapter.java:36-41`) rather than implementing true
  per-occurrence-count scaling — a real behavioral gap for any client relying on OpenAI's actual
  frequency-penalty semantics.
- **GBNF grammar gaps**: char classes are byte-only (reject codepoints > 255,
  `GbnfGrammar.java:670-671`), and bounded repetition `{min,max}` is artificially capped at
  `max ≤ min + 8` (`GbnfGrammar.java:450-451`) — an arbitrary ceiling with no equivalent in
  llama.cpp's grammar engine.
- **JSON-Schema-to-grammar covers a much smaller subset** than llama.cpp's
  `json-schema-to-grammar.cpp`: no `$ref`, `oneOf`/`anyOf`/`allOf`/`not`/`if-then-else`, `pattern`,
  `format`, or any of the numeric/string bound keywords (`minLength`/`maxLength`/`minimum`/
  `maximum`/etc). These fail closed with HTTP 400 rather than silently ignoring the constraint,
  which is the right failure mode, but the practical effect is that many real-world JSON schemas
  (anything using `$ref` for shared definitions, which is extremely common) simply cannot be used
  for structured output today.

### 1.5 Speculative decoding

- **Draft-model speculative decoding currently regresses to 0.52x** instead of speeding anything up
  (`docs/performance.md:641-680`). Root cause, per the project's own measurement: the draft model's
  own forward passes route through the same per-launch-overhead-bound GPU dispatch path as the
  target model, so a second real model's sequential decode isn't offset by the verify-side batching
  savings on the tested hardware (GTX 1080). This is not a llama.cpp feature gap per se — llama.cpp
  has the identical sequential-draft-forward structure — but llama.cpp's lower per-launch dispatch
  overhead (see §1.6) is exactly why the same algorithm helps there and hurts here. Closing this
  requires fixing the dispatch-overhead problem, not the speculative-decoding logic itself.
- **No lookahead decoding** (Jacobi-iteration n-gram-pool, llama.cpp's `examples/lookahead`) and
  **no Medusa-style multi-head or self-speculative decoding** — Juno has exactly two strategies
  (ngram-simple, draft-simple) vs. llama.cpp's broader menu.
- Speculative decoding, and per-request LoRA adapters, are wired for the single-request
  `generate()` path only — `generateBatch()` (static multi-request) and MoE/Phi/cluster pipelines
  fall back to the slower, correctness-preserving serial default.

### 1.6 The recurring GPU dispatch-overhead problem

This shows up independently in three different subsystems investigated this session, which makes
it worth calling out as one root cause rather than three separate ones:

1. `CudaGraphSession`/`CudaRmsNorm` (commit `0c519f1`): moving RMSNorm to GPU measured **6.9x
   slower** than CPU scalar per-call, and even with CUDA-graph launch-fusion, still 2.2x slower —
   because `MatVec`'s API is host-`float[]`-in/host-`float[]`-out, so every GPU op still pays a host
   round-trip before and after. `RopeKernel`/`ResidualAddKernel`/`SwiGluKernel` were never even built
   as a result (`docs/performance.md`, "Phase B checkpoint: no-go").
2. Draft-model speculative decoding (§1.5): the extra model's serial forward passes are dominated by
   per-launch host/FFI dispatch cost, not FLOPs.
3. `VectorQuantKernels.dot()` (CPU SIMD): rejected from the hot path because per-call dispatch
   overhead at vision-scale batch widths made it "tens to hundreds of times slower than sequential
   matVec" on hosts where `SPECIES_PREFERRED` is only 128-bit.

llama.cpp's ggml graph executor amortizes exactly this kind of overhead by building a whole
computation graph once and executing it with minimal per-op host involvement (and on GPU, CUDA
graphs capture the *entire* per-token op sequence, not a single op at a time). Juno's `MatVec`
interface is fundamentally op-at-a-time with host round-trips between ops; an "activation stays
resident on GPU across a full transformer layer (or several)" redesign is the kind of change that
would make GPU-resident RMSNorm/RoPE/SwiGLU, CUDA-graph replay, and draft-model speculative
decoding all pay off at once, instead of each being independently rejected as "measured, doesn't
help, left as scaffolding."

### 1.7 GPU backend breadth

- CUDA and ROCm/HIP only. No Metal (Apple Silicon), Vulkan, SYCL/oneAPI, or CANN backend exists —
  llama.cpp supports all of these. This is a real reach/hardware-coverage gap, not just a
  performance one: Juno currently cannot run GPU-accelerated on a Mac at all.
- CPU fallback has **no actual SIMD in the hot path**. `CpuMatVec`'s core dot-product loop is scalar
  Java parallelized only via `IntStream.parallel()` (thread-level, not vector-level); the
  `jdk.incubator.vector`-based kernel exists but is deliberately excluded from the hot path due to
  measured regressions. llama.cpp's ggml CPU backend has hand-tuned AVX2/AVX-512/NEON kernels for
  every quant type. This is worth calling out because `CLAUDE.md`'s own module table describes
  `node` as having "JDK Vector API (CPU kernels)" — true of the code that exists, misleading about
  what's actually on by default (see §2.7).
- CUDA and ROCm both lack a batched tiled-GEMM path for large prefill batches on ROCm specifically
  (CUDA has one via `CudaFp16GemmOps`; ROCm only has strided-batched-GEMV, so large-batch ROCm
  prefill is GEMV-looped, not a real GEMM). ROCm is, in practice, a second-class backend despite the
  "vendor-neutral by design" framing in `CLAUDE.md`.

### 1.8 Multi-GPU / tensor parallelism

- Juno has **no NCCL/RCCL** and no single-process multi-GPU path. All multi-GPU scaling is
  multi-process via gRPC (one JVM/one GPU per node), with cross-node reduction done as a plain
  client-side float-array sum ("star AllReduce") rather than a collective-communication library.
  This is a deliberate design choice (`assets/meta-juno-doc.md`: "Java distributed tooling
  (Hazelcast, gRPC) over NCCL/MPI"), not an oversight, but it does mean Juno cannot currently do
  fast intra-host multi-GPU tensor parallelism the way llama.cpp/vLLM can with NCCL — every
  cross-GPU exchange pays a gRPC round trip. See also §2.2 for a deeper problem in the TP path
  itself.

### 1.9 MoE model support

- Only Qwen3-MoE has a mixture-of-experts handler. Mixtral is not supported (its GGUF
  `general.architecture` is `llama`, same string as dense Llama/Mistral, so it falls into
  `LlamaTransformerHandler`, which has no `ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps` code path at
  all). DeepSeek-MoE and Gemma-MoE are likewise unsupported. llama.cpp supports the whole family of
  popular MoE architectures through its generic ggml graph. See §2.4 for the correctness-risk angle
  of this (does it fail closed, or silently produce wrong output?).

### 1.10 Vision

- No dynamic/high-resolution tiling ("anyres" — LLaVA-1.6/NeXT-style tile+thumbnail composition).
  Juno's `ImagePatchEmbedder` does a single fixed square resize per image.
- No multi-image-per-request support (one `image` part per chat request).
- No native dynamic-resolution ViT (Qwen2-VL-style 2D-RoPE variable patch grids) — fixed learned
  absolute position embeddings only.
- Vision is `--local` mode only; cluster mode doesn't wire it at all.

### 1.11 LoRA

- GGUF-LoRA adapters can only be used after an offline one-way conversion into Juno's own `.lora`
  format (`GgufLoraImporter` + `juno lora-import`) — there is no native GGUF-LoRA inference path the
  way llama.cpp loads a GGUF-LoRA adapter directly at request time.
- No hot-swapping: `--lora-play` is a process-wide, startup-time flag; per-request adapter selection
  (`x_juno_loras`) is explicitly unwired and fails closed on every schedule.
- No gradient checkpointing / activation checkpointing, and no QLoRA-style quantized-base-weight
  training (base weights are dequantized to FP16/FP32 for training).

### 1.12 Server surface

- No `/v1/completions` (legacy) endpoint, no reranking endpoint, no audio/Whisper-style multimodal.
- Tool/function calling is prompt-injection + grammar-masking rather than genuine model-native
  function calling, and is fail-closed on any chat template other than llama3/chatml/qwen3 (Phi-3,
  Mistral, Gemma, TinyLlama, vision templates all reject tool use with HTTP 400 today).
- `RegistryService` (dynamic cluster membership via Hazelcast) is defined in the proto and has
  Hazelcast declared as a dependency in 5+ module `pom.xml`s, but has **zero implementation** —
  dead surface area that currently buys nothing. Either build it or remove the unused dependency and
  proto surface.
- Cluster topology is hard-capped at exactly 3 nodes in `ClusterHarness` for both pipeline- and
  tensor-parallel modes — no elastic N-node clustering exists yet, unlike llama.cpp's more flexible
  (if more minimal) `rpc-server`.

---

## 2. Internal inconsistencies in Juno

These are places where the code, comments, or documentation disagree with each other, or where a
component is more (or less) built than its surrounding description suggests. This section is about
Juno's own house-keeping, independent of llama.cpp comparison.

### 2.1 Sampler pipeline: three different described orders, one actual order

`Sampler.java`'s real runtime order (verified in code, `Sampler.java:99-106`) is:
`presencePenalty → repetitionPenalty → temperature → topK → softmax → topP → sample`.
`SamplingStep.java:20-21`'s class javadoc describes a *different* order: `temperature → topK →
topP → softmax → penalty → sample`. `RepetitionPenaltyStep.java:22` separately labels itself "Step
5" even though the real pipeline runs it second. Three sources of truth in one small module,
only one of which is correct. This is a low-cost but real doc-drift problem — worth a
documentation pass, and a general signal that per-file javadoc "pipeline order" comments in this
codebase should not be trusted without checking the orchestrating class.

### 2.2 `TensorShardContext` describes work that was never wired in

`TensorShardContext.java`'s javadoc describes column-parallel Q/K/V projection and row-parallel
output projection with per-layer AllReduce — a real tensor-parallel compute design. But this class
is referenced **only** by its own unit test and by `ClusterHarness` (the test harness); there is no
column/row-parallel slicing logic anywhere in `LlamaTransformerHandler`, `Phi3TransformerHandler`,
`Qwen3TransformerHandler`, or any other handler. `TensorParallelClusterIT` — the integration test
that exercises the tensor-parallel path end to end — runs each of the 3 forked node JVMs against a
`CyclicForwardPassHandler` **stub** that returns a fixed dummy logit regardless of rank, and only
checks that the AllReduce sum still argmaxes correctly. In other words: the gRPC transport, the
AllReduce reduction, and the shard-assignment data model are real and tested; the actual per-layer
weight-sliced forward pass that would make tensor parallelism *do* anything useful against a real
model is not implemented. Anyone reading `TensorShardContext`'s javadoc without checking call sites
would reasonably believe tensor parallelism is functionally complete. It is not — it currently only
broadcasts the same full computation to every node and reduces (mathematically inert unless the
per-node computations actually differ).

### 2.3 `FaultTolerantPipeline` is tested but not wired into the production cluster path

`FaultTolerantPipeline` implements circuit-breaker + retry across replica pipelines and has its own
unit tests, but is referenced only by `HealthReactor` and those tests — not by `CoordinatorMain`,
`ConsoleMain`, `ProcessPipelineClient`, or `TensorParallelPipelineClient`. The actual
`CoordinatorMain.buildPipeline`/`buildTensorPipeline` construct pipeline clients directly with no
retry/reconnect logic. Since pipeline/tensor-parallel nodes each hold a unique, non-interchangeable
shard, a lost node in production `./juno cluster` currently has no failover path despite a
fault-tolerance component existing in the codebase.

### 2.4 MoE architecture mismatch: unclear if this fails closed

`Qwen3MoeTransformerHandler` fails closed (`IOException`) if loaded against a GGUF where
`expert_count == 0`. It is not confirmed from this research pass whether the inverse case — a
Mixtral-style GGUF (architecture string `llama`, but containing `ffn_gate_exps`/`ffn_up_exps`/
`ffn_down_exps` tensors) loaded into `LlamaTransformerHandler` — is detected and rejected, or
whether it would silently run using only the dense-path tensors it recognizes and ignore the expert
tensors, producing wrong output without an error. Given the project's stated fail-closed design
principle (`CLAUDE.md:88-93`, and the ~30 other fail-closed call sites found across the codebase),
this asymmetry is worth verifying directly and fixing if it is in fact a silent-degrade path — it
would be the one place that principle is not enforced.

### 2.5 `generateBatch()`'s prefix cache is not session-gated, unlike every other path

`GenerationLoop.generate()` (single-request) and `ContinuousBatchEngine` both gate prefix-cache
reads/writes on `request.sessionId() != null`, specifically because an earlier bug
(`CHANGELOG.md:1716`, "Session 7") let the engine register a prefix-cache hit and then evict the
underlying KV, serving stale/freed KV to a later request. `GenerationLoop.generateBatch()` — the
static multi-request batching path — has no such gate: it calls `kvCache.findLongestPrefix()` and
`kvCache.cachePrefix()` unconditionally for every request in the batch, session or not, against the
one shared `PrefixCache` trie, and evicts each request's pipeline KV right after it finishes. Two
unrelated stateless requests processed in the same or a later static batch that happen to share a
common prompt prefix (e.g. the same system prompt from two different API clients) could register a
trie hit and then have a later request skip prefill against KV that was never written under its own
request ID — the same bug class already fixed once for the single-request path, apparently not
carried over to the batch path. This was found by code reading, not by reproducing the failure at
runtime; it should be verified with a targeted test (two concurrent stateless `--parallel`-batched
requests sharing a system prompt) before being treated as confirmed, but it is a plausible real
correctness bug and the single highest-priority item in this report to actually chase down.

### 2.6 `RegistryService`/Hazelcast: declared, never implemented

`RegistryService` (`GetShardMap`/`RegisterNode`/`RecomputeShards`) is defined in `inference.proto`
with a comment describing it as "Hazelcast-backed," and `hazelcast` is a declared Maven dependency
in the root `pom.xml` and 5+ module `pom.xml`s. Neither the gRPC service nor any actual Hazelcast
usage exists anywhere in `src/main` — dynamic cluster membership is aspirational proto surface with
an unused dependency sitting behind it. This inflates the perceived architecture (a reader of the
proto file would assume dynamic membership exists) and adds real dependency-management overhead
(CVE tracking, version bumps) for a library that does nothing today.

### 2.7 "JDK Vector API CPU kernels" oversells what's on the hot path

`CLAUDE.md`'s module table describes `node` as containing "`VectorQuantKernels`/`SimdThreadPool`
(JDK Vector API CPU kernels)" without qualification. In practice, only the Q8_0 dequantize step is
vectorized in production; the general-purpose SIMD dot-product kernel is implemented, tested, and
then explicitly excluded from the hot path because it regresses badly at large batch widths on
hosts with a narrow `SPECIES_PREFERRED`. `SimdThreadPool`'s own dedicated thread pool is similarly
unused in the actual hot path (`forEachRow` uses `ForkJoinPool.commonPool()` directly, because an
earlier attempt to route through the dedicated pool caused a 37-260x regression from nested-parallel
-stream pathology). The module *has* Vector API code; the module's *default runtime behavior* is
thread-parallel scalar Java. This is worth a one-line doc correction so future readers don't assume
CPU inference is SIMD-accelerated when it mostly is not.

### 2.8 CudaGraphSession/CudaRmsNorm: tested, dormant, still in the tree

Commit `0c519f1` added `CudaGraphSession`/`CudaRmsNorm` with real unit tests on real hardware, then
concluded (per `docs/performance.md`'s Phase B analysis) that the approach regresses decode and left
the code in place, unwired, as "scaffolding" for a future activation-residency redesign. This is
honest and well-documented, but it's the second GPU-kernel effort (after speculative decoding) that
followed the same "build → measure → regress → leave as dead code rather than deleting or fixing
the underlying dispatch-overhead problem" pattern. Left unaddressed, this pattern will keep
producing more dormant scaffolding every time someone tries to move another elementwise op to GPU,
until the actual root cause (§1.6) is fixed once.

### 2.9 Minor doc/code naming mismatches

- `CLAUDE.md` describes LoRA training as using an "Adam optimizer"; `LoraAdamOptimizer` actually
  implements AdamW (decoupled weight decay) plus LoRA+ (separate A/B learning rates) — a materially
  better optimizer than what's documented, just mislabeled.
- The GBNF bounded-repetition cap (`max ≤ min + 8`) has no comment explaining why 8 specifically was
  chosen, making it read as an arbitrary implementation shortcut rather than an intentional limit —
  worth either documenting the reasoning or removing the cap.

---

## 3. How Juno appears to have been built, and what that suggests should change

Reading across all five investigated subsystems, a consistent engineering pattern emerges:

**Strengths, evident from the code:**
- Heavy reliance on real hardware measurement (JFR events, live GPU smoke tests) rather than
  assumption — several real bugs (Q6_K dequant corruption, an off-by-one KV corruption in ngram
  speculative decoding, vision RoPE/normalization bugs) were caught this way, with root causes
  documented in commit messages and comments rather than just silently patched.
  Failures are reported *honestly* even when a feature doesn't pan out (the draft-model 0.52x
  regression, the continuous-batching 0.86x throughput result, the GPU-resident-RMSNorm no-go) rather
  than hidden or spun.
- A genuine "fail closed over silently degrade" discipline is applied broadly and consistently
  (~30 call sites): unsupported chat templates, JSON schema keywords, distributed-pipeline
  embeddings, GPU-training-without-a-GPU, cluster+speculative-decoding combinations all raise
  explicit errors instead of producing subtly wrong output. This is a real, deliberate strength and
  the one place §2.4 is worth double-checking specifically because it's the one place the pattern
  might not hold.
- Vendor-neutral GPU abstraction is a stated goal and mostly followed at the API boundary
  (`GpuBindings.createMatVec()` has no `instanceof` dispatch) — even though ROCm is behind CUDA in
  actual kernel coverage (§1.7), the *interface* discipline that would let that gap be closed later
  without an API redesign is in place.

**What this pattern has produced that should change:**
- **Docs and javadoc comments describe intent/design more often than they describe implemented
  reality**, and several of them (`TensorShardContext`, "JDK Vector API CPU kernels", the sampler
  pipeline order comments, `RegistryService`) have drifted far enough from the actual wiring that a
  reader — human or agent — would draw materially wrong conclusions about what Juno can currently
  do. A periodic "does this comment's claim have a call site" audit for load-bearing architectural
  claims (not every comment) would catch this before it compounds further.
- **Measured-regression scaffolding accumulates instead of getting consolidated into a single root
  fix.** Three independent efforts (GPU-resident elementwise ops, draft-model speculative decoding,
  CPU SIMD hot path) hit the same underlying wall — per-operation dispatch overhead with no
  cross-op residency — and each was individually shelved rather than being recognized as one
  problem. The activation-residency redesign implied by §1.6 is the highest-leverage architectural
  investment visible in this codebase: it is the prerequisite for GPU-resident norm/RoPE/SwiGLU,
  for draft-model speculative decoding to actually pay off, and likely for a future
  FlashAttention-style fused kernel, all at once — rather than three separate narrow fixes.
- **Feature completeness is uneven across surfaces in ways that aren't always flagged.** Continuous
  batching, tensor parallelism (in the deeper sense of §2.2), and per-request LoRA are all
  "built enough to pass their own integration tests" without being built to the point of doing the
  thing their name implies in production. This isn't dishonest — each gap is documented somewhere
  — but the documentation is scattered (a javadoc here, a CHANGELOG line there, a howto.md caveat
  elsewhere) rather than centralized, making it easy for the true state of a given subsystem to get
  lost between sessions. A single "known limitations" ledger per major subsystem, kept next to the
  code it describes rather than in a separate planning doc, would reduce this.
- **Dead/unused surface area should be pruned or finished, not left ambiguous.** `RegistryService`
  + Hazelcast (§2.6) and the tensor-parallel per-layer slicing (§2.2) are the two clearest examples:
  each looks, from its proto/javadoc, like a real capability, but is actually 0% and ~20% built
  respectively. Either investment should continue to completion, or the surface (proto RPCs, unused
  dependencies, misleading javadoc) should be removed so the codebase's stated capabilities match
  its actual ones.

---

## Appendix: subsystem-by-subsystem source map used for this report

- Quantization / GEMM / GPU kernels: `node/src/main/java/cab/ml/juno/node/GgufReader.java`,
  `GgufKQuantCodec.java`, `QuantizationLayout.java`, `MatVec.java`, `CpuMatVec.java`,
  `CudaMatVec.java`, `RocmMatVec.java`, `Q4KMmqKernel.java`, `node/src/main/cuda/*.cu`,
  `CudaGraphSession.java`, `VectorQuantKernels.java`, `SimdThreadPool.java`,
  `TensorShardContext.java`.
- Scheduling / KV cache: `coordinator/src/main/java/cab/ml/juno/coordinator/RequestScheduler.java`,
  `GenerationLoop.java`, `ContinuousBatchEngine.java`, `ContinuousMixedStepPolicy.java`,
  `ContinuousPrefillState.java`, `ServeSchedulePolicy.java`, `BatchConfig.java`,
  `kvcache/src/main/java/cab/ml/juno/kvcache/DenseKvTensor.java`, `Q8_0KvCodec.java`,
  `KvPageTable.java`, `PagedKvTensor.java`, `PagedKvArena.java`, `KvBlockPool.java`,
  `PrefixCache.java`, `KVCacheManager.java`.
- Sampling / grammar / speculative decoding: `sampler/src/main/java/cab/ml/juno/sampler/*.java`,
  `coordinator/.../DraftProposer.java`, `DraftModelSession.java`, `NgramDraftCache.java`,
  `node/.../ForwardPassHandler.java`, `LocalInferencePipeline.java`.
- LoRA / vision: `lora/src/main/java/cab/ml/juno/lora/*.java`,
  `node/src/main/java/cab/ml/juno/node/GgufLoraImporter.java`, `LoraResidentWeights.java`,
  `LoraMmqPolicy.java`, `vision/src/main/java/cab/ml/juno/vision/*.java`.
- Server / API / clustering: `coordinator/.../InferenceApiServer.java`, `OpenAiChatHandler.java`,
  `OpenAiTools.java`, `OpenAiResponseFormat.java`, `EmbeddingsHandler.java`,
  `FaultTolerantPipeline.java`, `node/.../ForwardPassHandlerLoader.java`, `GpuContext.java`,
  `GpuBindings.java`, `api/src/main/proto/inference.proto`, `juno-player/.../ClusterHarness.java`,
  `juno-master/.../CoordinatorMain.java`, `scripts/run.sh`.
