# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Juno ("Java Unified Neural Orchestration") — distributed LLM inference and LoRA fine-tuning on the
JVM, pure Java (no Python, no GIL). CUDA/ROCm GPU acceleration via Panama FFI. JDK 25+, Maven 3.9+
multi-module project.

## Build, test, run

```bash
mvn clean package -DskipTests          # build — juno-player emits thin jar + *-shaded.jar runnable

mvn test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,vision,metrics,juno-player
                                        # unit tests — no model file, no GPU needed
mvn test -pl node -Dtest=ClassName     # single test class
mvn test -pl node -Dtest=ClassName#methodName   # single test method

mvn verify -pl juno-master             # integration tests — forks 3 JVM nodes (stub mode)
                                        # includes ThreeNodeClusterIT and TensorParallelClusterIT
mvn verify -pl juno-master -Pintegration -DMODELS=/abs/a.gguf,/abs/b.gguf
                                        # ModelLiveRunnerIT — requires real model files

./juno test --model-path /path/to/model.gguf   # real-model smoke test (10 checks, exits 0/1)
```

GPU tests (require the corresponding vendor toolkit + GPU):

```bash
mvn test -Dgroups=gpu -pl node --enable-native-access=ALL-UNNAMED    # NVIDIA/CUDA
mvn test -Dgroups=rocm -pl node --enable-native-access=ALL-UNNAMED   # AMD/ROCm, Linux only
```

Run locally (single-JVM REPL, fastest startup — use for everyday experimentation):

```bash
./juno local --model-path models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
./juno local --model-path models/... --api-port 8080   # + REST API (OpenAI-compatible)
```

`./juno cluster ...` runs a real 3-node forked-JVM cluster (gRPC pipeline/tensor parallel) —
use for GPU deployments and multi-node scenarios. `./juno lora ...` is the LoRA fine-tuning REPL;
`./juno merge ...` bakes a trained `.lora` adapter into a standalone GGUF. Each subcommand has
`--help` with full flag docs. `models/` holds local `.gguf`/`.llamafile` test fixtures — do not
commit changes there or read them into context; they are large binaries.

`--add-modules jdk.incubator.vector` is required at compile and run time for `VectorQuantKernels`;
already wired into `pom.xml`, `scripts/run.sh`/`run.bat`, and `mvn test`'s surefire `argLine`.
Today only its Q8_0 dequantize step is on the default CPU path (the general SIMD dot-product kernel
is implemented and tested but deliberately not used there, see the `node` row below). Without the module the
kernels fall back to a scalar path (correctness-preserving, slower).

## Architecture

Module dependency graph (acyclic): `node` depends on `api, lora, registry, kvcache, health`;
`vision` depends on `node, coordinator, registry, tokenizer, sampler`; `coordinator` depends on
`node`; `juno-player` depends on `vision, coordinator, node, registry, ...`; `juno-master` and
`juno-node` are the shaded standalone executables.

| Module | Role |
|---|---|
| `api` | OpenAPI spec, protobuf/gRPC contracts |
| `registry` | Shard planning, model registry (`ModelIdResolver` — shared model-id-to-loaded-model resolution logic used by REST handlers) |
| `coordinator` | `Scheduler`, `GenerationLoop`, REST (`InferenceApiServer`); `ContinuousBatchEngine`/`ContinuousPrefillState` for the `continuous` schedule (paged KV, mixed prefill/decode steps); `ServeSchedulePolicy` auto-falls back cluster mode to `static` |
| `node` | `GgufReader`, transformer handlers (`LlamaTransformerHandler`, `Phi2TransformerHandler`, `Phi3TransformerHandler`), `MatVec` backends (`CpuMatVec`/`CudaMatVec`/`RocmMatVec`), GPU FFI (`GpuBindings`→`CudaBindings`/`RocmBindings`, vendor-neutral), `VectorQuantKernels`/`SimdThreadPool` (JDK Vector API helpers; the default CPU matmul path is thread-parallel scalar Java via `SimdThreadPool.forEachRow` on the common pool, with only the Q8_0 dequantize step vectorized, because the general SIMD dot kernel regressed at vision-scale batch widths on narrow-vector hosts) |
| `kvcache` | Dense KV (`DenseKvTensor`, `Q8_0KvCodec`) for `static` schedule; paged KV (`KvPageTable`/`PagedKvTensor`/`PagedKvArena`) for `continuous`; handlers dual-path via `SessionKvLayout` |
| `lora` | Adapter tensors, `.lora` checkpoint format, AdamW optimizer with LoRA+ learning-rate groups (no GGUF, no CUDA) |
| `vision` | Image-to-text: GGUF-based CLIP ViT encoder + `ForwardPassHandler` decorator. `LlavaHandlerFactory` is the only place that knows "this is a vision model" — kept out of `node` to avoid a dependency cycle. Serves `POST /v1/vision/chat` |
| `tokenizer`, `sampler`, `health`, `metrics` | Shared infrastructure; `metrics` carries JFR instrumentation |
| `juno-player` | `ConsoleMain` REPL, `ClusterHarness`, `LoraMergeMain`, `JunoPlayer`/`LoraTrainer`/`JunoHttpClient` (JVM embedding facade) |
| `juno-master`, `juno-node` | Shaded deploy jars (standalone coordinator / node executables) |
| `juno-bom` | Maven BOM — aligned versions for all `cab.ml` artifacts |

Key cross-cutting behaviors to know before touching inference/serving code:

- **Two schedules**: `static` (default; dense KV, fixed micro-batch via `--parallel`) vs.
  `continuous` (`--schedule continuous`; paged KV, running-set batching, long prompts chunk via
  `--prefill-batch` mixed with decode). Cluster launchers auto-fallback continuous→static.
- **GPU backend selection is vendor-neutral by design**: `GpuContext.selectBindings()` auto-detects
  CUDA first then ROCm (override `-Djuno.gpu.backend=cuda|rocm|auto`); `GpuBindings.createMatVec()`
  dispatches without `instanceof` checks. New GPU functionality should go through `GpuBindings`, not
  a vendor-specific class, unless the feature is genuinely CUDA-only (e.g. the packed-Q4 GEMV path).
- **Vision wiring is load-bearing and order-sensitive**: `ConsoleMain.runLocalRepl()` must call
  `prepareVisionHandler()` (which wraps the first node's handler in `VisionAwareForwardPassHandler`)
  *before* `LocalInferencePipeline.from()`, because the pipeline snapshots handler references at
  construction and never re-reads them. Vision is `--local` mode only — `runClusterRepl()` does not
  wire it.
- **OpenAI API surface** (`OpenAiChatHandler`): `stop`/`seed`/`presence_penalty` map into
  `SamplingParams`; `response_format` `json_object`/`json_schema` and `tools`/`tool_choice` compile
  to GBNF and mask illegal tokens before sampling (`GrammarSession`, `ToolCallGrammar`,
  `JsonSchemaToGbnf`); unsupported schema keywords and tool-calling on unsupported chat templates
  fail closed (HTTP 400) rather than silently ignoring the request.

See `docs/agent-arch.txt` for the full low-level module/class map (kept up to date per-change —
update it alongside `docs/howto.md` and `README.md` when architecture changes), `docs/howto.md` for
CLI/flag/API usage, and `docs/performance.md` + `docs/perf-compare/` for benchmark methodology and
results.

## Project conventions

- **Implementation approach**: unit tests first, but only for valuable business logic — don't test
  trivially. Design implementation details with performance in mind (this is a performance-sensitive
  inference engine). Follow KISS. Prefer adding a new Java class over extending an existing one.
  No emojis in code, comments, docs, or CLI/log output — be strict and precise.
- When a change is user-facing or touches architecture, update the relevant docs alongside the code:
  `docs/agent-arch.txt` (architecture/class map), `docs/howto.md` (CLI/API usage), `README.md`.
- **No competitor product names** (`llama.cpp`, `vLLM`, `llama-server`, etc.) in user-facing docs or
  doc prose — use Juno-native language instead (e.g. "GPU layer offload" not "matches llama `-ngl`").
  Exception: internal engineering docs under `docs/infra-plan/` and `docs/perf-compare/`
  (filenames/run metadata only).
- **No internal "Infra tier" numbers** (`Tier 5`, `Tiers 14-16`, etc.) in shipped docs, code
  comments, JFR `@Description`, CLI help, or error messages — those belong only in
  `docs/infra-plan/` planning docs (and `docs/lora-plan/`, which uses its own tier numbers).
- **Performance gates on hot-path changes**: when a change touches the forward pass, MatVec, GPU
  residency, batching, or KV paths, run `scripts/performance-tests/compare-lora.sh` (and
  `compare-vision.sh` if it touches vision/CLIP/vision MatVec) against the last published baseline,
  and publish results under `docs/perf-compare/<timestamp>-.../`. See
  `docs/infra-plan/PLAN-Infra-ROADMAP.md` execution rules for exact thresholds and when this is
  optional (API-only tiers with no MatVec/forward/KV/vision changes).
- New flags/features must not silently no-op on other surfaces that should support them (e.g.
  `--lora-play`, LoRA train, vision) — see `PLAN-Infra-ROADMAP.md` §6.
- **Git is read-only** unless the user's message explicitly names the write operation (e.g. "commit
  these files", "amend", "reset to origin"): no `git commit`, `git push`, or `git add` on your own
  initiative. `status`/`log`/`diff`/`show`/`branch` are always fine.
- Never add `Co-Authored-By: Claude`, `noreply@anthropic.com`, or any other Claude/Anthropic
  attribution trailer or footer to a commit message, PR description, or any file in this repo
  (`.java`, `.md`, etc.). Commit as the user only.
