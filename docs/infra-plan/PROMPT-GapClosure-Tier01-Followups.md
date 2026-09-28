# Agent prompt: gap-closure Tier 01 follow-ups (five items, in order)

Copy everything below the line into a new agent session.

---

## Task

You are a senior performance engineer (JVM, Panama/FFM, CUDA, LLM inference) working in
`/home/medion/Repo/juno`. Execute the five items below **in order**, one at a time, each to its own
exit condition, and report after each. The owner approved this order on 2026-09-27 after reviewing
the step-2 pass of the gap-closure plan's Tier 01. Items 1 to 4 are fixes the step-2 pass found
outside Tier 01's scope; item 5 is Tier 01's own next step.

1. **Diagnose** whether Qwen2/Qwen3 use the wrong RoPE pairing (perplexity A/B with a control).
2. **Fix `min_tokens`** so the model's own end signals cannot end a request below the minimum.
3. **Fix the Qwen RoPE pairing** — only if item 1 confirms it.
4. **Remove the redundant RoPE angle computation** on the CPU (bit-identical table), then take
   **one** re-baseline of the benchmark sweep.
5. **Tier 01 implementation step 3a**: wire the residency path (norm, QKV projection, RoPE) into
   `LlamaTransformerHandler`'s decode path behind an opt-in flag, then step 4's measurement.

Stop and ask the owner at every point this prompt says "ask", and whenever a finding changes scope
or sequencing. Do not improvise past a decision point.

## Hard rules (binding; `CLAUDE.md` wins on any conflict)

- Read `CLAUDE.md` first. Git is **read-only** unless the owner's message names the operation (no
  `commit`, `add`, `push`, `reset`, `amend`). `status`/`log`/`diff`/`show` are fine.
- **Never** add Claude/Anthropic attribution anywhere (commits, PR text, code, comments, docs).
- No competitor product names (`llama.cpp`, `vLLM`, `llama-server`, `llama-bench`, `ggml`) in
  production docs, code, comments, CLI help or error messages. Allowed only in
  `docs/gap-closure-plan/`, `docs/infra-plan/`, `docs/perf-compare/` (filenames/run metadata).
  Say "the reference implementation" or "the reference tool" elsewhere.
- No internal tier numbers ("Tier 5", "Infra tier") in code, javadoc, JFR descriptions, CLI help,
  error messages or production docs. No emojis anywhere.
- Tests first: write the test, watch it fail for the right reason, then implement. Prefer a new class
  over extending an existing one. Hot path: no allocation or boxing per token.
- New flags must not silently no-op on surfaces that should support them; fail closed with an
  explicit, once-logged notice. A new CLI flag is added to `compare-llama-cpp.sh`'s pass-through set
  in the same change (gap-closure README).
- Plan rules: `docs/gap-closure-plan/README.md` execution rules 1-9 are binding. Rule 9: every
  change outside Tier 01's scope (items 1-4 here) is recorded in
  `docs/gap-closure-plan/TIER-01-gpu-activation-residency.md` under **"Out-of-tier changes"**,
  naming what it touched, whether it is a measurement boundary, and which baselines it invalidates.
- **Never run a build (`mvn package/compile/install`) while a `mvn test` run is in progress** — it
  taints the in-flight module. Read Maven results from `BUILD SUCCESS` / surefire reports, never from
  the exit code of a pipeline. Run `mvn -o clean install -DskipTests` before any verification run:
  the IDE's Eclipse compiler has written broken class files into `target/` before.
- GPU memory checks (`memGetInfo`) are device-wide: run GPU tests and measurements on an idle device
  (`nvidia-smi --query-compute-apps=pid --format=csv,noheader` empty).
- If context runs low mid-item: stop, write a `docs/infra-plan/PROMPT-<short-name>.md` handoff in this
  same structure, and tell the owner.

## Environment

- Branch `67-gap-inference`. HEAD at hand-off: `cc94c53`, **plus uncommitted Tier 01 step-2 work**
  (listed below). Run `git status` / `git log --oneline -5` first; the owner may have committed it
  since. Never discard working-tree changes you did not make.
- GPU: 1x GTX 1080 (8 GiB, Pascal sm_61), CUDA 12.0, `nvcc` at `/usr/bin/nvcc`. PTX is compiled by
  hand and checked in: `nvcc -ptx -arch=compute_61 -O3 -o ../resources/cab/ml/juno/node/<name>.ptx
  <name>.cu` from `node/src/main/cuda/`. No ROCm hardware (mark ROCm rows `NEEDS-AMD-HARDWARE`).
- CPU: Xeon E5-1650 v2, 12 threads, AVX2, no AVX-512.
- Reference implementation source (for reading, not for shipping): `/home/medion/Repo/llama.cpp`
  (`src/llama-model.cpp`, converter package `conversion/`). Reference binaries used by the harness:
  `../llama.cpp/build-cuda/bin` (GPU), `../llama.cpp-bin/llama-b9551` (CPU). There is **no**
  perplexity binary in `build-cuda/bin`.
- Models (never read into context, never modify): `models/` holds, among others,
  `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`, `qwen2.5-3b-instruct-q4_k_m.gguf`, `Qwen3-1.7B-Q4_K_M.gguf`,
  `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`, `Phi-3.5-mini-instruct-Q4_K_M.gguf`,
  `mistral-7b-instruct-v0.1-q4_k_m.gguf`, `llama-1-30b.Q4_K_M.gguf`, `moondream2-q5_k.llamafile`.

## Commands and expected baselines

```bash
mvn -o clean install -DskipTests
mvn -o test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,vision,metrics,juno-player
#   at hand-off: 1731 tests, 0 failures, 0 errors, 46 skipped, about 23 min
#   (registry 93, lora 116, kvcache 78, health 23, node 642/41, tokenizer 109/2, sampler 81,
#    coordinator 322/1, vision 95, metrics 61, juno-player 111/2)
mvn -o clean verify -pl juno-master          # 20 tests: InProcess 6, ThreeNode 8, TensorParallel 5, UnsupportedArch 1
mvn -o -pl node test -Dtest='ClassName'      # single class; GPU tests run by default on this host

# Short, unpublished parity check (prompt 128/128, generated 64/64 must hold):
./scripts/performance-tests/compare-llama-cpp.sh --gpu --models tinyllama,qwen2.5-3b,Phi-3.5-mini,mistral-7b \
  --n-gen 64 --juno-reps 1 --juno-warmup 1 --reps 1 --no-publish --no-tuned-lane
#   results in target/perf-compare/<stamp>/; read <model>-compare.json: .generation_parity, .prompt_parity

# Full reference sweep (publishes under docs/perf-compare/<stamp>/):
./scripts/performance-tests/compare-llama-cpp.sh --gpu --reps 3 --juno-reps 3 --juno-warmup 2
./scripts/performance-tests/compare-llama-cpp.sh --cpu --reps 3 --juno-reps 3 --juno-warmup 2
./scripts/performance-tests/compare-lora.sh --reps 3      # gate: train >= 0.95x, playback tps >= 0.80x
./scripts/performance-tests/resident-chain-microbench.sh  # Tier 01 residency microbench
```

RoPE share of the forward pass, from a published sweep's JFR JSON:
`jq '.models[0].metrics | {rope: .["juno.Rope.decode.total_ms"], fwd: .["juno.ForwardPass.decode.total_ms"]}' docs/perf-compare/<stamp>/<model>-generate-rep2-juno-jfr.json`
(also `.prefill.total_ms` on the `-prefill-rep2-` files).

## Read first

1. `CLAUDE.md`; `docs/gap-closure-plan/README.md` (execution rules, Test infrastructure,
   Feature-complete definition).
2. `docs/gap-closure-plan/TIER-01-gpu-activation-residency.md` — especially the execution-record
   section **"2026-09-27 — implementation step 2"**, its **"Open for the owner"** list (items 1-4 there
   are this prompt's items 1-4 in a different order), and **"Out-of-tier changes"**.
3. `docs/perf-compare/20260927T025430Z-tier01-resident-chain/INDEX.md` (the chain measurement) and
   `docs/perf-compare/20260926T060301Z-tier01-rmsnorm-roundtrip/INDEX.md` (the round-trip baseline).
4. The code named in each item below.

## Already done (do not redo)

- **Tier 00** complete (2026-09-23). **Tier 01**: exit criteria 1-5 (measurement tooling, prompt
  parity, re-derived pp target, parity preconditions, pre-tokenizer parity) and, as of step 2, the
  **"Residency primitive implemented, unit-tested, and documented"** box are ticked. Implementation
  steps 1 and 2 are done; step 3 is item 5 here.
- **Step-2 work (uncommitted at hand-off):** `ResidentChain` (region: one stream, buffers, close frees
  all), `ResidentActivation` (host/device cross only at `upload`/`materialize`; upload-race guard via
  a sync epoch), `KernelParams` (preallocated launch block, `invokeExact`), `RopeKernel` +
  `node/src/main/cuda/rope.cu` + `rope.ptx` (adjacent pairs only; double angle/sincos, separately
  rounded float rotation; bit-identical to `LlamaTransformerHandler.rope`), `CudaRope`,
  `CudaRmsNorm.normalizeResident`, `RmsNormKernel.launchResident`, `ResidentChainMicrobench` +
  `scripts/performance-tests/resident-chain-microbench.sh`; tests `ResidentActivationTest` (8),
  `RopeKernelParityTest` (6), `ResidentChainMicrobenchTest` (7), one new `CudaRmsNormTest` case; docs
  (`docs/agent-arch.txt`, `docs/howto.md`, `docs/performance.md`, `docs/perf-compare/README.md`),
  `CHANGELOG.md` Session 94. None of it is referenced by a handler.
- **Chain measurement (GTX 1080, dim 2048):** resident chain 3.65x the CPU chain at decode, 15.86x at
  prefill; 0.51 / 0.55 of op-at-a-time. About 95% of the CPU chain is `rope()` recomputing
  `pow`/`cos`/`sin`, so the CPU ratio mostly measures CPU waste; the chaining ratio is the one that
  isolates residency.
- **Parity at HEAD verified:** all four sweep models (and tinyllama Q2_K) at 128/128 prompt tokens and
  64/64 generated tokens after `cc94c53` — its turn-marker change is recorded in Tier 01's
  Out-of-tier table as not a measurement boundary.

---

## Item 1 — Qwen RoPE pairing: perplexity A/B with a control (diagnosis; default behavior unchanged)

**Why.** RoPE pairs dimensions either adjacent `(x[2i], x[2i+1])` or split-half `(x[i], x[i+d/2])`
(the "NeoX"/`rotate_half` convention). The Q/K weight rows are laid out for one of them; the wrong one
does not crash but assigns learned frequencies to the wrong dimension pairs, degrading quality
silently, worse at longer context. Source evidence, verified on this host:

- Reference `src/llama-model.cpp`, `llama_model_rope_type`: `LLM_ARCH_LLAMA` -> `LLAMA_ROPE_TYPE_NORM`
  (adjacent); `LLM_ARCH_QWEN2`, `QWEN3`, `QWEN3MOE`, `PHI2`, `PHI3` -> `LLAMA_ROPE_TYPE_NEOX`.
- Reference converter: `conversion/llama.py` permutes Q/K rows (`LlamaModel.permute`, around lines
  100 and 141-143) so adjacent pairing is right for LLaMA; `conversion/qwen.py` (`Qwen2Model` ~52,
  `Qwen3Model` ~154, `Qwen3MoeModel` ~252) never permutes.
- Juno: `LlamaTransformerHandler.rope(float[] x, int pos, int nHeads, int headDim, float ropeTheta)`
  (static, adjacent) serves `qwen2`/`qwen2.5` (`LlamaFamilyArchitectures.VERIFIED`); `Qwen3Rope.apply`
  delegates to it (non-YaRN) or uses `applyAdjacentRotations` (YaRN); `Qwen3Rope.applyBackward` ->
  `LoraTrainingMath.ropeBackward` (adjacent). Juno uses split-half correctly for Phi-2/Phi-3
  (`Phi3Rope.applyNeoxRotations`, `Phi2Rope`) with a comment explaining why adjacent is wrong there.
  Nothing in Juno permutes Qwen Q/K rows at load.
- A greedy `qwen2.5-3b` run answers "The capital of France is Paris." — this does **not** settle it
  (short prompt, early positions). `Qwen2LiveForwardTest` only asserts the first token is not
  end-of-turn.

**What to build.** A pairing choice that defaults to today's behavior everywhere, plus a test-only way
to force the other pairing. Suggested shape: `enum RopePairing { ADJACENT, SPLIT_HALF }`, a
`rope(..., RopePairing)` overload (split-half body mirrors `Phi3Rope.applyNeoxRotations` without
frequency factors), a pairing field on `LlamaTransformerHandler` and on the Qwen3 path, set to
`ADJACENT` for every architecture in this item, and a package-private load overload that lets a test
override it. **No system property, no CLI flag** (a production knob that selects a known-wrong
geometry is a silent-degrade hazard). Rope call sites in `LlamaTransformerHandler`: single decode
(`transformerLayer`), batched decode (`transformerLayerBatch`), prefill window — grep `rope(`.

**The A/B (a GPU-free, model-gated test in `node`, e.g. `RopePairingPerplexityLiveTest`):**
- A fixed natural English text of about 600-1000 tokens in `node/src/test/resources/` (public domain;
  no competitor names), tokenized raw — no chat template, no BOS games beyond what the tokenizer does.
- Teacher-forced: for each position `t`, forward token `t` at position `t` (see
  `Qwen2LiveForwardTest` for the `ForwardRequest.withTokens(kvKey, tokens, pos)` pattern), take
  log-softmax of the logits, accumulate `-log p(token[t+1])`; perplexity = `exp(mean)`. CPU backend is
  fine (`LlamaTransformerHandler.load(path, ShardContext)`); budget a few minutes per run.
- Models: `qwen2.5-3b` and `Qwen3-1.7B` under both pairings; **control** `tinyllama` Q4_K_M (LLaMA,
  permuted weights) under both — adjacent **must** win there, or the method does not discriminate.
- Decision rule: confirmed if split-half is clearly lower on both Qwen models (state the margin; expect
  a large gap, report whatever it is) **and** adjacent is clearly lower on the control. Print all six
  perplexities.

**Exit condition.** Numbers recorded in Tier 01's execution record (new dated subsection) and as an
Out-of-tier row ("diagnosis only; default unchanged; not a measurement boundary"). Tests in `node`
pass. If the result is **inconclusive** (small margins, control fails): **ask** before item 3.

## Item 2 — `min_tokens` must hold against the model's own end signals

**What it is.** `min_tokens` (native `sampling.minTokens`, chat `min_tokens`) exists so the benchmark
generates exactly `n_gen` tokens, as the reference tool does. Published contract
(`api/src/main/resources/juno-api.yaml` ~561-578, `openapi.yaml` ~402): *"Only the end-of-sequence
token is held back: an explicit `stop` sequence still ends generation below the minimum, because a
caller who asked for a stop string asked for it unconditionally."*

**The gap.** `MinTokenFloor` (`sampler/.../MinTokenFloor.java`, ctor `(int eosTokenId, int
minTokens)`) masks only the EOS logit below the minimum. Commit `cc94c53` made generation also stop on
chat turn markers: by id (`GenerationLoop.resolveSamplingParams` ~630 merges
`tokenizer.chatTurnTokenIds()` into every request's `stopTokenIds` via
`OpenAiAdapter.mergeStopTokenIds`) and by text (`EosOutputFilter`, role headers such as `<|user|>`).
`Sampler.isStopToken` (~143) and both filters ignore the minimum. So a request ends below its
minimum on stops the caller never asked for (`<|endoftext|>`, `<|im_start|>`, `<|end|>`, text
headers). The harness help (`compare-llama-cpp.sh --juno-min-tokens`) promises the minimum holds
"before a stop token may end the request" — neither the contract nor the code says that. Not seen on
the benchmark prompt (all four sweep models 64/64), so it is latent.

**Fix.** Hold back the model's own end signals; keep the documented rule for the caller's.
- `MinTokenFloor` masks EOS **plus** the tokenizer's turn-marker ids, **minus** any id the caller
  explicitly asked to stop on. `resolveSamplingParams` merges both sets today, so keep the caller's
  set separately (compute it before merging). Call sites: `GenerationLoop` ~254 (static batch) and
  ~445 (single request), `ContinuousBatchEngine` ~195.
- Below the minimum, the text-level turn-marker stop in `EosOutputFilter` must not end the request;
  held-back text must then be emitted, not dropped. Check all three emit paths
  (`GenerationLoop.emitToken` ~685-711, the batch loop ~320-350, `ContinuousBatchEngine` ~301-350).
- Caller `stop` strings/ids still end below the minimum (unchanged, documented).
- Keep the grammar yield rule (if only ending tokens are legal, ending is allowed) for the widened set.
- Update both contract files and the harness help text in the same change.

**Tests first** (`sampler`: `MinTokenFloorTest`, `SamplerMinTokensTest`; `coordinator`:
`GenerationLoopEosPieceTest`, `EosOutputFilterTest`, a continuous-engine test): a turn-marker id is
masked below the minimum and released at it; a caller-requested stop id still ends below it; each of
the three generation paths produces N tokens when the model proposes a turn marker early; the
text-level header case. They must fail on current code for the right reason.

**Exit condition.** Full 11-module `mvn -o test` green; quick parity check still 64/64; Out-of-tier
row for `cc94c53` updated to say the gap is closed; `CHANGELOG.md` entry; not a measurement boundary.

## Item 3 — Qwen RoPE pairing fix (only if item 1 confirmed)

- Set `SPLIT_HALF` for `qwen2`/`qwen2.5` in `LlamaTransformerHandler` and for both `Qwen3Rope` paths
  (non-YaRN and YaRN) — this covers `qwen3` and `qwen3moe`. Grep every `rope(`/`Qwen3Rope` call in
  `node/src/main`, including batched and prefill paths.
- LoRA training must match or adapters train against the wrong geometry: `LoraTrainingMath.ropeBackward`
  (used by `Qwen3Rope.applyBackward`), `Qwen2LoraTrainableHandler`, and any Qwen path in
  `LoraTrainableHandler`. No Qwen adapter exists on disk, so no retraining is owed; state that.
- Vision text backbones (`llama`, `phi2`) are unaffected; confirm.
- The GPU `RopeKernel` is adjacent-only: the resident path (item 5) must stay off for Qwen until the
  kernel gains a split-half mode (or add the mode here, with a parity test against the CPU split-half).
- Keep item 1's A/B as a permanent regression test (perplexity ceilings per model).
- Generated tokens change for Qwen models (CHANGELOG note); throughput does not (same FLOPs) — confirm
  with the quick parity check. Out-of-tier row: correctness fix, not a throughput boundary.
- Full `mvn -o test` + `mvn -o clean verify -pl juno-master`.

## Item 4 — RoPE angles computed once, bit-identical; then one re-baseline

**Why.** `LlamaTransformerHandler.rope()` recomputes `Math.pow`, `Math.cos`, `Math.sin` per pair,
per head, per layer, per call; a position has only `headDim/2` distinct angles. On TinyLlama each
angle is evaluated 792 times per token (36 heads x 22 layers). Measured in the GPU reference sweep
`docs/perf-compare/20260925T172231Z/` (rep 2), where every matrix product is on the GPU and RoPE is
on the CPU: decode share of forward pass tinyllama 9.9%, mistral-7b 10.8%, qwen2.5-3b 8.0%; 128-token
prefill share tinyllama 23.5%, mistral-7b 23.4%. Phi-3.5 emits no `juno.Rope` span (unmeasured) and
`Phi3Rope` builds its table once per call (far less waste). Estimated effect: roughly +10% tg and
+25-30% pp on Llama-family models — an estimate; measure it.

**Build.** New class `RopeTable` (`node`): per `(headDim, ropeTheta)`, cos/sin per position, values
computed with **exactly** today's expressions —
`double freq = 1.0 / Math.pow(ropeTheta, (2.0 * i) / headDim); double angle = pos * freq;`
`float cosA = (float) Math.cos(angle); float sinA = (float) Math.sin(angle);` — so results are
bit-identical. Lazily filled blocks of positions up to `MAX_SEQ_LEN` (32768), published immutably
(thread-safe, no locks on the read path), shared by all handlers with the same parameters (about 8 MB
at head size 64 if fully filled). `rope()` keeps its signature (and item 1's pairing overload) and
reads the table. Optional, only if measured to matter: `Qwen3Rope`'s YaRN cache and
`LoraTrainingMath.ropeBackward`.

**Tests first.** Bit-identity against a verbatim copy of the old implementation for random positions
in `[0, 32767]`, head sizes 64/96/128, bases 10000/500000/1e6; concurrent readers; no per-call
allocation.

**Perf gate (required — forward-pass change) and the single re-baseline.**
- `compare-lora.sh --reps 3` against the last published LoRA baseline
  (`docs/perf-compare/20260924T201548Z-lora/`): train >= 0.95x, wall-clock playback tps >= 0.80x.
- `compare-llama-cpp.sh --gpu` and `--cpu` full sweeps (`--reps 3 --juno-reps 3 --juno-warmup 2`),
  published. This **is a measurement boundary** (README rule 9): label the new sweeps the reference,
  mark `20260925T172231Z` (GPU) and `20260925T174146Z` (CPU) superseded-as-reference (keep them),
  record the new RoPE shares, and update the README "Program target" table with a new column rather
  than overwriting the old one. Re-run any row whose repetitions spread more than 15% of their median.
- Re-run `resident-chain-microbench.sh`: the CPU chain gets much cheaper, so expect the decode-width
  "resident chain vs CPU" to fall to roughly 0.3x (estimate) while the chaining ratio holds. Tier 01's
  contingency classes a decode-width miss with a prefill-width pass as a **partial result handed to
  Tier 01B item 2, not a downgrade** — record it that way and tell the owner.
- Docs: `docs/performance.md`, `docs/perf-compare/README.md`, `CHANGELOG.md`, Tier 01 record.

## Item 5 — Tier 01 implementation step 3a (then step 4)

**Why "3a" and not "norm + RoPE only".** Today's GPU decode layer (MMQ projections, GPU attention on)
does, per layer per token: CPU norm; QKV round trip (`matVecProjectionSameX` ->
`backend.sgemvSameX(DeviceQ4KMatrix[]...)`: upload, 3 kernels, download, sync); CPU RoPE; KV mirror
append (`DeviceKvCache.appendToken`: FP16 conversion on the host, 2 synchronous copies); attention
(`CudaGqaAttention.attendBatched`: 4 synchronous uploads, launch, 1 download); O, gate/up, down each a
round trip; CPU residual, norm, SwiGLU — about a dozen synchronizing transfers per layer. Wiring only
norm and RoPE through the chain **adds** round trips (download the norm for the host projection,
upload Q/K for RoPE, download again), because `MatVec` is host-in/host-out.

**3a: one region = norm -> QKV projection -> RoPE, one download of q/k/v.**
```
upload x -> GPU norm (resident) -> QKV MMQ kernels read/write device memory -> GPU RoPE -> materialize q,k,v
```
Same number of synchronizations as today's QKV round trip, with norm and RoPE moved onto the GPU.
Needs:
- A device-in/device-out projection entry point for the K-quant MMQ path. `CudaMatVec.sgemv(
  DeviceQ4KMatrix, float[])` already runs `Q4KMmqKernel.launch(A, dX, dQ8, dY, stream)` (line ~194)
  between its own upload and download. Add a resident variant taking `ResidentActivation` in/out,
  launching on the **chain's** stream with a **chain-owned** Q8 scratch (the per-thread
  `FP32_SCRATCH.dQ8` could be overwritten from the other stream mid-kernel). MMQ is CUDA-only, so a
  CUDA-only method is acceptable under `CLAUDE.md`'s vendor-neutral rule; keep `GpuBindings` for
  buffers/streams.
- One `ResidentChain` per thread (`ThreadLocal`); local mode runs 3 in-process nodes by default.
- Norm weights uploaded once at load as `1 x dim` `DeviceFloatMatrix` (`attnNorm[li]`); one `CudaRope`
  per handler (`cfg.headDim()`, `cfg.ropeTheta()`).
- Opt-in flag mirroring `GpuAttentionOptions` (`ENV_PROPERTY = "JUNO_GPU_ATTENTION"`, system property
  or env): e.g. `--gpu-residency on|off|auto` / `JUNO_GPU_RESIDENCY`, **default off** until step 4
  measures it; CLI help in `ConsoleMain`, `docs/howto.md`, pass-through in `compare-llama-cpp.sh`.
- Fail-closed/announced-once fallbacks to today's path: any non-MMQ projection; `--lora-play`
  (`applyLoraInPlace` works on host q/k/v); `qwen2`/`qwen3` until the GPU kernel has split-half
  (item 3); QKV biases (`bq != null`) unless a bias-add kernel is added; `--parallel` multi-decode
  (rows at different positions — the kernel takes `startPos + row` only) unless per-row positions
  are added. ROCm: `CudaRope.tryCreate` returns null — announce, `NEEDS-AMD-HARDWARE`.
- **3b (KV append and attention inside the region, download only the attention output) is the bigger
  decode win but the plan assigns attention residency to Tier 02. Ask before taking it into Tier 01.**

**Tests (from Tier 01's list):** greedy decode identical (CPU path unaffected) or within tolerance
(GPU) with the flag on vs off — `LlamaTransformerHandlerVerifyParityTest` or a sibling;
`ModelLiveRunnerIT` GPU check with the flag on (tinyllama, mistral-7b);
`scripts/performance-tests/smoke-tier01-gpu-residency.sh` (on/off diff on tinyllama, mistral-7b,
llama-1-30b; no VRAM leak across repeated requests via `memGetInfo` on an idle device); cluster
pipeline/tensor smoke (activations materialize at the gRPC/AllReduce boundary); LoRA train + playback
unaffected.

**Step 4 (measurement), thresholds from Tier 01's Threshold block:** re-run the microbench and the
end-to-end gates on the wired path; tg within 0.95x of the (item 4) baseline on every sweep model with
the flag on; `compare-lora.sh` train >= 0.95x, playback >= 0.80x; device memory back to its
pre-request level per request. Read the contingency honestly (decode miss + prefill pass = partial
result). Then update Tier 01's remaining exit boxes and `Status:` line; `CudaGraphSession` stays
explained-not-wired unless measured worthwhile.

## Known, recorded, not in this order (do not fix unless asked; Tier 14 owns them)

- `docs/performance.md`: 24 tier-number hits (incl. bare "Tier 15", "Tier 5"); `docs/agent-arch.txt`:
  2 `PLAN-Infra-TierN.md` pointers left (~lines 105, 110).
- Ten competitor-name mentions in `node` code comments (`GgufKQuantCodec`, `Qwen3Rope`,
  `Phi3RopeConfig`, `LlamaConfig` x2, `GgufReader` x3, `Phi2Rope`, `Phi3Rope`).
- Emoji glyphs in CLI output: `scripts/run.sh` (`ok`/`warn`), `scripts/aws/launcher.sh`,
  `ConsoleMain` ~1155.
- `scripts/performance-tests/check-plan-thresholds.sh` does not exist yet — Tier 01B's first deliverable.

## Exit checklist (report after each item, and a final standalone report)

- [ ] Item 1: six perplexities recorded; confirmed / refuted / inconclusive stated; default unchanged.
- [ ] Item 2: tests failed first for the right reason; fix on all three generation paths; contracts,
      harness help, `CHANGELOG.md` updated; Out-of-tier row updated; full suite green.
- [ ] Item 3 (if confirmed): Qwen2/Qwen3 (incl. YaRN, LoRA backward) on split-half; regression test;
      parity check; `CHANGELOG.md`; Out-of-tier row.
- [ ] Item 4: bit-identity tests; RoPE share before/after; `compare-lora.sh` and both
      `compare-llama-cpp.sh` sweeps published; new reference labeled, old superseded, README target
      table extended; microbench re-read against the contingency; Out-of-tier row (measurement boundary).
- [ ] Item 5: step 3a behind a default-off flag with every fallback announced and tested; parity,
      live, smoke, cluster and LoRA checks; step 4 gates published; Tier 01 boxes and `Status:` honest.
- [ ] Every item: `mvn -o test` for touched modules (full 11-module run at least at the end),
      `mvn -o clean verify -pl juno-master`, real output shown; no commit; uncommitted files listed.
- [ ] Final report: items done, drift found, files changed, commands and pass/fail, exit boxes ticked
      and remaining, decisions needed from the owner, exact next step.
