# Agent prompt: Tier 13 Phase C — Stage 4 (bake-off + docs)

Copy everything below the line into a new agent session.

---

## Task

Finish **Stage 4 of 4** of the GPU-resident attention feature (Tier 13 Phase C, `--gpu-attention`).
Stages 1–3 (device-resident KV mirror, the real CUDA kernel, and wiring it into
`LlamaTransformerHandler`) are **done and fully verified** — this prompt is scoped to what's left:
**CLI flag wiring, the bake-off, and documentation.** Do not redesign or re-verify Stages 1–3; they
are correct (see "What's already done" below) and their tests already pass. Do not start Stage 5 or
any new Infra tier.

**Nothing from this feature is committed yet** — `git status` shows everything below as modified or
untracked in the working tree, on branch `67-inference`. Treat it as your own in-flight work to
finish, not something to review from scratch.

## Why this exists

JFR instrumentation (a `juno.Attention` event added earlier in this same effort) proved attention —
not the GEMM/MatVec path — is the dominant cost at realistic context length: **78.3% of prefill wall
time** and **64.2% of decode wall time** at ~512 tokens context (TinyLlama, GTX 1080; see
[`docs/perf-compare/20260916T003101Z-prefill`](../perf-compare/20260916T003101Z-prefill/) and
`docs/performance.md`'s "Prefill microbatching" section). Attention ran entirely on scalar CPU Java
even when the model's weights were fully GPU-resident. This feature moves attention onto the GPU. The
whole point of Stage 4 is to **measure and publish** whether it actually closed that gap, and to
document the feature so it's discoverable and its known limitations are honest.

## What's already done (Stages 1–3 — verified, do not redo)

**Design decision already made and confirmed with the user:** v1 is a straightforward parallel
kernel (materialize the score row, softmax, weighted-V-sum — parallelized across GPU threads), **not**
a tiled/online-softmax FlashAttention-2 kernel. That fuller design is an explicit future follow-on,
out of scope here.

**Scope already fixed:** `LlamaTransformerHandler` only (covers `llama`/`mistral`/`qwen2` GGUF
architectures). CUDA only. `Phi2TransformerHandler`/`Phi3TransformerHandler`/`Qwen3TransformerHandler`/
`Qwen3MoeTransformerHandler`, ROCm, and `--lora-play`/LoRA train are all **named follow-ups** (each has
its own copy of the attention math / its own KV map — see `LoraTrainableHandler`), matching exactly
how Tier 13B's MMQ rollout handled the same handlers. Vision is wired automatically (no vision-specific
code) since `VisionAwareForwardPassHandler` delegates every forward call to an internal
`LlamaTransformerHandler`.

**New files (all compiling, all tests green):**

| File | Purpose |
|------|---------|
| `node/src/main/java/cab/ml/juno/node/GpuAttentionOptions.java` | `on\|off\|auto` policy, env var `JUNO_GPU_ATTENTION` (clone of `MmqOptions`) — **no CLI flag yet, this is Stage 4's first job** |
| `node/src/main/java/cab/ml/juno/node/GqaMath.java` | Attention math extracted from `LlamaTransformerHandler.gqaInto`/`gqa` (zero behavior change — both delegate to it now); also the parity-test CPU oracle |
| `node/src/main/java/cab/ml/juno/node/DeviceKvCache.java` | Device-resident FP16 KV mirror: dual-write alongside the host `SessionKvTensor` (unchanged), grow-and-preserve (D2D copy), lifecycle tied to `LlamaTransformerHandler.evict()`. Has a numerical note in its javadoc (see "Known limitation" below) |
| `node/src/main/cuda/gqa_attention.cu` → `node/src/main/resources/cab/ml/juno/node/gqa_attention.ptx` | The real kernel: one block per (batch-row, head), 3-pass (QK^T+max, softmax, weighted-V-sum). Compile with `nvcc -ptx -arch=compute_61 -O3 -o ../resources/cab/ml/juno/node/gqa_attention.ptx gqa_attention.cu` from `node/src/main/cuda/` if you ever touch the `.cu` — the `.ptx` is checked in, not built by Maven |
| `node/src/main/java/cab/ml/juno/node/GqaAttentionKernel.java` | PTX loader/launcher via CUDA Driver API (mirrors `Q4KMmqKernel`) |
| `node/src/main/java/cab/ml/juno/node/CudaGqaAttention.java` | Handler-facing `attendBatched(...)` — real dispatch, grow-and-keep-max scratch, batched-pointer design (one launch serves prefill window / single decode / `--parallel` multi-decode) |
| `node/src/test/java/cab/ml/juno/node/{GpuAttentionOptionsTest,DeviceKvCacheLifecycleTest,GqaAttentionKernelParityTest,LlamaTransformerHandlerGpuAttentionLiveTest}.java` | All green; the live test loads a real GGUF + CUDA and proves `gpuAttentionActive()`, device-byte allocate/free correctness, and greedy-token parity |

**Modified files:** `LlamaTransformerHandler.java` (new `gqaGpu`/`kvCacheDev` fields, `deviceLayersFor`/
`layerGpuResident` helpers, `evict()` extended, all three attention call sites — prefill batch, single
decode, multi-decode — now build a `B`-sized batch and call `attendBatched`, falling back to `GqaMath`
if it returns `false`); `GpuBindings.java`/`CudaBindings.java` (`D2D` memcpy-kind constant added).

**Test status (must stay this way):** full non-GPU `node` suite 522/522; full GPU-tagged suite
(`mvn test -Dgroups=gpu -pl node`) 91/91. Flag defaults **off** — behavior is unchanged unless a user
opts in.

**Known limitation, already documented in `DeviceKvCache`'s javadoc — do not "fix", it's expected:**
multi-token greedy-decode sequences can occasionally diverge between `--gpu-attention on` and `off`
on some prompts (verified: single-step logits match tightly via the parity test; three independent
harnesses — single-shard decode, single-shard batched-prefill+decode, and the real 3-node pipeline
shape — showed byte-identical output for a fixed prompt; only a real chat-templated prompt run through
the interactive CLI showed divergence after ~15+ tokens). This is FP16 KV rounding occasionally
flipping a close greedy decision, the same class of behavior already accepted for `--mmq` and other
reduced-precision paths in this codebase, none of which guarantee multi-step token-sequence identity.
**Do not spend time trying to make on/off bit-identical across full generations — that is not the bar
here or anywhere else in this codebase.**

## Read first (mandatory)

1. [`models/CLAUDE.md`](../../models/CLAUDE.md) — tests first, KISS, prefer new classes, list changed
   files (no zip)
2. [`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) — Execution rules **§1–§6**, especially §2 (perf
   compare at tier completion — exact steps to run) and §6 (feature × surface interaction matrix
   template, "no silent flag ignore")
3. [`PLAN-Infra-Tier13.md`](PLAN-Infra-Tier13.md) — Phase A/B history; **Phase C has no section yet —
   add one**, following the same structure as the existing Phase A/B sections
4. `docs/performance.md` — read the "Prefill microbatching" section's Tier 17 subsection in full; it
   already documents the 78.3%/64.2% attention-share finding that motivated this feature and is where
   the bake-off numbers below should be added as a follow-on
5. [`docs/perf-compare/README.md`](../perf-compare/README.md) — publish layout; see how the Tier 17
   GPU batched-prefill-GEMM bake-off section is written (mirrors what you're adding for Phase C)
6. `MmqOptions.java` / `GpuAttentionOptions.java` side by side — the CLI/env pattern to replicate
   exactly for the new flag's `ConsoleMain`/`run.sh`/`compare-llama-cpp.sh` wiring

## What Stage 4 actually needs to do

### 1. Wire `--gpu-attention` as a real CLI flag (not just an env var)

Mirror `--mmq` exactly — every line below has a direct `--mmq` analog to copy the shape of:

- `juno-player/src/main/java/cab/ml/juno/player/ConsoleMain.java`:
  - a `private static String gpuAttention = null;` field (mirrors `mmq` field, line 220)
  - in the arg-parsing switch, a `case "--gpu-attention":` mirroring `case "--mmq":` (line 687)
  - the "apply as system property" line mirroring line 373-374
  - the "fall back to env var" block mirroring lines 476-479
  - a help-text line mirroring line 936 (state default off, CUDA-only, and that it's a measured
    decode/prefill throughput lever like `--mmq`, not a peer-latency claim)
- `scripts/run.sh`: `grep -n "mmq\|MMQ" scripts/run.sh` to find every occurrence (the local-var read
  defaulting from `JUNO_MMQ`, the `--mmq)` flag-parsing case, a help-text line, and the
  `mmq_arg`/pass-through-to-java construction near the final `exec "$JAVA"` call) and replicate each
  one's shape for `gpu_attention`/`JUNO_GPU_ATTENTION`.
- `scripts/performance-tests/compare-llama-cpp.sh`: **required by ROADMAP §2** ("when a tier ships a
  new CLI/env flag, add that flag to `compare-llama-cpp.sh`'s own pass-through option set... in the
  same change"). `grep -n "mmq\|MMQ" scripts/performance-tests/compare-llama-cpp.sh` to find every
  occurrence (the `JUNO_MMQ=""` default, the `--mmq on|off|auto` help line, the parse case, the
  `"juno_mmq":` JSON field, and the `if [[ -n "$JUNO_MMQ" ]]; then java_args+=(--mmq "$JUNO_MMQ"); fi`
  passthrough) and mirror each one.

After this, `--gpu-attention on|off|auto` must work identically whether set via CLI flag or
`JUNO_GPU_ATTENTION` env var (same precedence as `--mmq`: CLI flag wins if given, else env var, else
default off).

### 2. Run the bake-off (ROADMAP §2 steps)

```bash
# The specific repro that found and motivated this whole feature — run this first,
# it's the number that actually answers "did this work":
./scripts/performance-tests/compare-prefill-batch.sh --gpu --n-prompt 512 --prefill-values 1,32
# (once wired) re-run with --gpu-attention on via whatever flag compare-prefill-batch.sh
# exposes, or JUNO_GPU_ATTENTION=on prefixed — check that script's own flag surface first,
# it may need the same passthrough treatment as compare-llama-cpp.sh if it launches Juno directly

# Standing regression gate — run both with the flag off (must be unchanged) and on:
./scripts/performance-tests/compare-llama-cpp.sh --gpu --vector 0
JUNO_GPU_ATTENTION=on ./scripts/performance-tests/compare-llama-cpp.sh --gpu --vector 0
# or --gpu-attention on once CLI-wired, per whichever compare-llama-cpp.sh ends up exposing

# LoRA regression-only gate (this feature is an explicit no-op for LoRA — confirm flat):
./scripts/performance-tests/compare-lora.sh --gpu --baseline release-0.1.2
```

Pull `juno.Attention.prefill.total_ms` / `juno.MatVec.duration.total_ms` / `ForwardPass.prefill.total_ms`
from the JFR metrics in the `on` run and compare the attention-share percentage against the 78.3%/64.2%
baseline this feature set out to fix — that delta is the tier's real, honest speed claim. Do **not**
claim a peer-latency win without a number; state whatever the measured ratio actually is, same honesty
standard as the `--mmq` docs ("measured decode-throughput win... not peer latency").

Publish results under `docs/perf-compare/<timestamp>/` (+ `-lora/` for the LoRA regression run).
Vision compare (`compare-vision.sh`) is **not required** — this feature doesn't touch Phi-2/vision
paths at all (wired automatically via delegation, no MatVec/vision-specific code changed).

### 3. Fill in the Tier 13 Phase C interaction matrix (ROADMAP §6, required before "feature complete")

Add a "Phase C" section to `PLAN-Infra-Tier13.md` (after the existing Phase A/B sections) with this
table filled in (values already determined by the design, just need writing up in the doc's required
format):

| Surface | Cell |
|---|---|
| Base inference | **wired** (Llama-family/Mistral/Qwen2); Phi-2/Phi-3/Qwen3/Qwen3-MoE **follow-up** |
| `--lora-play` | **explicit no-op + warn** (separate handler class, own KV map/attention math) |
| LoRA train | **explicit no-op + warn** |
| Vision | **wired automatically** (delegates to internal `LlamaTransformerHandler`) |
| `--parallel` | **wired** (per-stream device pointers, one batched launch) |
| `--gpu-layers` | **wired** — activates only for GPU-resident layers, scalar fallback below cutover |
| `--prefill-batch` | **wired** |
| `--mmq` | **wired**, orthogonal — attention only ever consumes the host `float[]` Q/K/V regardless of which device dtype produced it |
| CUDA | **wired** |
| ROCm | **follow-up** — no kernel, `CudaGqaAttention.tryCreate` returns `null` on non-CUDA backends, falls back to scalar CPU, never silent (check `LlamaTransformerHandler`'s constructor log line for the warning path) |
| Default | **off** |

Check the "explicit no-op + warn" cells against §6's actual requirement ("startup warning... says so"):
confirm `--lora-play`/LoRA train genuinely warn if `--gpu-attention on` is passed and silently
ignored today (it's a different handler class that never reads `GpuAttentionOptions` at all — verify
whether that counts as "explicit no-op" already or needs an actual warning added; if the latter, add
a minimal one-line warning in `LoraTrainableHandler`'s constructor, do not build out the full feature
there).

### 4. Update docs

- `docs/performance.md`: add a new subsection (near the existing "Prefill microbatching" / Tier 17
  section, since it directly follows up on that section's attention-share finding) with the bake-off
  numbers from step 2, the `--gpu-attention` flag description, and the known FP16-divergence
  characteristic (one sentence, matching the tone already used for `--mmq`'s honesty notes).
- `docs/agent-arch.txt`: add `GpuAttentionOptions`/`CudaGqaAttention`/`DeviceKvCache`/
  `GqaAttentionKernel` to the `node` module's class map, near the existing `MmqOptions`/
  `GpuLayerOffload` entries.
- `docs/howto.md`: add `--gpu-attention on|off|auto` to the CLI flags table, alongside `--mmq`/
  `--gpu-layers`.
- `README.md`: one line if it lists GPU features at that level of detail (check current structure
  first — don't add a section that doesn't match the file's existing granularity).
- `docs/infra-plan/PLAN-Infra-ROADMAP.md`: update the Tier 13 row in the "Feature catalog" table to
  mention Phase C is feature complete (or note what's still open, if the bake-off doesn't fully close
  the gap — be honest either way, matching how the Phi-3.5 0.5× gate is reported as open rather than
  fudged).

Remember: **no Infra tier numbers or competitor product names** in any of these four files except
inside `docs/infra-plan/`/`docs/perf-compare/` — describe the feature by flag name and behavior
("GPU-resident attention", "`--gpu-attention`"), never "Tier 13 Phase C" in user-facing prose.

## Design constraints (carried over, still apply)

- Only one Infra tier in flight — this is still Tier 13 Phase C, do not start a new tier.
- KISS; prefer new classes over extending; keep the flag default **off** until the bake-off actually
  justifies a claim.
- No silent flag ignore anywhere `--gpu-attention` could reach (ROADMAP §6) — every surface in the
  interaction matrix above must warn or fail closed if it can't honor the flag.
- All supported models (§5): Phi-2/Phi-3/Qwen3/Qwen3-MoE staying on the correct (untouched, still
  correct) scalar path is fine as a documented follow-up — just don't claim the flag works there.

## Exit when

1. `--gpu-attention on|off|auto` works identically via CLI flag and env var, in `ConsoleMain`,
   `run.sh`, and `compare-llama-cpp.sh`.
2. Bake-off published under `docs/perf-compare/<timestamp>/` (+ `-lora/`) with an honest attention-
   share-before-vs-after number, not just a raw tg/pp ratio.
3. `PLAN-Infra-Tier13.md` has a Phase C section with the interaction matrix complete (no empty cells)
   and an exit checklist matching the ROADMAP template.
4. `docs/performance.md`, `docs/agent-arch.txt`, `docs/howto.md`, `README.md`, and the ROADMAP feature
   catalog are updated to match measured reality — no unearned peer-latency claims.
5. Full `node` suite still green (`mvn test -pl node` and `mvn test -Dgroups=gpu -pl node`) after any
   code touched in this stage (the CLI wiring is the only code change expected — Stages 1–3's Java is
   already done and should not need edits).

## Preview

List changed/added files for preview; never zip.
