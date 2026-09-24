# Tier 02: Attention & long context

Status: not started
Gap analysis refs: §1.2

## Objective

Close the three attention/long-context gaps: no tiled/online-softmax attention kernel (full QK^T
materialization instead), no sliding-window attention, and a hard failure (rather than
context-shifting) once a sequence exceeds `MAX_SEQ_LEN`. By the end of this tier, Juno should
handle long-running conversations gracefully (shift instead of hard-fail), support
sliding-window-attention model families correctly, and have an attention kernel that doesn't pay
O(seq²) memory for long contexts.

## Why this tier, why now

This tier depends on Tier 01's activation-residency primitive: a tiled/online-softmax attention
kernel is exactly the kind of multi-step, GPU-resident-intermediate-state computation that pays off
once activations don't round-trip to host between steps. Doing this before Tier 01 would repeat the
same "measured regression, shelved as scaffolding" pattern the gap analysis flags in §2.8. It's
sequenced before KV cache work (Tier 03) because context-shifting and sliding-window attention are
policy decisions about *what* the KV cache holds, which the Tier 03 paged-KV redesign needs to know
about before it locks in a block-table layout.

## Scope

### In scope

1. **Tiled/online-softmax attention kernel** (CUDA), replacing `gqa_attention.cu`'s current
   full-materialization design, built on Tier 01's residency primitive. Must remain numerically
   correct (verified against the existing scalar CPU attention path) while reducing peak memory at
   long sequence lengths. **It inherits whatever architecture coverage Tier 01B actually delivered —
   read that tier's exit table, do not assume it covered everything.** Tier 01B ships the kernel path
   for Phi-2, Phi-3, Qwen3 and Qwen3-MoE but explicitly leaves any architecture it could not measure
   on a real model with its default resolved off behind an explicit notice, and at the time of
   writing no `qwen3` or `qwen3moe` file exists on disk to measure with. So the starting state for
   this tier is "default-on for the architectures 01B measured," not "default-on everywhere."
   Whatever that set turns out to be, this rewrite must not shrink it — no architecture may regress
   to the scalar path as a side effect — and must re-run Tier 01B's per-architecture greedy-decode
   divergence characterisation for every member of it, since a tiled online-softmax accumulates in a
   different order and its divergence profile is not the one Tier 01B measured.
2. **Context-shifting**: when a session's KV would exceed `MAX_SEQ_LEN`, instead of throwing
   `IllegalStateException` in `DenseKvTensor`/`KvPageTable`/`PagedKvTensor`/`DeviceKvCache`, support
   an explicit, opt-in shift policy (drop oldest N non-system tokens, keep going) — opt-in because
   silently dropping context is itself a form of silent degrade the project's fail-closed philosophy
   would otherwise reject; the default behavior stays a clear error unless the caller opts in via a
   flag/request parameter.
3. **Sliding-window attention** for model families that need it (Mistral-style windowed causal
   mask) — add the windowing option to the attention path and confirm it's applied only when the
   loaded GGUF's metadata specifies a window size, not globally.

### Out of scope

- Extending the tiled attention kernel to ROCm (tracked in Tier 10, same reasoning as Tier 01).
- Vision attention (`VisionEncoder`'s own CLIP-style attention) — vision has very different batch
  shapes (~741-wide) and is handled in Tier 11.
- Automatic (non-opt-in) context management heuristics — this tier ships the mechanism and an
  explicit opt-in flag, not a policy for when to use it automatically.

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | scalar attention stays the correctness oracle; sliding-window and context-shift logic must work identically CPU and GPU since they're KV/scheduling-level, not kernel-level |
| 2 | CUDA GPU inference | tiled kernel is CUDA-only this tier |
| 3 | ROCm GPU inference | FAIL-CLOSED or NEEDS-AMD-HARDWARE for the tiled kernel; sliding-window/context-shift (KV-level, not kernel-level) should work on ROCm's existing GEMV path without needing the new kernel |
| 4 | Static schedule | context-shift must interact correctly with `BatchConfig`'s "all requests start at step 0" constraint — a mid-batch shift changes sequence length for one member only |
| 5 | Continuous schedule | context-shift interacting with `ContinuousBatchEngine`'s per-slot state needs its own test — a slot that shifts mid-stream must not corrupt `ContinuousPrefillState` bookkeeping for other slots |
| 6 | Single-node local mode | primary dev surface |
| 7 | Pipeline-parallel cluster | sliding-window metadata (window size) must propagate to every node holding a shard of a windowed model |
| 8 | Tensor-parallel cluster | same |
| 9 | LoRA training | confirm training's own attention/backward path (`LoraTrainingMath`) is unaffected — training doesn't currently use the fused GQA kernel |
| 10 | LoRA playback | confirm LoRA-modified attention projections still compose correctly with the new tiled kernel |
| 11 | Vision | N/A — out of scope, must verify vision's separate `VisionEncoder` attention path is untouched |
| 12 | OpenAI REST surface | context-shift needs an explicit request-level opt-in (e.g. an `x_juno_*` extension field, following the project's existing `x_juno_grammar`/`x_juno_loras` naming convention) |
| 13 | Native REST surface | same opt-in surfaced there too |
| 14 | CLI | a `--context-shift` (or similarly named) flag for `local`/`cluster`, off by default |

## Implementation steps

1. Write correctness tests for the existing scalar attention path as the oracle (if not already
   fully covered) before touching the kernel.
2. Design and implement the tiled/online-softmax kernel using Tier 01's residency primitive;
   validate numerically against the oracle at multiple sequence lengths, including lengths that
   would have overflowed the old kernel's scratch buffer.
3. Implement context-shift as an opt-in KV-cache operation (works for both dense and paged KV);
   wire the opt-in flag through CLI and both REST surfaces.
4. Implement sliding-window attention, driven by GGUF metadata. No window key is read anywhere
   today — `GgufReader` has no `sliding`/`window` reference at all — so this tier adds the read via
   the existing generic `metaInt` accessor and must first confirm which key real exporters write
   (`<arch>.attention.sliding_window` is the expected spelling; verify against a real file rather
   than assuming).

   **Validation does not depend on Tier 08, and must not be deferred to it.** An earlier draft
   deferred the windowed-model check until "Tier 08 makes that model loadable," which created a
   cycle: Tier 08's Gemma handler needs this tier's windowing mechanism, and Tier 00's audit already
   established that `gemma-4-E4B` uses patterned sliding-window attention with a 512 window. Break it
   by splitting the validation:
   - **This tier** validates the mechanism against a synthetic fixture — a GGUF whose metadata
     declares a window — plus a unit-level test that the windowed causal mask ignores exactly the
     tokens outside the window and that a model *without* the key is bit-identical to pre-tier
     behaviour. That is sufficient to prove the mechanism and to ship it.
   - **Tier 08** carries the end-to-end validation on `gemma-4-E4B` as one of its own exit criteria,
     once its handler makes that file loadable. Tier 08's file states this explicitly so the
     obligation is tracked rather than lost between the two tiers.

   Also check whether `mistral-7b-instruct-v0.1` (already present) declares a window before
   requesting any new download.
5. Run the full cross-surface smoke matrix, including a long-context stress case (loop conversation
   turns until the shift boundary is hit) for both static and continuous schedules.

## Tests to write/upgrade before implementation

- **New unit tests** for the tiled kernel: exact-match (within float tolerance) against the scalar
  CPU oracle at short, medium, and long (near/at `MAX_SEQ_LEN`) sequence lengths.
- **New unit tests** for context-shift: `DenseKvTensor`/`KvPageTable`/`PagedKvTensor`/
  `DeviceKvCache` each get a test that grows a session past `MAX_SEQ_LEN` with the opt-in flag set
  and confirms it shifts instead of throwing, and a test confirming the *default* (flag unset)
  behavior still throws (no silent regression of the existing fail-closed guarantee).
- **New unit test** for sliding-window: confirm a windowed model's attention correctly ignores
  tokens outside the window, and a non-windowed model's behavior is bit-identical to before this
  tier (regression guard).
- **`ModelLiveRunnerIT`**: add a long-context check (generate past the old hard-fail point with
  `--context-shift` enabled). The real-model windowed check belongs to Tier 08 and is listed in that
  tier's exit criteria; this tier's windowed coverage is the synthetic fixture plus the unit-level
  mask tests above, which is enough to ship the mechanism without waiting on a later tier.
- **New bash smoke script**: `scripts/performance-tests/smoke-tier02-attention-context.sh` —
  drives a long multi-turn conversation via the REST API until the shift boundary, asserts the
  server keeps responding instead of erroring, and asserts a second run *without* the opt-in flag
  still gets the documented hard error at the same point.
- **Perf gate (required)**: the tiled kernel is a hot-path change — `compare-lora.sh` plus a
  dedicated long-context latency/memory microbenchmark, plus `compare-llama-cpp.sh` for a
  llama.cpp-relative pp/tg reading on the same models (per README's llama.cpp-relative gate); publish
  under `docs/perf-compare/`.

  **Threshold.** Measure the old full-materialization kernel's peak attention scratch first, then
  hold the tiled kernel to both of these:
  - peak attention scratch at the longest tested sequence length drops by **>= 80%**;
  - peak attention scratch scales sub-quadratically — doubling the sequence length must **less than
    double** it, which is the property the rewrite exists to buy and the one a percentage alone does
    not capture.

  Throughput must not regress: tg ratio within 0.95x and pp ratio within 0.95x of the pre-tier
  baseline for every sweep model, median of three runs per the README's noise-floor rule.

## Models needed

Existing dense models (`tinyllama`, `mistral-7b`, `qwen2.5-3b`) cover the tiled-kernel and
context-shift work. Sliding-window-specific validation may need a model whose GGUF metadata
actually declares a window size — check whether `mistral-7b-instruct-v0.1` (already present) has
this metadata before assuming a new download is needed; flag to the user only if it doesn't.

## Exit criteria

- [ ] Tiled attention kernel numerically matches the CPU oracle at all tested sequence lengths and
      reduces peak GPU memory at long context vs. the old full-materialization kernel (measured).
- [ ] Context-shift works correctly, opt-in only, for both dense and paged KV, both schedules.
- [ ] Default (non-opt-in) behavior is unchanged — still a clear, documented error past
      `MAX_SEQ_LEN`.
- [ ] Sliding-window attention verified correct against a synthetic windowed-metadata fixture and a
      no-op (bit-identical) for non-windowed models. Real-model validation on `gemma-4-E4B` is Tier
      08's exit criterion, not this tier's — confirm it is listed there before closing this one.
- [ ] Cross-surface checklist fully resolved.
- [ ] Perf gate published, both memory thresholds above met, no throughput regression.
- [ ] The context-shift opt-in is in `api/src/main/resources/openapi.yaml` and `juno-api.yaml`
      alongside the code that reads it (README feature-complete rule).
- [ ] Docs (`docs/howto.md`, `docs/agent-arch.txt`, `docs/performance.md`) updated, Juno-native
      language only.
- [ ] `CHANGELOG.md` entry added.
