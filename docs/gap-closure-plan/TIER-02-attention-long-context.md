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
   long sequence lengths.
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
4. Implement sliding-window attention, driven by GGUF metadata (`rope.scaling.*`-style metadata,
   confirm which key llama.cpp-format GGUF exporters actually use — check `Devstral`'s Mistral3
   metadata if it turns out to be a windowed architecture, once Tier 08 makes that model loadable).
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
  `--context-shift` enabled) and a sliding-window model check once Tier 08 makes a windowed model
  loadable (may need to sequence this specific sub-check after Tier 08, noted here so it isn't
  forgotten).
- **New bash smoke script**: `scripts/performance-tests/smoke-tier02-attention-context.sh` —
  drives a long multi-turn conversation via the REST API until the shift boundary, asserts the
  server keeps responding instead of erroring, and asserts a second run *without* the opt-in flag
  still gets the documented hard error at the same point.
- **Perf gate (required)**: the tiled kernel is a hot-path change — `compare-lora.sh` plus a
  dedicated long-context latency/memory microbenchmark, plus `compare-llama-cpp.sh` for a
  llama.cpp-relative pp/tg reading on the same models (per README's llama.cpp-relative gate); publish
  under `docs/perf-compare/`. Threshold: peak GPU memory at the longest tested sequence length must
  drop by a stated percentage vs. the old full-materialization kernel (measure the old kernel's
  actual number first, then set this tier's target relative to it — do not accept "some reduction"
  with no number).

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
- [ ] Sliding-window attention verified correct for a windowed model and a no-op (bit-identical) for
      non-windowed models.
- [ ] Cross-surface checklist fully resolved.
- [ ] Perf gate published with no unexplained regression.
- [ ] Docs (`docs/howto.md`, `docs/agent-arch.txt`, `docs/performance.md`) updated, Juno-native
      language only.
- [ ] `CHANGELOG.md` entry added.
