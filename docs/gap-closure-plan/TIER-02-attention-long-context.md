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

**What Tier 01 handed over (2026-09-27).** Tier 01 shipped the decode residency region through RoPE:
`ResidentQkvPath` runs, per layer at single-sequence decode, the norm, the Q/K/V projections and RoPE on
the device with one download of q, k and v, behind `--gpu-residency` (default off), for +3.6% to +5.4%
decode where it runs. The attention half of that region ("step 3b") is here, not there, by the owner's
decision: after the region the layer still pays a host FP16 conversion and two synchronous copies for
the KV append, and four synchronous uploads plus one download for attention, and removing those is
attention work. `CudaGraphSession` came with it, because a captured graph only pays once a region
issues many launches per wait. Scope items 4 to 6 below carry all three.

**Defect found 2026-09-28: Phi-3.5 always rotates with the long-context RoPE factors.** Found from a
console report: `./juno local` on `Phi-3.5-mini-instruct-Q4_K_M.gguf` answered "Hello" correctly, then
ran on to `max_tokens` with training-data text ("---", "## Instruction 2 ... {ct}"). The stop path was
not at fault: no turn marker or role header was ever produced, so the Session 93 stop (`ChatTurnMarkers`)
had nothing to catch. The model rarely produced `<|end|>`. Measured on the CPU path, teacher-forced
over the same 19 prompt+answer token ids (tokenization confirmed identical to the reference engine's
`/tokenize`, including the `29871` space pieces after each control token and no BOS), probability of
`<|end|>` (32007) after "How can I help you today?":

| Build | Factors | P(`<\|end\|>`) | P(`\n`) |
|---|---|---|---|
| Juno HEAD | long (always) | 0.502 | 0.487 |
| Juno, `Phi3RopeConfig` swapped to short by reflection | short | 0.992 | 0.007 |
| reference engine b9551, `-c 512` / `-c 4096` | short | 0.996 | 0.004 |
| reference engine b9551, `-c 8192` | long | 0.675 | 0.297 |

At the console default (temperature 0.7, top-k 50, top-p 0.9) this is roughly a coin flip per reply:
7 of 10 GPU runs and 4 of 6 CPU runs of a single "Hello" hit the token limit; greedy stops every time.

Cause: `Phi3RopeConfig.selectFactors()` (since `a384152`) returns the long factors whenever the GGUF's
trained `phi3.context_length` (131072) exceeds `rope.scaling.original_context_length` (4096), which is
true for every request, so a 10-token prompt is rotated as if it were a long-context one. The reference
engine picks by the configured per-sequence context instead (the last row shows it degrades the same
way when configured above 4096). **Selecting by Juno's own capacity does not fix it**: `MAX_SEQ_LEN` is
32768, above 4096, so the fix is a context policy, which is why it sits in this tier. Options for the
owner (scope item 7): (a) short factors unless the session is explicitly configured for more than
4096 tokens, failing closed when a short-factor session would cross 4096; (b) the model author's
per-sequence semantics, short until the sequence crosses 4096 and long after, which needs cached K
re-rotated at the crossing and so shares machinery with context-shift (scope item 2).

Also open: at matched long factors Juno reads 0.502 against the reference's 0.675. The short-factor
rows agree to 0.004, so this gap may be a second, long-factor-only discrepancy (for example in how
`attn_factor` or the factor tensor is applied) and must be explained before item 7 closes.

Scope of the damage: every Phi-3.5 reply on every surface, CPU and GPU, since `a384152`. Not a
throughput measurement boundary (the factors change rotation angles, not work); Phi-3.5 greedy-parity
or divergence readings against the reference engine are affected, readings of Juno against itself are
not. If the fix lands before this tier starts, record it under the active tier's **Out-of-tier
changes** (README execution rule 9) and tick item 7 here.

## Scope

### In scope

1. **Tiled/online-softmax attention kernel** (CUDA), replacing `gqa_attention.cu`'s current
   full-materialization design, built on Tier 01's residency primitive. Must remain numerically
   correct (verified against the existing scalar CPU attention path) while reducing peak memory at
   long sequence lengths. **It inherits whatever architecture coverage Tier 01B actually delivered —
   read that tier's exit table, do not assume it covered everything.** Tier 01B ships the kernel path
   for Phi-2, Phi-3, Qwen3 and Qwen3-MoE, and its exit criteria forbid leaving any of them with the
   default resolved off for want of a measurement: a real file exists for every one
   (`phi-2.Q4_K_M.gguf`, `Phi-3.5-mini-instruct-Q4_K_M.gguf`, `Qwen3-1.7B-Q4_K_M.gguf`, and
   `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` at partial offload; see [`INVENTORY.md`](INVENTORY.md)).
   So the expected starting state is default-on for Llama-family, Mistral, Qwen2, Phi-2, Phi-3, Qwen3
   and Qwen3-MoE. It can still be narrower if Tier 01B measured an architecture and found the kernel
   bought nothing or diverged unacceptably there; that tier's exit table is the record, and an earlier
   draft of this paragraph was wrong twice about which files existed (the two Qwen3 files, then the
   plain Phi-2), so read that table and check `models/` rather than trusting this paragraph.
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
3. **Sliding-window attention** for model families that need it — add the windowing option to the
   attention path and confirm it is applied only when the loaded GGUF's metadata declares a window,
   not globally. The mechanism must cover **both** shapes real exporters write: a uniform window
   (every layer windowed, the Mistral-style case) and a **patterned** window (a repeating period in
   which some layers attend globally), because the only real windowed file on disk declares the
   patterned form — see implementation step 4 for the exact keys and why a uniform-only mechanism
   would strand Tier 08.
4. **Attention inside the decode residency region (Tier 01's "step 3b").** Extend `ResidentQkvPath`
   (Tier 01) so that, per layer at single-sequence decode, k and v are appended to the device KV
   mirror (`DeviceKvCache`) without a host round trip and attention (`CudaGqaAttention`) reads q from
   the region, downloading only the attention output. Today, after the region, the layer still pays:
   FP16 conversion on the host plus two synchronous copies for the KV append, four synchronous uploads
   plus one download for attention. Keep the CPU KV tensors written (they are the fallback and the
   source of truth for the mirror); the order and failure handling of `DeviceKvCache.appendToken` and
   the written-prefix watermark (`c91f879`) must hold. Scope and fallbacks as in Tier 01:
   single-sequence decode, CUDA, K-quant device projections, adjacent RoPE, no Q/K/V bias; announce
   (log **and** console, as `GpuResidencyOptions.consoleNotice` does) everything else. Tests:
   bit-identity of the region against the op-at-a-time path; greedy parity on/off on tinyllama,
   mistral-7b, llama-1-30b; per-request device memory flat. Files: `ResidentQkvPath`,
   `GpuResidencyOptions`, `LlamaTransformerHandler`, `DeviceKvCache`, `CudaGqaAttention`; tests
   `ResidentQkvPathTest`, `LlamaTransformerHandlerGpuResidencyTest`; smoke
   `scripts/performance-tests/smoke-gpu-residency.sh`.
5. **`CudaGraphSession` (moved from Tier 01 by the owner on 2026-09-27).** After item 4 lands, measure
   whether capturing one layer's region as a CUDA graph and replaying it (position read from device
   memory, or updated per step via an exec-node parameter update) beats plain launches on the GTX 1080
   at decode width. **Decision rule**: wire it behind the same flag if it saves at least 5% of decode
   forward-pass time on tinyllama and mistral-7b with greedy output unchanged; otherwise delete
   `CudaGraphSession` and `CudaGraphSessionTest` and record the measurement. Either way the class stops
   being dormant in this tier.
6. **Default of `--gpu-residency`.** After item 4 (and 5 if wired), re-measure the region on vs off on
   all four sweep models (full `compare-llama-cpp.sh --gpu` sweeps, both published) and put the default
   to the owner with the numbers; do not change it unasked.
7. *Moved 2026-09-30 (plan review): the factor-selection fix is now
   [Tier 01B](TIER-01B-prefill-throughput.md) scope item 8, an out-of-tier correctness fix, because
   Phi-3.5-mini is a sweep model and the defect affects every reply. What remains here: if the owner
   chose option (a) there, decide whether option (b) is still wanted; and explain the long-factor gap
   (0.502 against the reference's 0.675). Verify 01B's fix against current code before relying on it.*
   *Reduced 2026-10-01: Tier 01B shipped option (a) as a fixed cap (owner decision; Juno has no
   per-session context setting to configure "above 4096" with): `Phi3RopeConfig.selectFactors()`
   returns the short factors whenever the file has them, and `requirePosition` fails closed at
   `original_context_length` on every `Phi3Rope` caller. `Phi3EndOfTurnLiveTest` reads P(32007) =
   0.9924 (0.5016 before). What remains here: (1) whether option (b) is wanted, which would lift
   the 4096-token cap and needs the context-shift machinery; and (2) the long-factor gap (0.502
   against 0.675), which now matters only to option (b), since no Juno sequence rotates with the
   long factors any more. The rest of this item's original text is kept below as the record.*
   **Phi-3.5 LongRoPE factor selection** (defect found 2026-09-28, see "Why this tier, why now").
   Replace `Phi3RopeConfig.selectFactors()`'s trained-context test with the policy the owner picks
   (option a or b), applied identically on every path that calls `Phi3Rope` (CPU, GPU, batched prefill,
   LoRA training's `ropeExtBackward`), and explain the long-factor 0.502 vs 0.675 gap. Files:
   `Phi3RopeConfig`, `Phi3Rope`, `Phi3TransformerHandler`; test `Phi3RopeLoadTest` plus the live test
   below.
8. **The whole decode layer inside the region, on every handler with a device weight path** (added
   2026-09-30; the README's "Program objective: the layer runs on the device"). After item 4 the region
   covers norm, Q/K/V, RoPE, the KV append and attention, and the layer then leaves the device for the
   output projection, the residual add, the second norm and the FFN, each through today's
   op-at-a-time `MatVec` with its own upload and download. Extend the region through the rest of the
   layer: output projection (packed GEMV, as Q/K/V already are), both residual adds, the FFN norm, gate
   and up, SwiGLU (Tier 01B item 6's kernel at width 1) and down, so one decode layer is one upload of
   the residual row and one download of the layer output. Where the region also spans the next layer
   (the residual stream never leaving the device between layers), take it: the threshold below is the
   minimum, the objective is one crossing per forward pass.

   Then give the region to `Phi3TransformerHandler` and `Qwen3TransformerHandler`, the other two
   handlers with device weights, through the same `ResidentChain` and the same flag. That needs the
   split-half RoPE mode (Tier 01B item 6 builds it) and, for Phi-3, its LongRoPE factors folded into the
   inverse-frequency table the kernel already takes from the host, selected by the policy Tier 01B item
   8 fixed. Q/K/V biases (Qwen2) and Qwen3's per-head Q/K norms are the two operations the region does
   not have today; add them, or keep the model on the op-at-a-time path and announce it (log and
   console, as `GpuResidencyOptions.consoleNotice` does). This is the decode lever the 0.70x GPU tg
   end-of-plan target on Phi-3.5-mini depends on, and before this item no tier owned it.

   Order within the tier: item 4, then this item, then item 5 (graph replay pays more once the region
   issues more launches per wait), then item 6 (the default is put to the owner on the final region).

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
| 7 | Pipeline-parallel cluster | sliding-window metadata must propagate to every node holding a shard of a windowed model — **both** the window width and the layer pattern, and the pattern has to be interpreted against each shard's global layer indices, not its local ones, or a sharded patterned model windows the wrong layers |
| 8 | Tensor-parallel cluster | same |
| 9 | LoRA training | confirm training's own attention/backward path (`LoraTrainingMath`) is unaffected — training doesn't currently use the fused GQA kernel |
| 10 | LoRA playback | confirm LoRA-modified attention projections still compose correctly with the new tiled kernel |
| 11 | Vision | N/A — out of scope, must verify vision's separate `VisionEncoder` attention path is untouched |
| 12 | OpenAI REST surface | context-shift needs an explicit request-level opt-in (e.g. an `x_juno_*` extension field, following the project's existing `x_juno_grammar`/`x_juno_loras` naming convention) |
| 13 | Native REST surface | same opt-in surfaced there too |
| 14 | CLI | a `--context-shift` (or similarly named) flag for `local`/`cluster`, off by default |
| 15 | JVM embedding facade | the context-shift opt-in must be settable through `JunoPlayer`/`JunoHttpClient`, or explicitly documented as REST/CLI-only — an embedder that cannot opt in still hits the hard `MAX_SEQ_LEN` error this tier exists to make survivable |

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
   the existing generic `metaInt` accessor.

   **Read both keys, not one.** The expected spelling was confirmed against the only real windowed file
   on disk: `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` declares **`gemma4.attention.sliding_window`** *and*
   **`gemma4.attention.sliding_window_pattern`**, which matches Tier 00's audit finding that its
   windowing is *patterned* rather than uniform — a subset of layers attend globally and the rest attend
   within the window, on a repeating period. So the mechanism must express "every Nth layer is global,
   the rest are windowed", not a single window size applied to every layer. Read
   `<arch>.attention.sliding_window` for the width and `<arch>.attention.sliding_window_pattern` for the
   period, treat a present width with an absent period as "every layer windowed" (the uniform case, which
   is what a Mistral-style exporter writes), and treat an absent width as today's unwindowed behaviour
   whatever the period says.

   This matters because [Tier 08](TIER-08-model-architecture-breadth.md) carries the real-model
   validation on that exact file and is explicitly forbidden from adding a second windowing path. A
   mechanism built here for a single uniform window would make Tier 08's exit criterion unreachable
   without reopening this tier — which its own text anticipates ("that is a Tier 02 defect surfacing
   late") but which costs one design decision to avoid instead.

   Also confirmed while establishing the above, so nobody re-checks it: **`mistral-7b-instruct-v0.1`
   declares no window key at all.** Its full metadata key set is `llama.{block_count, context_length,
   embedding_length, feed_forward_length, attention.head_count, attention.head_count_kv,
   attention.layer_norm_rms_epsilon, rope.dimension_count, rope.freq_base, vocab_size}`. So there is no
   real windowed file this tier can validate against before Tier 08 lands its Gemma handler, and the
   synthetic-fixture split below is the only option rather than a convenience.

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
6. Scope items 4, 8, 5 and 6, in that order: attention inside the decode residency region, then the
   rest of the layer and the Phi-3 and Qwen3 handlers (item 8), then the `CudaGraphSession`
   measurement and its wire-or-delete decision, then the two on/off sweeps and the flag default put to
   the owner. Item 4 is independent of the tiled kernel (item 1) at decode width
   and may land before it; if it does, item 1 must keep the region's bit-identity test passing.

## Tests to write/upgrade before implementation

- **New unit tests** for the tiled kernel: exact-match (within float tolerance) against the scalar
  CPU oracle at short, medium, and long (near/at `MAX_SEQ_LEN`) sequence lengths.
- **New unit tests** for context-shift: `DenseKvTensor`/`KvPageTable`/`PagedKvTensor`/
  `DeviceKvCache` each get a test that grows a session past `MAX_SEQ_LEN` with the opt-in flag set
  and confirms it shifts instead of throwing, and a test confirming the *default* (flag unset)
  behavior still throws (no silent regression of the existing fail-closed guarantee).
- **New unit tests** for sliding-window, one per metadata shape: (a) a *uniform* windowed fixture —
  attention ignores exactly the tokens outside the window, on every layer; (b) a *patterned* windowed
  fixture declaring both `sliding_window` and `sliding_window_pattern` — asserting **which** layers
  attend globally and which are windowed, since the period is the half of the mechanism a single-window
  test cannot reach; (c) width present, period absent — resolves to the uniform case; (d) a non-windowed
  model's behaviour is bit-identical to before this tier (regression guard).
- **`ModelLiveRunnerIT`**: add a long-context check (generate past the old hard-fail point with
  `--context-shift` enabled). The real-model windowed check belongs to Tier 08 and is listed in that
  tier's exit criteria; this tier's windowed coverage is the synthetic fixture plus the unit-level
  mask tests above, which is enough to ship the mechanism without waiting on a later tier.
- **Phi-3.5 end-of-turn live test (scope item 7)**: teacher-force the 19 ids
  `32010 29871 13 10994 32007 29871 13 32001 29871 13 10994 29991 1128 508 306 1371 366 9826 29973`
  on `Phi-3.5-mini-instruct-Q4_K_M.gguf` and assert P(32007) at the last position is **>= 0.95**
  (0.502 before the fix, 0.992 with short factors); a unit test that a short session selects the short
  factors and, under option (b), that crossing 4096 switches to the long ones.
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

  **Milestone (README milestone table, added 2026-09-30).** The GPU pp ratio at `n_prompt=2048` over
  the ratio at `n_prompt=512` is **>= 0.90** on every sweep model, from this tier's closing
  `compare-llama-cpp.sh --gpu --pin-clocks` sweeps at both lengths. Tier 01B established that prefill
  does not fall off between 128 and 512 once attention is on the GPU; nothing has measured 2048, which
  is where a full-materialization attention kernel's cost shows and where the tiled kernel has to earn
  its place in throughput as well as in memory. The reference is Tier 01C's first 2048 sweep.

  **End-of-plan tg targets (raised to 0.70x on 2026-09-30).** Record the GPU tg ratio on Phi-3.5-mini
  and mistral-7b after items 4 to 6 against the 0.70x end-of-plan targets. Items 4 and 5 are the plan's
  main decode levers, so state per model how far they moved it and, if 0.70x is out of reach on what is
  left in the plan (in particular, the residency region does not reach the Phi-3 and Qwen3 handlers),
  say which mechanism is missing and that no tier owns it.

  Throughput must not regress: Juno tg and pp t/s **>= 0.95x** the pre-tier build on every sweep
  model, from a same-hour interleaved A/B with pinned clocks against the pre-tier build (README, "No-regression gates tighter than the floor are Juno-against-Juno"). The llama.cpp-relative ratios are recorded against the program target, not
  gated.
- **Attention inside the residency region (scope item 4)**: extend `ResidentQkvPathTest` with a
  bit-identity case of the region (now through the KV append and attention) against the op-at-a-time
  path at several positions and across concurrent threads, a case that the CPU KV tensors and the
  device mirror's written-prefix watermark agree after every appended token, and the no-retained-device-
  memory and short-lived-thread pooling cases; extend `LlamaTransformerHandlerGpuResidencyTest` for
  greedy parity on/off; re-run `scripts/performance-tests/smoke-gpu-residency.sh` on tinyllama,
  mistral-7b and llama-1-30b (greedy output identical on/off, per-request device memory flat).
  **Threshold**: end-to-end tg with the region on **>= 1.0x** the region-off run on every model where
  the region runs, and **>= 0.95x** everywhere — read from a same-hour A/B alternating the flag
  (region off, on, off, on, off, on) with pinned clocks, per the README's no-regression rule.
- **`CudaGraphSession` (scope item 5)**: a decode-width microbenchmark of one layer's region, graph
  replay against plain launches, plus greedy parity with replay on. **Threshold**: the decision rule in
  scope item 5 (at least 5% of decode forward-pass time on tinyllama and mistral-7b with greedy output
  unchanged, or delete the class and its test).
- **The whole decode layer (scope item 8)**: extend `ResidentQkvPathTest` (or its successor) with
  bit-identity of the full-layer region against the op-at-a-time path at several positions and across
  concurrent threads, per handler; greedy parity on/off on tinyllama, mistral-7b, Phi-3.5-mini and
  Qwen3-1.7B; per-request device memory flat in `smoke-gpu-residency.sh`, which gains the two new
  handlers. A `@Tag("gpu")` test reads the decode-phase copy counts off `juno.DeviceStaging` for one
  generated token and asserts the threshold below, so the property is held by a test and not only by a
  sweep.
  **Threshold** (from a `--device-spans` run, decode phase, per generated token, on every model where
  the region runs): host-to-device copies **<= 1 x layers** and device-to-host copies **<= 1 x layers
  + 1** (the `+ 1` is the logits); the end-to-end tg with the region on **>= 1.0x** region-off from a
  same-hour pinned A/B alternating the flag; greedy output identical on/off. On a prefill window, the
  same run shows **0** activation device-to-host copies inside the window apart from the window's
  final hidden rows and logits, which Tier 01B items 2 and 6 deliver and this tier re-verifies after
  its own kernel changes. Record the GPU tg ratios against the 0.70x end-of-plan targets.
- **Flag default (scope item 6)**: two published `compare-llama-cpp.sh --gpu` sweeps on all four sweep
  models, `--gpu-residency on` and `off`, read side by side.

## Models needed

Existing dense models (`tinyllama`, `mistral-7b`, `qwen2.5-3b`) cover the tiled-kernel and
context-shift work. **No new download is needed for the windowed work, and none would help.**
`mistral-7b-instruct-v0.1` was checked and declares no window key (its full `llama.*` key set is
listed in implementation step 4), and the only file on disk that does declare one —
`gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` — is not loadable until Tier 08's Gemma handler exists. That is
why this tier ships against synthetic fixtures and Tier 08 carries the real-model validation.

## Exit criteria

- [ ] Phi-3.5's factor-selection fix verified as landed by Tier 01B item 8 (end-of-turn live test
      still >= 0.95), option (b) decided if 01B shipped option (a), and the long-factor gap to the
      reference engine explained or fixed (scope item 7).
- [ ] Tiled attention kernel numerically matches the CPU oracle at all tested sequence lengths and
      reduces peak GPU memory at long context vs. the old full-materialization kernel (measured).
- [ ] Context-shift works correctly, opt-in only, for both dense and paged KV, both schedules.
- [ ] Default (non-opt-in) behavior is unchanged — still a clear, documented error past
      `MAX_SEQ_LEN`.
- [ ] Both window keys are read — `<arch>.attention.sliding_window` and
      `<arch>.attention.sliding_window_pattern` — and the patterned case is covered by a unit test
      asserting which layers attend globally and which are windowed, not only that a window is applied.
      A present width with an absent period resolves to the uniform case.
- [ ] Sliding-window attention verified correct against synthetic uniform and patterned
      windowed-metadata fixtures, and a no-op (bit-identical) for non-windowed models. Real-model
      validation on `gemma-4-E4B` is Tier 08's exit criterion, not this tier's — confirm it is listed
      there before closing this one, and confirm the keys this tier chose are the ones that file
      declares (they are, as of the check recorded in implementation step 4).
- [ ] Attention inside the decode residency region (scope item 4): k and v appended to the device
      mirror and attention read from the region, one download of the attention output per layer;
      bit-identical to the op-at-a-time path by test; greedy output identical on/off on tinyllama,
      mistral-7b and llama-1-30b; per-request device memory flat in `smoke-gpu-residency.sh`.
      **Threshold**: end-to-end tg with the region on >= 1.0x the region-off run on every model where
      it runs, and >= 0.95x everywhere.
- [ ] The whole decode layer runs inside the region (scope item 8) on the LLaMA family, Phi-3 and
      Qwen3, or the handler or operation that cannot is announced: decode-phase copies per generated
      token <= 1 x layers host-to-device and <= 1 x layers + 1 device-to-host, read off
      `juno.DeviceStaging` by test and by a published `--device-spans` run; tg region-on >= 1.0x
      region-off (same-hour pinned A/B); greedy output identical; per-request device memory flat; zero
      activation device-to-host copies inside a prefill window re-verified.
- [ ] `CudaGraphSession` decided by measurement (scope item 5): wired behind `--gpu-residency` if it
      saves at least 5% of decode forward-pass time on tinyllama and mistral-7b with greedy output
      unchanged, otherwise deleted with `CudaGraphSessionTest` and the measurement recorded. Not
      dormant either way.
- [ ] `--gpu-residency` default put to the owner (scope item 6) with the two published on/off sweeps
      on all four sweep models; changed only on the owner's decision.
- [ ] Cross-surface checklist fully resolved.
- [ ] Perf gate published, both memory thresholds above met, no throughput regression.
- [ ] Milestone (pp ratio at 2048 over 512 >= 0.90 on every sweep model) reported met or missed with
      its number; GPU tg ratios recorded against the 0.70x end-of-plan targets, with the missing
      mechanism named if they are out of reach.
- [ ] The context-shift opt-in is in `api/src/main/resources/openapi.yaml` and `juno-api.yaml`
      alongside the code that reads it (README feature-complete rule).
- [ ] Docs (`docs/howto.md`, `docs/agent-arch.txt`, `docs/performance.md`) updated, Juno-native
      language only.
- [ ] `CHANGELOG.md` entry added.
