# Tier 02B: Context policy (context shifting, sliding windows, Phi-3.5 LongRoPE)

Status: not started
Gap analysis refs: §1.2 (the context half)

**Split out of [Tier 02](TIER-02-attention-long-context.md) on 2026-10-04 (plan review).** This tier's
items were Tier 02's items 2, 3 and 7. Item mapping, for references written before the split:

| Here | Was |
|---|---|
| item 1, context shifting | Tier 02 item 2 |
| item 2, sliding-window attention | Tier 02 item 3 |
| item 3, Phi-3.5 LongRoPE remainder | Tier 02 item 7 |

## Objective

Stop failing hard once a sequence exceeds `MAX_SEQ_LEN` when the caller opts in to context shifting,
apply sliding-window attention exactly where a model's GGUF declares it (uniform or patterned), and close
the Phi-3.5 LongRoPE remainder Tier 01B left: whether sequences may cross 4096 tokens on the long factors,
and why Juno's long-factor reading differs from the reference engine's.

## Why this tier, why now

These are policy decisions about *what* the KV cache holds, and [Tier 03](TIER-03-kv-cache-maturity.md)'s
block-table layout needs them settled before it is locked in. That is the whole of the reason this tier
sits before Tier 03, and it is why it was split from Tier 02: none of its items carries a throughput
target, and keeping them in Tier 02 held every remaining GPU throughput lever behind feature work.

It runs after Tier 02 so that sliding-window masking is a parameter of the tiled kernel Tier 02 ships
(Tier 02 item 1 builds the kernel with a per-layer window parameter for exactly this), not a mask bolted
onto the full-materialization kernel that Tier 02 replaces.

**Defect found 2026-09-28: Phi-3.5 always rotated with the long-context RoPE factors.** Found from a
console report: `./juno local` on `Phi-3.5-mini-instruct-Q4_K_M.gguf` answered "Hello" correctly, then
ran on to `max_tokens` with training-data text. Measured on the CPU path, teacher-forced over the same 19
prompt+answer token ids (tokenization confirmed identical to the reference engine's `/tokenize`),
probability of `<|end|>` (32007) after "How can I help you today?":

| Build | Factors | P(`<\|end\|>`) | P(`\n`) |
|---|---|---|---|
| Juno HEAD (2026-09-28) | long (always) | 0.502 | 0.487 |
| Juno, `Phi3RopeConfig` swapped to short by reflection | short | 0.992 | 0.007 |
| reference engine b9551, `-c 512` / `-c 4096` | short | 0.996 | 0.004 |
| reference engine b9551, `-c 8192` | long | 0.675 | 0.297 |

Cause: `Phi3RopeConfig.selectFactors()` (since `a384152`) returned the long factors whenever the GGUF's
trained `phi3.context_length` (131072) exceeded `rope.scaling.original_context_length` (4096). Tier 01B
item 8 shipped option (a) as a fixed cap (owner decision, 2026-10-01): `selectFactors()` returns the short
factors whenever the file has them, and `requirePosition` fails closed at `original_context_length` on
every `Phi3Rope` caller. `Phi3EndOfTurnLiveTest` reads P(32007) = 0.9924. Two things remain, and they are
item 3 below.

## Scope

### In scope

1. **Context shifting**: when a session's KV would exceed `MAX_SEQ_LEN`, instead of throwing
   `IllegalStateException` in `DenseKvTensor`/`KvPageTable`/`PagedKvTensor`/`DeviceKvCache`, support
   an explicit, opt-in shift policy (drop oldest N non-system tokens, keep going) — opt-in because
   silently dropping context is itself a form of silent degrade the project's fail-closed philosophy
   would otherwise reject; the default behavior stays a clear error unless the caller opts in via a
   flag/request parameter. Shifting re-rotates the cached K of the kept positions by the shift distance
   (or recomputes it); state which, and keep the CPU KV tensors and the device mirror's written-prefix
   watermark consistent across a shift.
2. **Sliding-window attention** for model families that need it — add the windowing option to the
   attention path and confirm it is applied only when the loaded GGUF's metadata declares a window,
   not globally. The mechanism must cover **both** shapes real exporters write: a uniform window
   (every layer windowed, the Mistral-style case) and a **patterned** window (a repeating period in
   which some layers attend globally), because the only real windowed file on disk declares the
   patterned form — see implementation step 3 for the exact keys and why a uniform-only mechanism
   would strand Tier 08. On the GPU, the window is Tier 02's tiled kernel's per-layer window parameter;
   on the CPU, the scalar attention path takes the same per-layer window.
3. **Phi-3.5 LongRoPE remainder.** Tier 01B shipped option (a), a fixed cap at 4096 tokens. What remains:
   (a) decide with the owner whether option (b) is wanted — the model author's per-sequence semantics,
   short factors until the sequence crosses 4096 and long after, which needs cached K re-rotated at the
   crossing and so shares machinery with item 1; and (b) explain the long-factor gap (0.502 against the
   reference's 0.675 at matched long factors), which matters only if option (b) is taken, since no Juno
   sequence rotates with the long factors any more. Verify Tier 01B's fix against current code first.
   Files: `Phi3RopeConfig`, `Phi3Rope`, `Phi3TransformerHandler`; tests `Phi3RopeLoadTest`,
   `Phi3EndOfTurnLiveTest`.

### Out of scope

- Automatic (non-opt-in) context management heuristics — this tier ships the mechanism and an
  explicit opt-in flag, not a policy for when to use it automatically.
- Any attention-kernel performance work — Tier 02.

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | sliding-window and context-shift logic must work identically CPU and GPU since they're KV/scheduling-level, not kernel-level; scalar attention gains the per-layer window |
| 2 | CUDA GPU inference | the window is the tiled kernel's parameter; context shift must keep `DeviceKvCache`'s mirror and watermark consistent |
| 3 | ROCm GPU inference | sliding-window/context-shift (KV-level) should work on ROCm's existing path, where attention runs on the CPU; NEEDS-AMD-HARDWARE for final validation |
| 4 | Static schedule | context-shift must interact correctly with `BatchConfig`'s "all requests start at step 0" constraint — a mid-batch shift changes sequence length for one member only |
| 5 | Continuous schedule | context-shift interacting with `ContinuousBatchEngine`'s per-slot state needs its own test — a slot that shifts mid-stream must not corrupt `ContinuousPrefillState` bookkeeping for other slots |
| 6 | Single-node local mode | primary dev surface |
| 7 | Pipeline-parallel cluster | sliding-window metadata must propagate to every node holding a shard of a windowed model — **both** the window width and the layer pattern, and the pattern has to be interpreted against each shard's global layer indices, not its local ones, or a sharded patterned model windows the wrong layers |
| 8 | Tensor-parallel cluster | same |
| 9 | LoRA training | confirm training's own attention/backward path (`LoraTrainingMath`) honours a declared window or fails closed on a windowed model; training does not use context shift |
| 10 | LoRA playback | a LoRA-played session that shifts must keep the adapter's composition correct |
| 11 | Vision | a shift must never drop image-token positions silently; fail closed on a vision session that would need to shift, or keep the image span, and test whichever is chosen |
| 12 | OpenAI REST surface | context-shift needs an explicit request-level opt-in (e.g. an `x_juno_*` extension field, following the project's existing `x_juno_grammar`/`x_juno_loras` naming convention) |
| 13 | Native REST surface | same opt-in surfaced there too |
| 14 | CLI | a `--context-shift` (or similarly named) flag for `local`/`cluster`, off by default |
| 15 | JVM embedding facade | the context-shift opt-in must be settable through `JunoPlayer`/`JunoHttpClient`, or explicitly documented as REST/CLI-only — an embedder that cannot opt in still hits the hard `MAX_SEQ_LEN` error this tier exists to make survivable |

## Implementation steps

1. Run `scripts/performance-tests/check-plan-thresholds.sh` first, then verify Tier 01B's Phi-3.5 fix
   against current code (`Phi3EndOfTurnLiveTest` still >= 0.95).
2. Implement context-shift as an opt-in KV-cache operation (works for both dense and paged KV, and the
   device mirror); wire the opt-in flag through CLI, both REST surfaces and the facade.
3. Implement sliding-window attention, driven by GGUF metadata. No window key is read anywhere
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
   without reopening this tier.

   Also confirmed while establishing the above, so nobody re-checks it: **`mistral-7b-instruct-v0.1`
   declares no window key at all.** Its full metadata key set is `llama.{block_count, context_length,
   embedding_length, feed_forward_length, attention.head_count, attention.head_count_kv,
   attention.layer_norm_rms_epsilon, rope.dimension_count, rope.freq_base, vocab_size}`. So there is no
   real windowed file this tier can validate against before Tier 08 lands its Gemma handler, and the
   synthetic-fixture split below is the only option rather than a convenience.

   **Validation does not depend on Tier 08, and must not be deferred to it.** Tier 08's Gemma handler
   needs this tier's windowing mechanism, so the validation is split:
   - **This tier** validates the mechanism against a synthetic fixture — a GGUF whose metadata
     declares a window — plus a unit-level test that the windowed causal mask ignores exactly the
     tokens outside the window and that a model *without* the key is bit-identical to pre-tier
     behaviour. That is sufficient to prove the mechanism and to ship it.
   - **Tier 08** carries the end-to-end validation on `gemma-4-E4B` as one of its own exit criteria,
     once its handler makes that file loadable.
4. Item 3: put option (b) to the owner with what it costs (the re-rotation machinery item 1 builds) and,
   if taken, explain or fix the long-factor gap.
5. Run the full cross-surface smoke matrix, including a long-context stress case (loop conversation
   turns until the shift boundary is hit) for both static and continuous schedules.

## Tests to write/upgrade before implementation

- **Plan check, first**: `scripts/performance-tests/check-plan-thresholds.sh` passes before any other
  test or code in this tier (README execution rule 7).
- **New unit tests** for context-shift: `DenseKvTensor`/`KvPageTable`/`PagedKvTensor`/
  `DeviceKvCache` each get a test that grows a session past `MAX_SEQ_LEN` with the opt-in flag set
  and confirms it shifts instead of throwing, and a test confirming the *default* (flag unset)
  behavior still throws (no silent regression of the existing fail-closed guarantee). A shifted
  session's next-token logits match a fresh session prefilled with the kept tokens at their shifted
  positions, within float tolerance.
- **New unit tests** for sliding-window, one per metadata shape: (a) a *uniform* windowed fixture —
  attention ignores exactly the tokens outside the window, on every layer; (b) a *patterned* windowed
  fixture declaring both `sliding_window` and `sliding_window_pattern` — asserting **which** layers
  attend globally and which are windowed, since the period is the half of the mechanism a single-window
  test cannot reach; (c) width present, period absent — resolves to the uniform case; (d) a non-windowed
  model's behaviour is bit-identical to before this tier (regression guard); (e) a pipeline shard's
  window pattern read against global layer indices.
- **`ModelLiveRunnerIT`**: add a long-context check (generate past the old hard-fail point with
  `--context-shift` enabled). The real-model windowed check belongs to Tier 08.
- **Phi-3.5 end-of-turn live test (item 3)**: teacher-force the 19 ids
  `32010 29871 13 10994 32007 29871 13 32001 29871 13 10994 29991 1128 508 306 1371 366 9826 29973`
  on `Phi-3.5-mini-instruct-Q4_K_M.gguf` and assert P(32007) at the last position is **>= 0.95**;
  under option (b), a unit test that crossing 4096 switches to the long factors.
- **New bash smoke script**: `scripts/performance-tests/smoke-context-policy.sh` — drives a long
  multi-turn conversation via the REST API until the shift boundary, asserts the server keeps
  responding instead of erroring, and asserts a second run *without* the opt-in flag still gets the
  documented hard error at the same point.
- **Standing CPU and allocation gate** (README, "Test infrastructure"): run against the pre-tier jar
  and score it before closing this tier.
- **Perf gate (required)**: context shift and the window mask touch the KV and attention paths —
  `compare-lora.sh`, plus `compare-llama-cpp.sh` for a llama.cpp-relative reading (per README's
  llama.cpp-relative gate); publish under `docs/perf-compare/`.

  **Threshold.**
  - Nothing changes for a request that does not opt in and a model that declares no window: Juno tg
    and pp t/s **>= 0.95x** the pre-tier build on every sweep model, from a same-hour interleaved A/B
    with pinned clocks (README, "No-regression gates tighter than the floor are Juno-against-Juno"),
    and greedy output bit-identical.
  - The decode step that performs a shift takes **<= 3.0x** the median decode step at the same depth,
    on tinyllama and mistral-7b at `MAX_SEQ_LEN`, CPU and GPU, median of three. A shift that stalls a
    stream for seconds is a latency defect even if it is correct; above the bound, move the
    re-rotation off the decode step or record why it cannot be.

## Models needed

Existing dense models (`tinyllama`, `mistral-7b`, `qwen2.5-3b`) cover context shifting;
`Phi-3.5-mini-instruct-Q4_K_M.gguf` covers item 3. **No new download is needed for the windowed work,
and none would help** (see implementation step 3).

## Exit criteria

- [ ] Phi-3.5's factor-selection fix verified as landed by Tier 01B item 8 (end-of-turn live test
      still >= 0.95), option (b) decided with the owner, and, if taken, the long-factor gap explained or
      fixed (item 3).
- [ ] Context-shift works correctly, opt-in only, for dense KV, paged KV and the device mirror, both
      schedules.
- [ ] Default (non-opt-in) behavior is unchanged — still a clear, documented error past
      `MAX_SEQ_LEN`.
- [ ] Both window keys are read — `<arch>.attention.sliding_window` and
      `<arch>.attention.sliding_window_pattern` — and the patterned case is covered by a unit test
      asserting which layers attend globally and which are windowed, not only that a window is applied.
      A present width with an absent period resolves to the uniform case.
- [ ] Sliding-window attention verified correct against synthetic uniform and patterned
      windowed-metadata fixtures on the CPU and GPU paths, and a no-op (bit-identical) for non-windowed
      models. Real-model validation on `gemma-4-E4B` is Tier 08's exit criterion, not this tier's —
      confirm it is listed there before closing this one.
- [ ] Cross-surface checklist fully resolved, including row 15 (facade).
- [ ] Perf gate published: Juno t/s >= 0.95x the pre-tier build, the shift step <= 3.0x the median
      decode step, and the standing CPU and allocation gate met.
- [ ] The context-shift opt-in is in `api/src/main/resources/openapi.yaml` and `juno-api.yaml`
      alongside the code that reads it (README feature-complete rule).
- [ ] Docs (`docs/howto.md`, `docs/agent-arch.txt`) updated, Juno-native language only.
- [ ] `CHANGELOG.md` entry added.
