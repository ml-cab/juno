# Tier 02B: Context policy (context shifting, sliding windows, Phi-3.5 LongRoPE)

Status: **complete** (2026-10-09). Every exit criterion is checked. Context shifting is an opt-in on every handler,
both schedules, the CLI, both REST surfaces and the facade (cluster fail-closed, owner decision 2), with the device KV
copy shifted in place on the GPU; the shifting step takes 0.54x to 2.36x a decode step at 32,768 positions (pinned,
`<= 3.0x`). Sliding-window attention is read from both GGUF keys and applied on every CPU and GPU path, every handler,
LoRA play and training. No regression against the pre-tier build (pinned GPU and CPU, LoRA, vision; greedy output
identical). Phi-3.5's 4096 cap kept (owner decision 1).
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
   flag/request parameter. The boundary is the lower of `MAX_SEQ_LEN` and the handler's own position
   limit: for Phi-3.5 that is its 4096-token cap (`Phi3RopeConfig.requirePosition`), so a Phi-3.5 session
   shifts at 4096, not at 32768 (owner decision 1, 2026-10-08). Shifting re-rotates the cached K of the kept positions by the shift distance
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
   *Closed 2026-10-08 (owner decision 1): option (b) declined; the fixed 4096 cap stays and item 1 shifts
   Phi-3.5 sessions at 4096.* As written before the decision:
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
- *Added 2026-10-08 (step 2), confirmed by the owner the same day (decision 2):* **context shift in
  cluster and tensor-parallel mode** is FAIL-CLOSED, not implemented; the node RPC moved to
  [Tier 13](TIER-13-server-surface-clustering.md) item 5. The nodes have no call to shift a request's KV
  (the node RPCs carry forward passes only, and the cluster pipeline clients do not even forward
  `evict`). `./juno cluster --context-shift on` exits with an explicit error (launcher and
  `ConsoleMain`), and a request opting in on a cluster pipeline gets HTTP 400 naming the field. Rows 7
  and 8 of the checklist are therefore FAIL-CLOSED for context shift; their sliding-window half (item 2)
  is unaffected.
- *Added 2026-10-08 (step 2):* **context shift with `--spec-type draft-simple`** is FAIL-CLOSED (HTTP
  400 / error before any forward pass): the draft model keeps its own KV at positions tied to token
  indices, which a shift of the target's KV would leave misaligned. `ngram-simple` keeps no KV and is
  supported.

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

   *Corrected 2026-10-08 (step 3, plan-versus-code check), two claims that do not hold as written.*
   (1) **The pattern key is not a period on the only file that declares it.** `gemma-4-E4B` stores
   `gemma4.attention.sliding_window_pattern` as an array of 42 booleans, one per layer (`true` =
   windowed, `false` = global; the array repeats five windowed layers then one global), with
   `sliding_window = 512`. So the mechanism reads the pattern in both forms: a boolean array whose
   length must equal `<arch>.block_count` (the form this file writes), and an integer period `N`, under
   which the last layer of each period is global (`il % N == N - 1`), the same layers the array names
   for `N = 6`. Any other value type, an array of the wrong length, or a negative width or period is
   rejected at load rather than guessed at. (2) **Gemma 4 is not the only windowed file on disk.**
   `Phi-3.5-mini-instruct-Q4_K_M.gguf` declares `phi3.attention.sliding_window = 262144` with no
   pattern key, which under the rule above is a uniform window on every layer. It never binds: 262144
   exceeds the handler's 4096-token cap, and a window no shorter than the context attends over exactly
   the keys of no window (the kernel's documented contract, and the CPU path's by construction). So
   the Phi-3 handler reads and applies it, and the sweep model's output must stay bit-identical; the
   no-op guard covers it. Found by reading every file's metadata keys (`./juno gguf-info`), not tensor
   data. Gemma 4 also declares `key_length_swa`, `value_length_swa`, `rope.freq_base_swa` and
   `rope.dimension_count_swa` (windowed layers with their own head width and rotation); those are
   handler shape, Tier 08's, and not part of the window mechanism.

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
4. *Done 2026-10-08: option (b) declined (owner decision 1); nothing further to build.* Item 3: put option (b) to the owner with what it costs (the re-rotation machinery item 1 builds) and,
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
  *Corrected 2026-10-08 (step 2), two claims that do not hold as written.* (1) **The fresh-session oracle
  is false for any model with more than one layer.** In every layer after the first, the kept tokens'
  keys and values were computed while they attended to the discarded tokens, and a shift keeps them; only
  a recompute would match a fresh session. Measured on CUDA TinyLlama: max logit difference 1.48 against
  the fresh session; it passed on the synthetic CPU model only because that model's 0.02-range weights
  make attention inert (a negative control, keys moved with no rotation at all, also stayed within
  1e-3 there). The criterion is now: the shifted request's next logits match a request holding the
  **oracle KV** (the pre-shift KV with the discarded rows removed and each kept key unrotated at its old
  position and rotated at its new one with the handler's own forward rotation, independent of the
  shift's rotation code), within float tolerance on the CPU, and on the GPU within the device-vs-host
  attention difference the same handler shows without a shift. (2) **The opt-in is not a KV-class
  property.** The decision to shift belongs to the generation layer, which knows the request's choice,
  its system prompt and the pipeline's limit. The KV classes gain the primitive (move rows, re-rotate
  keys, truncate), and their tests fill to `MAX_SEQ_LEN`, shift, and keep writing, beside a test that
  the same write without a shift still throws.
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

- [x] Phi-3.5's factor-selection fix verified as landed by Tier 01B item 8 (end-of-turn live test
      still >= 0.95), option (b) decided with the owner, and, if taken, the long-factor gap explained or
      fixed (item 3).
      *Met 2026-10-08.* The live test reads 0.9924 on HEAD `de98ff6` (execution record, step 1).
      **Evidence (not published):** `Phi3EndOfTurnLiveTest`, a unit-level test. Option (b) declined by the
      owner (decision 1), so the long-factor gap needs no explanation; the shift boundary at 4096 for
      Phi-3.5 is carried by the context-shift criterion below.
- [x] Context-shift works correctly, opt-in only, for dense KV, paged KV and the device mirror, both
      schedules, and fires at the lower of `MAX_SEQ_LEN` and the handler's own position limit (Phi-3.5:
      4096, owner decision 1), with a Phi-3.5 test at that boundary.
      *Met 2026-10-08 (step 2 record):* `KvContextShiftTest`, `ContextShiftLiveTest` (every handler family
      against the oracle KV), the CUDA mirror tests, `GenerationLoopContextShiftTest` (both schedules),
      `Phi3ContextShiftAtLimitTest` (real Phi-3.5 on CUDA across 4096) and `smoke-context-policy.sh` (7 of 7).
- [x] Default (non-opt-in) behavior is unchanged — still a clear, documented error past
      `MAX_SEQ_LEN`.
      *Met 2026-10-08:* the KV cap still throws without a shift (`KvContextShiftTest`, 4 cases); a request
      without the opt-in fails at the limit on every path (`GenerationLoopContextShiftTest`,
      `ContextShiftRequestFieldTest`, `Phi3ContextShiftAtLimitTest`, smoke turn 7 on both schedules); the error
      is documented in `docs/howto.md` ("Maximum sequence length", "Context shifting").
- [x] Both window keys are read — `<arch>.attention.sliding_window` and
      `<arch>.attention.sliding_window_pattern` — and the patterned case is covered by a unit test
      asserting which layers attend globally and which are windowed, not only that a window is applied.
      A present width with an absent period resolves to the uniform case.
      *Met 2026-10-09 (step 3 record):* `SlidingWindowTest` (boolean-array and integer-period patterns name the
      global layers; width without pattern is uniform; real gemma-4-E4B header), and per handler layer the KV
      overwrite oracle (`LlamaTransformerHandlerSlidingWindowTest`, `SlidingWindowLiveTest`).
- [x] Sliding-window attention verified correct against synthetic uniform and patterned
      windowed-metadata fixtures on the CPU and GPU paths, and a no-op (bit-identical) for non-windowed
      models. Real-model validation on `gemma-4-E4B` is Tier 08's exit criterion, not this tier's —
      confirm it is listed there before closing this one.
      *Met 2026-10-09 (step 3 record).* **Evidence (not published):** unit and real-model tests. CPU exact (`LlamaTransformerHandlerSlidingWindowTest`, `SlidingWindowLiveTest`
      on every handler family and LoRA play and training); GPU (`SlidingWindowGpuTest`: decode within 5% of the
      window's effect of host attention, prefill matching width W over W - 1 and W + 1 by 2.17x to 7.3x); no-op bit
      for bit on the CPU (`GqaMathWindowTest`, wide-window handler test) and unchanged launch arguments on the GPU.
      End-to-end greedy bit-identity on the sweep models is scored by the perf gate below. Tier 08 lists the
      gemma-4-E4B criterion (its exit criteria, "Sliding-window attention validated end to end").
- [x] Cross-surface checklist fully resolved, including row 15 (facade).
      *Met 2026-10-09 (step 5 record, table):* PASS, N/A with reason, or FAIL-CLOSED (tested) on every row for both
      items; ROCm rows NEEDS-AMD-HARDWARE.
- [x] Perf gate published: Juno t/s >= 0.95x the pre-tier build, the shift step <= 3.0x the median
      decode step, and the standing CPU and allocation gate met.
      *Met 2026-10-09 (owner runs, pinned):* `docs/perf-compare/20261009T151747Z-context-policy-close-gate` (GPU and
      CPU, allocation, GC, greedy output; Mistral 7B CPU generation on the request-timed median by owner decision 5),
      `docs/perf-compare/20261009T195800Z-context-policy-final-gate` (GPU re-run on the final jar: tg 0.992x to 1.031x,
      pp 0.990x to 1.000x; shift step 0.541 / 2.362 / 0.581 / 0.630 against `<= 3.0`), LoRA and vision
      `docs/perf-compare/20261009T192344Z-context-policy-lora-vision`, reference-relative reading
      `docs/perf-compare/20261009T195529Z`.
      *2026-10-09, first two parts read (owner run, pinned):* `docs/perf-compare/20261009T151747Z-context-policy-close-gate`:
      GPU tg 0.978x to 1.005x, pp 0.983x to 0.999x; CPU tg 0.999x and 1.003x, pp 0.993x to 1.001x; allocation 0.998x
      to 1.000x; GC unchanged; greedy output equal on every run. Mistral 7B CPU generation scored on the request-timed
      median of three, 1.002x (owner decision 5). Parts A and C met; still open: the shift step's `<= 3.0x` bound
      (step 5).
- [x] The context-shift opt-in is in `api/src/main/resources/openapi.yaml` and `juno-api.yaml`
      alongside the code that reads it (README feature-complete rule).
      *Met 2026-10-08:* `contextShift` (native `InferenceRequest`) and `x_juno_context_shift` (chat
      completions, schema and extension table). `inference.proto` is unchanged: no RPC shape changed, the
      cluster path refuses the opt-in, and nothing implements its client-facing `InferenceService`.
- [x] Docs (`docs/howto.md`, `docs/agent-arch.txt`) updated, Juno-native language only.
      *2026-10-08: context-shift half done* (howto "Context shifting", flag row, facade; agent-arch class
      map; README). *2026-10-09: sliding-window half done* (howto "Sliding-window attention", agent-arch
      `SlidingWindow`/`GqaMath`, README). *2026-10-09, step 5:* the in-place device shift and its latency (howto, measured
      on the pinned gate), agent-arch (`DeviceKvCache.shiftInPlace`, `KvShiftKernel`, `ContextShiftStepBench`); the
      surface claims match the step 5 matrix.
- [x] `CHANGELOG.md` entry added.
      *2026-10-08: context-shift entry added (Session 123).* *2026-10-09: sliding-window entry added (Session
      124), extended with the in-place device shift in step 5.*

## Execution record

### 2026-10-08: implementation step 1 (plan check; Phi-3.5 factor fix verified)

**Plan check.** `scripts/performance-tests/check-plan-thresholds.sh` passes (22 tier files, milestone and
end-of-plan tables checked).

**Tier 01B's Phi-3.5 fix, verified against current code** (HEAD `de98ff6`, clean tree apart from the
untracked `.github/`). `Phi3RopeConfig.selectFactors()` returns the short factors whenever the file carries
them, and `requirePosition` fails closed at `original_context_length` when the file holds long factors back.
Every rotation path calls it: the host rotation (`Phi3Rope.ropeExt`/`ropeExtBackward`), the device rotation
(`CudaPhi3Rope`, prefill window and decode), and the two device-region entry points in
`Phi3TransformerHandler` (prefill window and decode layer), which check before any device work. So no Juno
sequence rotates with the long factors, on either backend.

| Test (`mvn -q test -pl node -am -Dtest=...`) | Tests | Result | Reading |
|---|---|---|---|
| `Phi3EndOfTurnLiveTest` (CPU, real `Phi-3.5-mini-instruct-Q4_K_M.gguf`) | 1 | pass | P(`<\|end\|>`) = **0.9924** (>= 0.95; Tier 01B read 0.9924) |
| `Phi3RopeFactorPolicyTest` | 5 | pass | short factors selected; 4095 accepted, 4096 refused on host forward and backward |
| `Phi3RopeLoadTest` | 1 | pass | short[0..2] = 1.00, 1.02, 1.03; long[0..2] = 1.08, 1.11, 1.14 |

All three are regression tests for code that already exists; they passed on their first run and were not
seen failing in this tier.

**Plan-versus-code check for the later steps** (read-only, nothing changed in code). Every claim the later
steps start from holds on `de98ff6`: no GGUF window key is read anywhere (`sliding` appears only in comments,
in `LlamaFamilyArchitectures` and `LlamaTransformerHandler`); `DenseKvTensor`, `PagedKvTensor`, `KvPageTable`
and `DeviceKvCache` each throw `IllegalStateException` past `MAX_SEQ_LEN`; Tier 02's tiled kernel already
takes a per-call `window` (`GqaAttentionKernel`, keys `[max(0, seqLen - window), seqLen)`, 0 meaning none); no
context-shift code, flag or request field exists yet.

**Finding that item 1 must absorb (not a scope change).** `MAX_SEQ_LEN` is 32768 (`DenseKvTensor`,
`DeviceKvCache`), but Phi-3.5-mini fails closed at position 4096 (`requirePosition`). Item 1 as written
shifts "when a session's KV would exceed `MAX_SEQ_LEN`", so a Phi-3.5 session would hit the 4096 error and
never shift. The shift boundary has to be the lower of the KV limit and the handler's own position limit,
and step 2's tests need a Phi-3.5 case at 4096 beside the dense models at `MAX_SEQ_LEN`. Under option (a)
this is the only way a long Phi-3.5 conversation survives at all, which is the main input to decision 1.
Note also for step 5 and the perf gate: the shift-step bound is read "at `MAX_SEQ_LEN`", which on
mistral-7b is a 32768-token session (about 4 GiB of F16 KV on its own); the dense-model stress cases are
long runs, to be budgeted accordingly.

**Decision raised with the owner.**

1. *Item 3, option (b): per-sequence LongRoPE (short factors until a sequence crosses 4096, long after,
   cached K re-rotated at the crossing).* Recommendation: **decline (b) and keep the fixed 4096 cap**, with
   item 1's context shift made to fire at the 4096 limit for Phi-3.5 (finding above). Reasons: with shifting,
   a long Phi-3.5 conversation keeps going without ever using the long factors; (b) would also need the
   long-factor gap explained first (Juno 0.502 against the reference's 0.675 at matched long factors), since
   switching to factors Juno reads worse than the reference would trade a clear error for silently weaker
   output past 4096; and (b) adds a second re-rotation trigger to every Phi-3 rotation path (host, device,
   region). If declined, the long-factor gap needs no explanation (the item says so), and the first exit
   criterion closes on that decision. If taken, step 4 builds it on step 2's re-rotation machinery and
   explains or fixes the gap before it ships.

   **Owner decision (2026-10-08): option (b) declined.** The 4096 cap stays, and context shifting fires at
   4096 for Phi-3.5. Recorded in item 1's scope, implementation step 4 and the first two exit criteria; the
   first exit criterion is ticked.

**Out-of-tier changes.** None.

### 2026-10-08: implementation step 2 (context shift: KV, handlers, generation loop, surfaces)

**What shipped.** Context shifting as an opt-in, re-rotating the cached keys (not recomputing them).

| Layer | Change |
|---|---|
| KV (`kvcache`) | `SessionKvTensor.readToken`/`compact` (dense: one array copy; paged: `KvPageTable.compact` copies rows slot to slot with `KvBlockPool.copyToken` and returns freed pages); `KvContextShift.shift` moves both tensors and passes each moved K row through the caller's rotation, layers in parallel. Stored bytes are moved, not re-encoded; a q8_0 K row is re-encoded once after rotation |
| Rotation (`node`) | `RopeShift`: a pure rotation by `-discard * freq[i]` per pair, built from each handler's own config: `standard` (LLaMA family, both pair layouts), `partialSplitHalf` (Phi-2), `Phi3Rope.shift` (LongRoPE factors and frequency scale), `Qwen3Rope.shift` (YaRN ramp). Attention-magnitude factors are not reapplied, because the moved key already carries them |
| Handlers | `ForwardPassHandler.contextLimit()` and `shiftKv(...)` (default throws: fail closed). Implemented on Llama, Phi-2, Phi-3, Qwen3, Qwen3-MoE and the three LoRA handlers through `HandlerContextShift`; `Qwen2LoraTrainableHandler` and the vision decorator delegate (the latter refuses a request that carried an image). After the host shift every in-use device mirror is rewritten from the host rows (`DeviceKvCache.replacePrefix`, one upload per tensor, watermark set to the new length; retired on out-of-memory) and the KV adapter's copy is evicted. Phi-3's limit is `Phi3RopeConfig.positionLimit()` (4096 on Phi-3.5, owner decision 1) |
| Pipeline | `InferencePipeline.contextLimit()`/`supportsContextShift()`/`shiftKv(...)`; `LocalInferencePipeline` shifts each distinct handler once and reports the lowest limit. The cluster and tensor-parallel clients keep the defaults (cannot shift) |
| Generation (`coordinator`) | `ContextWindow`, one per request, used by `GenerationLoop.generate`, `.generateBatch` and each `ContinuousBatchEngine` slot: before any forward that would cross the limit it shifts (discard = the larger of the overflow and half of what follows the kept prefix) and records `juno.ContextShift`; it cuts an over-long prompt to the kept prefix plus the latest tokens; it refuses up front a pipeline that cannot shift, `--spec-type draft-simple`, and a kept prefix over half the limit. A session whose KV was shifted or cut drops it instead of offering it as a prefix |
| Surfaces | `--context-shift on\|off` (`ConsoleMain`, `run.sh local`; `run.sh cluster` and a cluster launch refuse `on`), `JUNO_CONTEXT_SHIFT` (`ContextShiftOptions`); `x_juno_context_shift` (chat completions) and `contextShift` (native), each answering 400 where shifting cannot work; `JunoPlayer.Builder.contextShift`, `JunoHttpClient.withContextShift`; `openapi.yaml`/`juno-api.yaml`. `./juno test` check 10 (`ContextShiftCheck`) |

**Tests.** Unless marked, written first and seen failing for the right reason (methods stubbed to throw, or
the behaviour absent), then passing:

| Test | Count | Result | Reading |
|---|---|---|---|
| `KvContextShiftTest` (kvcache) | 17 | pass | dense and paged, f16 and q8_0, filled to `MAX_SEQ_LEN`: a write at the cap still throws without a shift (4, regression guard, passed before the change); after a shift the prefix is untouched, moved V rows byte-exact, moved K rows rotated, writing continues to the cap; `KvPageTable` returns the pages past the new length |
| `RopeShiftTest` (node) | 5 | pass | rotation at p, then shift back by d, equals rotation at p - d within 2e-3, for standard adjacent, standard split-half, Qwen3 YaRN, Phi-3 LongRoPE with attention factor, Phi-2 partial |
| `LlamaTransformerHandlerContextShiftTest` | 6 | pass | dense and paged KV match the oracle KV to 1e-4; a negative control (keys moved without rotation) differs by more than 1e-3, so the tolerance sees a missing rotation; unknown request and unsupporting handler fail closed. The synthetic model needed a wider weight range (`newTestInstance(..., weightRange)`, default unchanged) for the control to bite |
| `ContextShiftLiveTest` (real models, CPU) | 9 + 1 opt-in | pass | max logit difference against the oracle: TinyLlama 3e-5, Qwen2.5-3B 3e-5, Phi-3.5-mini 6e-5 (and `contextLimit()` 4096), Qwen3-1.7B 3e-5, Phi-2 handler (moondream2 backbone) 2e-5; LoRA playback: TinyLlama with its trained adapter 3e-5, Qwen2.5-3B 2e-5, Phi-3.5 5e-5, Qwen3 4e-5; Qwen3-Coder-30B-A3B 2e-5 (opt-in, `-Djuno.test.largeModels=true` and a 40 GB heap: the model does not fit the default test heap). Same greedy token in every case. Written after the handler code it scores (the handler code was driven by the synthetic test above), passed on first run except Phi-2, see below |
| `LlamaTransformerHandlerContextShiftGpuTest` (CUDA TinyLlama) | 1 | pass | mirror watermarks equal the shifted length after the shift and grow on the next decode; max logit difference against the oracle attended on the host 0.146, against 0.160 for device-vs-host attention without a shift; same greedy token |
| `ContextShiftLiveGpuTest` (CUDA) | 2 | pass | Phi-3.5: no mirror retired, watermarks at the shifted length, difference 0.225 against a 0.159 control; Qwen3-1.7B: 0.127 against 0.125. Regression tests over the mirror code already driven by the TinyLlama GPU test; passed on first run |
| `LocalInferencePipelineContextShiftTest` | 2 | pass | each distinct handler shifted once, also when one handler serves two stages; limit is the lowest |
| `GenerationLoopContextShiftTest` (coordinator) | 10 | pass | default still fails at the limit (regression guard, passed before the change); opt-in shifts and keeps the system prompt, positions stay under the limit; an over-long prompt is cut only with the opt-in; server default and explicit `false`; static batch shifts one member; continuous slot shifts; a session after a shift restarts from an empty KV; unsupported pipeline, draft-model speculation and an over-long system prompt fail closed before any forward |
| `ContextShiftRequestFieldTest` (coordinator, HTTP) | 3 | pass | `x_juno_context_shift` and `contextShift` run past the limit (200, all tokens); absent, the request fails there (500); a deployment that cannot shift answers 400 naming the field |
| `VisionAwareForwardPassHandlerContextShiftTest` | 2 | pass | text-only delegates; an image request fails closed until evicted. **Written after the code; passed on first run** |
| `JunoHttpClientContextShiftTest` (juno-player) | 4 | pass | both body shapes carry the field; a plain client sends none; `JunoPlayer.request` carries the builder's choice. **Written with the code; passed on first run** |
| `ContextShiftCheckTest` (juno-master, check 10, CUDA TinyLlama) | 1 | pass | 192-position limit: 200 tokens generated on each schedule, 3 shifts each, highest position 191 |
| `ModelLiveRunnerTest` | 8 | pass | suite selection now includes check 10 for `pipeline` and `tensor` |
| `Phi3EndOfTurnLiveTest`, `Phi3RopeFactorPolicyTest`, `Phi3RopeLoadTest` | 7 | pass | unchanged (step 1) |
| `ContextWindowKeepTest` (coordinator) | 6 | pass | the kept prefix holds the system prompt on six chat templates. Seen failing on Mistral and Phi-3 (kept prefix 0: those templates fold the system text into the first user turn, so the system messages formatted alone render none of it); fixed by formatting the system messages followed by an empty user turn |
| `PrefixCacheDeepInvalidateTest` (kvcache) | 1 | pass | a 20000-token prefix invalidated on a virtual thread; another key's prefix survives. Seen failing with `StackOverflowError` |
| `Phi3ContextShiftAtLimitTest` (juno-master, CUDA Phi-3.5) | 1 | pass | 3994-token prompt held open for 200 tokens crosses 4096 with the opt-in; without it the same request fails with the original-context-length error. Written after the code; passed on first run |
| `smoke-context-policy.sh` (new; Phi-3.5, CUDA, 16 GiB heap) | 7 checks | pass | both schedules: with the opt-in all 10 turns answer (turns 7 to 10 past the limit, their prompts cut to 2058 tokens); a request held open past the limit shifts during decode (4238 positions); without the opt-in turn 7 fails at 4096 with the documented error; `--context-shift on` answers the crossing turn by default and fails it with an explicit false. Run `target/context-policy-smoke/20261009T012313Z/` (not published: a smoke result, not a measurement) |

**Full unit reactor** (`mvn test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,vision,metrics,juno-player`):
registry, lora, kvcache, health pass; node 993 run, 1 failure, 45 skipped. The failure is
`PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance`, a device-wide free-VRAM check
(reads 84 MB free with the card filled, bound 64 MiB). **It fails identically on a clean export of HEAD
`de98ff6`** (72 MB), so it is environmental on this host today, not this change; reported, not loosened. The
failure skipped the later modules, which were then run with `-fae`: tokenizer, sampler, vision, metrics pass;
coordinator showed only the new `ContextWindowKeepTest` failing (the seen-failing run above); after the fix
coordinator (396) and juno-player (122) pass. kvcache re-run after the `PrefixCache` fix: 96 pass. Then
`mvn verify -pl juno-master`: 10 unit tests and 20 ITs pass.

**Defects found while doing this step.**
- *Fixed (in scope: context shift calls it on the request path):* `PrefixCache.invalidate` recursed once per
  cached token (three frames each) and overflowed a virtual thread's stack for a session of about 4000 tokens.
  Pre-existing: ending such a session (`evictSession`) would have hit it too. Now iterative.
- *Fixed by owner decision 4 (out-of-tier change below):* a stateless request that fails mid-generation was never evicted
  (`GenerationLoop.generate` evicts only on the success path), so its KV stays held for the life of the
  process. On Phi-3.5 near 4096 tokens that is about 3 GB of heap per failed request; the first smoke run ran
  out of heap on the request after such a failure. Pre-existing, outside this tier's scope (decision 4).
- *Handed to Tier 08 item 7 (owner decision 3):* `models/phi-2.Q4_K_M.gguf` cannot be loaded at all (separate Q/K/V tensors; the
  handler reads only a fused one). Recorded in `INVENTORY.md`; the Phi-2 handler was scored on moondream2's
  backbone instead (decision 3).

**Measurement boundary.** None for a request that does not opt in: with shifting off, `ContextWindow.place`
returns the position unchanged (one subtraction and a comparison per forward), and no forward, MatVec, KV write
or attention path changed. The tier's perf gate still has to show it (step 5, owner's pinned run).

**Decisions raised with the owner.**

2. *Context shift in cluster and tensor-parallel mode.* Implemented as FAIL-CLOSED (startup error for the flag,
   HTTP 400 for the request field), recorded under "Out of scope" pending your confirmation. Doing it for real
   needs a node RPC that shifts a request's KV on every node, and the cluster clients do not yet forward even
   `evict`. Recommendation: **confirm fail-closed for this tier**, and give the RPC to the tier that owns the
   cluster surface (Tier 13), which also owns the missing cluster `evict`.
3. *`phi-2.Q4_K_M.gguf` does not load.* Recommendation: **give it to Tier 08** (architecture breadth: teach the
   Phi-2 handler the split Q/K/V layout), and until then stop counting the file as a Phi-2 test model.
4. *Failed stateless requests leak their KV.* Recommendation: **fix it as an out-of-tier change now** (evict in
   a `finally` on the stateless path of `generate`, `generateBatch` and the continuous engine), recorded under
   this tier's out-of-tier heading, because context shifting makes long requests near the limit more likely
   and each leaked one costs gigabytes.

**Owner decisions (2026-10-08).** 2: fail-closed confirmed; the cluster shift RPC and the missing cluster
`evict` are [Tier 13](TIER-13-server-surface-clustering.md) item 5. 3: the split-Q/K/V Phi-2 file is
[Tier 08](TIER-08-model-architecture-breadth.md) item 7. 4: fix the failed-request KV leak now, as an
out-of-tier change (below).

**Out-of-tier changes.**
- *2026-10-08, owner decision 4: a request that fails mid-generation releases its KV.* `GenerationLoop.generate`
  and `.generateBatch` now run their bodies through a wrapper that, on failure, evicts a stateless request's
  KV (every member of a failed static batch), drops a session's KV when the request had context shifting on
  (it may no longer line up with the session's prompt), and evicts the draft model's KV; a session that never
  opted in keeps its KV as before. `ContinuousBatchEngine` does the same on a failed admit, a failed slot
  completion and a failed engine step (which fails every running slot). A failure while releasing is attached
  to the original exception. Touches the coordinator's request lifecycle only; no forward, MatVec, KV write or
  attention path, so not a measurement boundary and no published baseline is affected.
  `GenerationFailureEvictionTest` (3: single request, static batch, continuous slot) seen failing first (the
  KV stayed held), then passing.
- (`LlamaTransformerHandler.newTestInstance` gained a weight-range overload for tests; the default draws
  exactly the weights it drew before. Not a product change.)

**Next:** implementation step 3, sliding-window attention (both window keys, synthetic uniform and patterned
fixtures, CPU and GPU).


### 2026-10-09: implementation step 3 (sliding-window attention)

**Plan check.** `check-plan-thresholds.sh` passes (22 tier files). **Plan-versus-code check**: no window key was read
anywhere; the GPU kernel already took a per-call window, and `GqaAttentionTiledTest` already held it (window 1 to 500,
off-tile, deep into a long prompt); every other claim of step 3 held. Two claims did not (corrected in step 3 above,
dated): the only patterned file stores its pattern as a per-layer boolean array, not a period, and Phi-3.5-mini
declares a uniform 262144-token window. Read off every model file's metadata keys with `./juno gguf-info`, no tensor
data read.

**What shipped.**

| Layer | Change |
|---|---|
| Metadata (`node`) | `SlidingWindow`: width from `<arch>.attention.sliding_window` (absent or 0: none); layers from `<arch>.attention.sliding_window_pattern`: absent = every layer, boolean array of `block_count` entries (`true` = windowed), or integer period `N` (layer `i` global when `i % N == N - 1`; `N = 1` all global, `N = 0` all windowed). Another type, a wrong-length array, a negative width or period: `UnsupportedModelException` naming the key, at load. `forShard` reads the pattern at global layer indices |
| CPU attention | `GqaMath.attend(..., window)`: keys `[firstKey(seqLen, window), seqLen)`, the kernel's contract. Every handler's private copy (Phi-2, Phi-3, Qwen3, Qwen3-MoE, LoRA inference) now delegates to it; each copy ran the same arithmetic, so window 0 is bit-identical (`GqaMathWindowTest` against a copy of the old loop) |
| GPU attention | `layerWindows[li]` passed to `CudaGqaAttention.attendBatched` (four Llama sites), `GpuAttentionMirror.attendWindow`/`attendOne`/`attendStreams` (Phi-3, Qwen3), and through `Qwen3AttentionWeights.attentionWindow()` (abstract, so the MoE handler cannot inherit 0); `ResidentQkvPath` and `PrefillWindowRegion` take the array once (`withWindows`) and pass each layer's window to `GqaAttentionKernel.launch` (both passed 0 before) |
| Handlers | Llama, Phi-2, Phi-3, Qwen3, Qwen3-MoE and the three LoRA handlers read `SlidingWindow.forShard(r, ctx, log)` at load (logged when declared); Qwen2Lora and the vision decorator delegate |
| LoRA training | the four inline training attention loops start at the window's first key, so the stored attention weights are 0 outside it and the backward pass sends those keys no gradient |
| KV | unchanged: every row is stored; the window narrows what attention reads |

**Tests.** Written first and seen failing against stubs that threw (19 of 19), then passing, unless marked:

| Test | Count | Result | Reading |
|---|---|---|---|
| `SlidingWindowTest` (synthetic metadata GGUFs, `MetadataOnlyGguf` widened to typed keys) | 9 | pass | (a) uniform, (b) boolean pattern names layers 5, 11, ..., 41 global on a Gemma-shaped 42-layer file, integer period 6 names the same layers, (c) width without pattern is uniform, absent width is none whatever the pattern, (e) a shard of global layers 10..17 gets local `[512, 0, 512, 512, 512, 512, 512, 0]`; six malformed shapes refused naming the key. Real headers: Phi-3.5-mini 262144 on every layer; gemma-4-E4B 512 with layers 5, 11, 17, 23, 29, 35, 41 global; Mistral 7B v0.1 none |
| `GqaMathWindowTest` | 3 | pass | (d) window 0, window = seqLen and window 262144 bit-identical to the old loop, seqLen 1..40; a window equals attending over its last rows alone, bit for bit; NaN in the rows before the window leaves the output unchanged |
| `LlamaTransformerHandlerSlidingWindowTest` (synthetic, period 2 over 4 layers) | 7 | pass | overwriting the rows before the window: layers 0 and 2 (windowed) logits bit-identical, layers 1 and 3 (global) moved, dense and paged KV; prefill equals token-by-token decode to 1e-4 and differs from no window; multi-stream decode equals single decode; a 4096 window bit-identical to none (prefill and decode); a shard of global layers 1..2 gets `[0, W]` |
| `SlidingWindowLiveTest` (real models, CPU, forced period-2 window of 6, 24-token prompt) | 9 (1 opt-in skipped) | pass | windowed-layer overwrite moves the logits by exactly 0 and a global-layer overwrite by 13.1 to 36.9 on TinyLlama, Qwen2.5-3B, Phi-3.5-mini, Qwen3-1.7B, the Phi-2 handler (moondream2 backbone), LoRA playback on all four families; LoRA training: windowed training loss equals windowed inference loss to every printed digit (TinyLlama 267.62387, Phi-3.5 386.19543, Qwen3 375.97375; no window 245.4 to 250.9). Qwen3-MoE skipped (opt-in, large heap). Written after the code |
| `SlidingWindowGpuTest` (CUDA, same forced window of 8, 48-token prompt) | 5 | pass | decode and multi-stream decode, device against host attention in the same handler: 2.7% to 5.0% of the window's effect (bound 10%); prefill GPU against CPU at widths 7, 8, 9: width 8 closest on every model, by 2.17x (Qwen2.5-3B) to 7.3x (TinyLlama) (bound 1.5x). GPU attention and the prefill region active on every model; TinyLlama again with both regions off. Written after the code; see the finding below |
| `GpuAttentionHandlerParityTest`, `ContextShiftLiveGpuTest`, `LlamaTransformerHandlerContextShiftGpuTest`, `GqaAttentionTiledTest`, `GqaAttentionKernelParityTest` | 16 | pass | regression: the observer now carries the window and the parity oracle applies it |

**Finding: the GPU test's first bound was wrong, and was replaced, not loosened by hand.** The first version borrowed
the context-shift GPU tests' bound (`2 * control + 0.02`, control = the same comparison without a window) and failed 5
of 5. The windowed differences were 0.21 to 3.6 against windowed-versus-unwindowed effects of 7.6 to 25, so no path
ignored the window; but an 8-key window averages each row over fewer FP16 keys and moves the logits far from their
unwindowed values, so rounding grows with it. A tenth of the effect then passed decode on every model but failed prefill
on three (12% to 15%: rounding compounds over 48 tokens and 28 to 36 layers), which is also what an off-by-one could
look like. A diagnostic settled it before any bound changed: GPU prefill under width 8 against CPU prefill under 7, 8
and 9 (Qwen2.5-3B 2.18 / 1.01 / 2.62; Qwen3-1.7B 9.07 / 3.11 / 14.02), so the GPU applies exactly 8. The prefill
criterion is now that discrimination (width W closest by >= 1.5x), which fails for an ignored window and an off-by-one
alike. The diagnostic class was deleted after use.

**Measurement boundary.** None for a model that declares no window: `GqaMath` with window 0 is the old arithmetic bit
for bit, and every GPU launch receives the 0 it received before. Phi-3.5-mini now reads 262144, which reaches the kernel
and `GqaMath` as a window wider than any context it can reach (4096 cap), so the same keys are read. The tier's perf
gate (step 5) still has to show it: prepared as `dist/context-policy-close/run-gate.sh` parts A (GPU, four sweep
models, 512) and C (CPU and allocation), baseline `de98ff6` (`620a5caf606538b6`), started by the owner's waiter
(`wait-and-run.sh`) once `READY` is written. The shift-step latency bound is not in it (no harness yet; step 5).

**Cross-surface (sliding windows; the context-shift half is in step 2's record).** 1 CPU PASS (`SlidingWindowLiveTest`);
2 CUDA PASS (`SlidingWindowGpuTest`); 3 ROCm NEEDS-AMD-HARDWARE (attention runs on the CPU path there, which applies
the window); 4 static and 5 continuous PASS (dense and paged KV, multi-stream decode); 6 local PASS; 7 pipeline: shard
indexing PASS at unit level (`SlidingWindowTest`, `LlamaTransformerHandlerSlidingWindowTest`), each node reads its own
file at load; 8 tensor-parallel: every node runs the full model through the same handlers (`TensorShardContext`), so
global indices apply; no windowed cluster run (no real windowed model loads before Tier 08); 9 LoRA training PASS; 10
LoRA playback PASS; 11 vision N/A (moondream2 declares no window; the decorator delegates); 12-15 N/A (no flag, field or
facade setting: the window is model metadata). Rows 7, 8 and 12 to 15 are to be confirmed in step 5's matrix.

**Out-of-tier changes.** None.

**Next:** step 5: the cross-surface smoke matrix with a long-context stress case on both schedules, the shift-step
latency harness, and scoring the owner's gate.

### 2026-10-09: the owner's pinned no-regression gate (parts A and C)

Run by the owner through `dist/context-policy-close/wait-and-run.sh`, started automatically when step 3's tests had
passed and `READY` was written (04:35 to 06:56). Published: `docs/perf-compare/20261009T151747Z-context-policy-close-gate`
(baseline `de98ff6`, `620a5caf606538b6`; candidate `0f87603` plus step 3, `b12fbfb34a342e70`).

| Lane | Model | pp B/A | tg B/A | alloc/token B/A | GC ms A / B | Greedy equal |
|---|---|---|---|---|---|---|
| GPU, 512 | TinyLlama | 0.998 | 1.005 | 1.000 | 0.0 / 0.0 | 3 of 3 |
| GPU, 512 | Qwen2.5-3B | 0.999 | 0.999 | 1.000 | 0.0 / 0.0 | 3 of 3 |
| GPU, 512 | Phi-3.5-mini | 0.983 | 0.997 | 1.000 | 11.9 / 11.5 | 3 of 3 |
| GPU, 512 | Mistral 7B | 0.987 | 0.978 | 0.998 | 0.0 / 0.0 | 3 of 3 |
| CPU, 128 | TinyLlama | 0.993 | 0.999 | 1.000 | 7.7 / 7.6 | 3 of 3 |
| CPU, 128 | Mistral 7B | 1.001 | 1.003 (1 run per side) | 1.000 | 0.0 / 0.0 | 3 of 3 |

Every reading meets its threshold (tg and pp >= 0.95x, allocation <= 1.10x, GC <= 5 ms under a 5 ms baseline or
<= 1.25x, greedy output equal). Hot methods (CPU, top 10): same kernels, same order on both builds.

**Defect in the gate script, found in scoring (fixed).** The script printed `FAIL: mistral generation below 0.95x` and
`GATES MISSED`. The harness's span check had withheld Mistral 7B's CPU generation reading on two of three runs per side
(JFR timestamps about 89 ms off the engine clock on a 137-second request; `token_gen_tps: null`), and the scorer, copied
from the previous gate's script, turned each `null` into 0, so both medians were 0 and the ratio 0/0. The scorer now
takes the median over the readings that exist and reports the withheld ones; the saved runs were rescored without a
rerun (the table above). The previous gate's script has the same scorer; its published readings had no withheld
repetition (all its ratios are non-zero), so nothing published earlier is affected.

**Decision raised with the owner.**

5. *Mistral 7B CPU generation is one scorable run per side, not a median of three.* The engine-timed reading is
   0.5193 to 0.5207 t/s (1.003x); the request-timed rate, which the span check does not withhold, is a full median of
   three, 0.4739 to 0.4748 t/s (1.002x); prefill 1.001x; and the CPU path's only change is one integer comparison per
   attention call when no window is declared. Options: (a) accept the request-timed median of three as the reading for
   this row, recorded as such; (b) re-run part C on Mistral 7B alone (about 2 hours pinned, the span check may withhold
   again on a run this long). Recommendation: **(a)**.

   **Owner decision (2026-10-09): (a).** Mistral 7B's CPU generation row is scored on the request-timed rate, median
   of three: 0.4739 to 0.4748 t/s, **1.002x**, met. Parts A and C of the gate are met on every row.

**Next:** step 5 (cross-surface matrix with the long-context stress case on both schedules; the shift-step latency
harness, run by the owner as part D through the same waiter).

### 2026-10-09: implementation step 5 (cross-surface matrix, shift-step latency, closing gates)

**Plan check.** `check-plan-thresholds.sh` passes.

**Shift-step latency: a harness, a miss on the GPU, and the fix.** No harness existed for the bound (the decode step
that performs a shift `<= 3.0x` the median decode step at the same depth, at `MAX_SEQ_LEN`). Prefilling 32,768 tokens
on the CPU is not practical (Mistral 7B prefills at 0.87 t/s: about 10 hours per repetition), and what a step costs
does not depend on what the cache holds, so `ContextShiftStepBench` (`node`, a `main` like the other microbenches)
prefills a 32-token system prompt, fills the host KV directly to the limit, rebuilds the device mirror from it
(`LlamaTransformerHandler.rewriteDeviceKv`, benchmarks only), times five decode steps ending at the limit, then the
shift the generation loop performs (keep the prompt, discard half of what follows) plus the next decode step.
`scripts/performance-tests/context-shift-step-bench.sh` drives it per model and backend, pins clocks on request and
scores the median of three. First reading, 32,768 positions, unpinned, one repetition:

| Model | Backend | Median decode | Shift | Shift step | Ratio |
|---|---|---|---|---|---|
| TinyLlama | CPU | 5,217 ms | 84 ms | 2,847 ms | 0.55 |
| TinyLlama | GPU | 47.8 ms | 754 ms | 781 ms | **16.3** |
| Mistral 7B | CPU | 16,786 ms | 471 ms | 9,819 ms | 0.59 |
| Mistral 7B | GPU | 3,801 ms | 2,937 ms | 4,867 ms | 1.28 (the mirror does not fit at 32,768 next to the weights, so decode already attends on the host; the shift spent about 2.5 s trying to rebuild the mirror before retiring it) |

The cost was the device mirror rebuild: every row converted to FP16 in Java and re-uploaded, layer by layer. Per the
threshold's own instruction (above the bound, move the re-rotation off the decode step), the shift now runs on the
device: `DeviceKvCache.shiftInPlace` moves the kept rows down with device-to-device copies (chunked by the shift
distance, so overlapping ranges move intact) and `KvShiftKernel` (`kv_shift.cu`) rotates the moved FP16 K rows in place
with `RopeShift.cosSin`, the table the host shift applies (a shift rotates every moved key by the same distance, so one
per-pair table serves every RoPE variant, and no magnitude factor can be re-applied). Each product and sum is rounded
separately, so the device result equals the host rotation rounded to FP16. Nothing is allocated but the table.
`HandlerContextShift.shiftMirrors` replaces the rebuild in the Llama, Phi-3 and Qwen3 handlers and falls back to it for
a mirror short of the full history or without the kernel. After the fix, same conditions: TinyLlama GPU shift 754 ms
to 101 ms, ratio **2.54**; Mistral 7B GPU shift 2,937 ms to 495 ms, ratio 0.63. The remaining TinyLlama cost is the host
shift (moving about 1.5 GB of float rows), near memory bandwidth; a ring-buffer KV layout that would avoid moving rows
is Tier 03's territory. The pinned reading is part D of the owner's final gate.

**Tests (step 5 additions).**

| Test | Count | Result | Reading |
|---|---|---|---|
| `DeviceKvShiftTest` (CUDA) | 4 | pass | adjacent pairs, split-half with arbitrary per-pair frequencies, partial rotation, a discard shorter than the moved rows: K and V bit-identical to the host shift rounded to FP16, watermark at the new length. Written first, seen failing against a stub that threw |
| `ContextShiftLiveGpuTest`, `LlamaTransformerHandlerContextShiftGpuTest` | 3 | pass | regression through the new path: shifted request against the oracle KV: Phi-3.5 0.226 (control 0.159), Qwen3 0.161 (0.125), TinyLlama 0.158 (0.160), each within `2 * control + 0.02` |
| Unit reactor (11 modules) | 2,256 | 1 environmental failure | node 1,030 run, 1 failure, 46 skipped: `PrefillRegionHandlerParityTest`'s device-wide free-VRAM check (case 3 short by 12 MB, re-run alone: case 1 short by 0.4 MB). **It fails the same way on the pre-tier tree `de98ff6`, twice in two runs**, so environmental, not this change; reported, not loosened. The modules it skipped were run after: tokenizer 109, coordinator 399, vision 97, juno-player 122, all pass |
| `mvn verify -pl juno-master` (after `mvn install`) | 10 + 20 | pass | ThreeNodeClusterIT, TensorParallelClusterIT, InProcessClusterIT, UnsupportedArchitectureClusterIT |
| Earlier smoke scripts, unmodified | 188 checks | pass | `smoke-consistency.sh` 54, `smoke-grammar.sh` 19, `smoke-tools.sh` 14, `smoke-context-policy.sh` 7 (the long-context stress case: conversations past the 4096 limit on both schedules, with and without the opt-in), `smoke-long-prompt-prefill.sh` 36, `smoke-packed-kquant-matmul.sh` 48, `smoke-gpu-residency.sh` 10 (run with `--models tinyllama...,mistral...`: its default list also loads llama-1-30b, a 3-hour run). No FAIL line in any. Outputs under `target/` (smoke results, not measurements) |

**Gates read in this step.**

| Gate | Reading | Threshold | Result | Where |
|---|---|---|---|---|
| LoRA train / playback, unpinned, median of three, both builds | 0.932x / 1.067x | `<= 1.25x` / `>= 0.80x` | met | `docs/perf-compare/20261009T192344Z-context-policy-lora-vision` |
| Vision latency / decode, unpinned, median of three, alternating | 0.996x / 1.026x | `<= 1.25x` / `>= 0.80x` | met | same |
| GPU and CPU no-regression, allocation, greedy output (parts A and C, pinned, owner) | see the gate entry above | | met | `docs/perf-compare/20261009T151747Z-context-policy-close-gate` |
| Shift step, pinned, median of three; GPU no-regression re-run on the final jar; reference-relative sweep at 512 | owed | `<= 3.0x`; `>= 0.95x` | running (owner, started 14:24 via `dist/context-policy-final/wait-and-run.sh`) | |

**Cross-surface checklist (both items).**

| # | Surface | Context shift | Sliding window |
|---|---|---|---|
| 1 | CPU | PASS (`ContextShiftLiveTest`, bench CPU lanes) | PASS (`SlidingWindowLiveTest`) |
| 2 | CUDA | PASS (GPU shift tests, `DeviceKvShiftTest`, smoke on Phi-3.5) | PASS (`SlidingWindowGpuTest`) |
| 3 | ROCm | NEEDS-AMD-HARDWARE (KV-level; attention on the CPU path there; the device shift is CUDA-only and falls back to the host rebuild) | NEEDS-AMD-HARDWARE (CPU attention path applies it) |
| 4 | Static | PASS (`GenerationLoopContextShiftTest`, smoke static) | PASS (dense KV, multi-stream decode) |
| 5 | Continuous | PASS (slot shift, smoke continuous) | PASS (paged KV) |
| 6 | Local | PASS | PASS |
| 7 | Pipeline cluster | FAIL-CLOSED (owner decision 2; startup error and HTTP 400 tested) | PASS at unit level (shard reads global layer indices; each node reads its own file); no real windowed model loads before Tier 08 |
| 8 | Tensor-parallel | FAIL-CLOSED (same) | PASS by construction (every node runs the full model through the same handlers); same caveat |
| 9 | LoRA training | N/A (training does not use context shift) | PASS (training loss equals windowed inference loss) |
| 10 | LoRA playback | PASS (`ContextShiftLiveTest` LoRA cases) | PASS |
| 11 | Vision | FAIL-CLOSED for a request carrying an image (tested); text-only requests shift | N/A (moondream2 declares no window; decorator delegates) |
| 12 | OpenAI REST | PASS (`x_juno_context_shift`, smoke) | N/A (model metadata, no request field) |
| 13 | Native REST | PASS (`contextShift`) | N/A |
| 14 | CLI | PASS (`--context-shift`, smoke) | N/A (no flag) |
| 15 | Facade | PASS (`JunoPlayer.Builder.contextShift`, `JunoHttpClient.withContextShift`) | N/A |

**Measurement boundary.** The in-place device shift runs only when a request shifts; no forward, MatVec, attention or
KV-write path changed for a request that does not opt in. Part A of the final gate re-runs the GPU no-regression on
the final jar to show it.

**Out-of-tier changes.** None.

**Next:** score the owner's final gate (parts A, D, L); then tick the perf-gate, docs and CHANGELOG boxes and close the
tier, or record what missed.

### 2026-10-09: the owner's final gate (parts A, D, L) and close

Run by the owner through `dist/context-policy-final/wait-and-run.sh` (14:24 to about 15:20), on the final jar
`16f8794d833b1dff` against `de98ff6`. Published: `docs/perf-compare/20261009T195800Z-context-policy-final-gate` (A and D)
and `docs/perf-compare/20261009T195529Z` (L, published by hand: the harness does not publish a `--juno-jar` run; the jar
is the tree's own build).

- **Part A (GPU no-regression, pinned):** tg 0.992x to 1.031x, pp 0.990x to 1.000x, allocation 0.994x to 1.004x, GC
  unchanged, greedy output equal on every run. Met.
- **Part D (shift step at 32,768, pinned, median of three):** TinyLlama CPU 0.541, GPU 2.362; Mistral 7B CPU 0.581, GPU
  0.630. Met. The tight lane is TinyLlama GPU, whose remaining cost is the host shift (about 90 ms).
- **Part L (reference-relative, GPU, `n_prompt=512`, pinned):** pp 0.735 (TinyLlama), 0.779 (Qwen2.5-3B), 0.584
  (Phi-3.5-mini, binding), 0.802 (Mistral 7B); tg 0.738, 0.412, 0.827, 1.014. Against the active end-of-plan rows: GPU
  tg every model `>= 0.70x` (Qwen2.5-3B 0.412, binding, unchanged: this tier does not touch it; Tier 02D's); Phi-3.5
  tg `>= 0.90x` (0.827); GPU pp at 512 `>= 0.70x` (0.584, Phi-3.5 binding). None moves in this tier, as expected of a
  tier with no throughput target; the reading stays a reading and the reference column is unchanged. Milestone rows for
  02B: none.

**Tier closed.** Every exit criterion checked. **Next tier:** [Tier 02C](TIER-02C-cpu-hot-path.md).
