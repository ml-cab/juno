# Tier 04B: Tokenizer fidelity

Status: not started
Gap analysis refs: none — the gap analysis has no tokenizer section, and no other tier in this plan
owns the `tokenizer` module. See "Why this tier, why now".

## Objective

Make Juno's tokenization match what a model's GGUF actually specifies, instead of applying one
BPE strategy to every file regardless of what the file says. Concretely: implement the pre-tokenizer
split each declared pre-type requires, and prove per model family that Juno's token IDs for a given
string match the reference tokenization for that family.

**Read this before starting: the first half of this tier already shipped in Tier 01.** Reading
`tokenizer.ggml.pre`, dispatching on it, and failing closed on an unimplemented pre-type were this
tier's items 1 and 3. They are now precondition 7 of the benchmark-parity work in
[`README.md`](README.md), landed by [Tier 01](TIER-01-gpu-activation-residency.md), because
`prompt_tokens` is the denominator of every pp ratio and the context depth of every tg ratio — so
leaving the fix here would have made this tier a second measurement boundary crossing five tiers of
already-published gates, including Tier 01B's `pp >= 0.10x` milestone. This tier's own argument is
what moved them: it says precondition 1's 10% tolerance "is a guard against the symptom. This tier
removes the cause," and a cause of a measurement error belongs with the measurement corrections.

What remains here is the deeper half, and it is still real work: the per-family split
implementations beyond the minimum the four sweep models needed, the cross-engine token-ID parity
evidence, and the LoRA-adapter provenance decision. Start by reading Tier 01's recorded enumeration
of declared pre-types and what it actually implemented, rather than re-deriving it.

## Why this tier, why now

Before this tier was written, the `tokenizer` module appeared in this plan exactly twice, both
times inside a `mvn test -pl ...` command. Nothing owned it. That is a gap in two directions at
once:

**Correctness.** `GgufTokenizer` pre-splits on special tokens and then runs BPE merges over the
whole remaining segment. As this tier was written it never read `tokenizer.ggml.pre`, and a repo-wide
grep found no reference to that key anywhere. Real GGUF files declare a pre-tokenizer type
(`llama-bpe`, `gpt-2`, `deepseek-llm`, `qwen2`, and others), and each one specifies a regex split
applied *before* merging — the split that decides whether `"1234"` is one token or four, whether
`"don't"` splits at the apostrophe, and how runs of punctuation and leading spaces group. Skipping it
does not throw; it silently produces a different token sequence from the one the model was trained
against. That is precisely the silent-degrade shape Tier 00 spent its effort removing from the
architecture path, still present one module over.

**Measurement — and this is why half of this tier moved.** Every llama.cpp-relative ratio this plan
collects divides a Juno throughput by a llama.cpp throughput for "the same prompt." If the two engines
tokenize that prompt differently, they are not processing the same number of tokens, and the ratio
silently absorbs the difference. The benchmark-parity preconditions in [`README.md`](README.md) require
Juno's `prompt_tokens` to be within 10% of `n_prompt`; that check is a guard against the symptom, and
removing the cause is a measurement correction, not a feature. So the key read, the dispatch and the
fail-closed path went to Tier 01 as precondition 7. The consequence for this tier is that it starts
from a codebase that already reads the key for the pre-types the sweep models declare, and its job is
the rest of the families plus the evidence.

Sequenced here because what remains needs nothing from Tiers 00-04 and nothing later depends on it,
but it must land before Tier 08 adds four new architecture families — each of which brings its own
pre-tokenizer type. Tier 01 implemented the types the four sweep models declare, not every type;
Tier 08's families may well declare others, and adding handlers for them against a pre-tokenizer that
only covers the sweep set would bake fresh divergences in. Tier 01's fail-closed path means such a
file is *rejected* rather than silently mistokenized, which is the safe failure but still a failure —
this tier is what makes those files work.

## Scope

### In scope

1. ~~**Read `tokenizer.ggml.pre` and dispatch on it.**~~ **Shipped by
   [Tier 01](TIER-01-gpu-activation-residency.md) as benchmark-parity precondition 7** — the metadata
   read, the pre-tokenizer abstraction, and the dispatch, for the pre-types the four sweep models
   declare. This tier verifies that rather than re-implementing it, and reads Tier 01's recorded
   enumeration of declared types across `models/` as the starting inventory. If Tier 01's abstraction
   needs widening to take the families below, widen it — do not add a second dispatch.
2. **Implement the regex split for every pre-type on disk that Tier 01 did not need.** Tier 01
   shipped `qwen2` and `llama-bpe` and recorded the full enumeration; three declared types on disk
   are still unimplemented and their files are refused at load: **`tekken`**
   (`Devstral-Small-2-24B`), **`minimax-m2`** (`minimax-m2.5-tiny`) and **`qwen35`**
   (`Qwen3.5-0.8B`) — the same three architectures Tier 08 adds handlers for, which is why this
   tier runs before it. A fourth case is not a refusal but a remaining divergence: a byte-level BPE
   vocabulary that declares `default` or nothing (`phi-2`, `moondream2`) keeps Juno's whole-run
   merge, where the reference applies a four-expression default split. Widen Tier 01's
   `BpePreTokenizer` rather than adding a second dispatch, and note that its single-pattern shape
   has to grow to a sequence of patterns for that default case. Re-run `./juno gguf-info` across
   `models/` and diff against Tier 01's enumeration first — files have been added between tiers
   twice already.
3. ~~**Fail closed on an unknown pre-type.**~~ **Shipped by Tier 01 as part of precondition 7** — a
   file declaring an unimplemented pre-type is rejected at load with an error naming it, and a file
   with no key keeps the pre-existing behaviour. This tier's job is to *shrink the set that hits that
   rejection* by implementing item 2, and to keep a regression test proving the rejection still fires
   for a type nobody has implemented yet. Re-verify both halves against the current code before
   trusting this line.
4. **Cross-engine token-ID parity evidence.** For each sweep model, record Juno's token IDs for a
   fixed corpus and compare against the reference tokenization for that model. Where a divergence
   remains after item 2, document it per model rather than leaving it unstated. This is the item that
   turns Tier 01's "the token counts match" into "the tokens match", and it is the substantive half of
   what is left here: a count can agree while the split underneath it does not.
5. **SentencePiece encoding in better than quadratic time** *(added 2026-10-04 by owner decision during
   Tier 01C step 6)*. `GgufTokenizer`'s SentencePiece path merges over the whole unsplit text
   (`mergeWholeRuns` / `mergeInPlace`), so encoding is quadratic in prompt length. Measured on the
   comparison harness's prompt (`x x x ...`): TinyLlama 4.3 / 23.7 / 262 ms, Phi-3.5-mini 0.9 / 13.5 /
   270 ms, Mistral 7B 0.8 / 13.7 / 253 ms at 128 / 512 / 2048 words, against 0.8 / 2.6 / 5.0 ms for
   Qwen2.5-3B's pre-split BPE. It enters time to first token, about 0.25 s at 2,048 tokens and several
   seconds at 8,192 (x16 per x4). Fix with a priority queue over adjacent pairs in the same
   highest-score, leftmost-first order, held to identical tokens against the current implementation over
   a corpus that includes the sweep prompts, with time to first token measured before and after.
   `juno.PromptEncode` (shipped in Tier 01C) times encoding per request, so the gain is readable from
   any `compare-llama-cpp.sh` run.

### Out of scope

- Tokenizer *performance*. `juno.Tokenizer.encode` spans already exist and encode time is not a
  measured bottleneck; this tier is about producing the right tokens, not producing them faster.
  If the pre-tokenizer regex turns out to be hot for long prompts, record the measurement and file
  it rather than optimizing here.
- Chat-template correctness (`ChatTemplate`, `MiniJinjaTemplate`) — a separate concern, already
  partly covered by Tier 08's template work and Tier 13's tool-calling templates.
- Adding new tokenizer model types (WordPiece, Unigram) beyond what files on disk declare.

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | tokenization is backend-agnostic; primary correctness surface |
| 2 | CUDA GPU inference | N/A — no backend interaction; verify token IDs are identical to row 1 |
| 3 | ROCm GPU inference | N/A, same reason |
| 4 | Static schedule | N/A — tokenization happens before scheduling |
| 5 | Continuous schedule | N/A, same reason |
| 6 | Single-node local mode | primary dev/test surface |
| 7 | Pipeline-parallel cluster | tokenization happens coordinator-side; confirm nodes never re-tokenize |
| 8 | Tensor-parallel cluster | same |
| 9 | LoRA training | **a real risk, and the decision still lives here even though the first tokenization change shipped in Tier 01.** Training tokenizes its corpus, so a pre-tokenizer change alters the token sequence a trained adapter was fitted to and an adapter trained before that change may no longer compose correctly with a base model tokenized after it. Tier 01 recorded the finding it was asked for: its change did **not** move TinyLlama's tokenization, so no adapter on disk crossed that boundary and the question arrives here without a live instance. Decide and document anyway before this tier moves any tokenization an adapter could have been trained under: re-train, version the `.lora` format with the pre-type it was trained under, or accept and warn |
| 10 | LoRA playback | same consideration, but **Tier 01 settled the instance**: `tinyllama-...Q4_K_M.lora` was trained against a SentencePiece base with no declared pre-type, whose tokenization is byte-identical across Tier 01's change, so that adapter is not on the far side of a boundary. The decision stays open and becomes live only if this tier changes a tokenization some adapter was trained under |
| 11 | Vision | vision splices image tokens at `<image>` positions — confirm the pre-tokenizer split does not fragment or reorder that marker |
| 12 | OpenAI REST surface | `usage.prompt_tokens` changes for some models; that is the correct value, but it is a user-visible change and needs a `CHANGELOG.md` note |
| 13 | Native REST surface | same |
| 14 | CLI | no new flags expected; `./juno gguf-info` should report the file's declared pre-type |

## Implementation steps

1. Read Tier 01's recorded enumeration of `tokenizer.ggml.pre` values and what it implemented, then
   re-enumerate across `models/` via `./juno gguf-info` and record the diff in this file. The diff —
   not the whole enumeration — is this tier's implementation list.
2. Write the token-ID parity corpus and its expected output per model family **first** — this is
   the test that must fail for the right reason before anything changes. Note what "the right reason"
   now means: Tier 01 landed the dispatch, so a failure here is a wrong or missing *split*, not an
   unread key. If the corpus passes on arrival for a given family, that is a result worth recording
   rather than a reason to weaken the corpus.
3. Implement each uncovered split, one pre-type at a time, re-running the parity corpus after each.
4. Confirm the fail-closed path Tier 01 added still fires for a pre-type left unimplemented, and that
   the set it fires on is now smaller than it was.
5. Resolve the LoRA question in rows 9 and 10 explicitly before closing the tier. Tier 01 may already
   have changed TinyLlama's tokenization as a side effect of precondition 7; if so, its record says so
   and the adapter on disk (`tinyllama-1.1b-chat-v1.0.Q4_K_M.lora`) is already on the far side of that
   change. The decision — re-train, version the `.lora` format with the pre-type it was trained under,
   or accept and warn — is still owned here, and is now a decision about an adapter whose provenance
   is known rather than a hypothetical.

## Tests to write/upgrade before implementation

- **Plan check, first**: `scripts/performance-tests/check-plan-thresholds.sh` passes before any other
  test or code in this tier (README execution rule 7).
- **New `GgufTokenizerPreTokenizerTest`**: one case per implemented pre-type, asserting the split
  boundaries directly (not just the final IDs), including the cases that distinguish the families —
  digit runs, contractions, punctuation runs, leading/trailing whitespace, CJK, emoji, and mixed
  scripts.
- **New token-ID parity test**: a fixed corpus tokenized per sweep model, asserted against recorded
  reference IDs. Where a reference is unobtainable, assert against a committed golden file and note
  in this tier's file that it is a Juno-derived golden, not an independent reference.
- **Fail-closed regression test**: Tier 01 shipped this behaviour and its test; keep it, and extend it
  to assert the rejection set *shrank* — a pre-type this tier implemented no longer rejects, while a
  still-unimplemented one still does. A GGUF with no pre-type key still loads on the pre-existing path.
- **`ModelLiveRunnerIT`**: assert `usage.prompt_tokens` for a fixed prompt per model matches the
  parity corpus expectation, so a future regression in this area shows up against a real model.
- **New bash smoke script**: `scripts/performance-tests/smoke-tokenizer.sh` — runs
  `./juno gguf-info` across `models/` reporting each file's declared pre-type, and drives a fixed
  prompt through `/v1/chat/completions` per sweep model asserting the expected `prompt_tokens`.
- **Standing CPU and allocation gate** (README, "Test infrastructure"): run against the pre-tier jar
  and score it before closing this tier.
- **Perf gate (required)**: this is not a forward-pass change, but it changes token counts, which
  is the denominator of every pp and tg figure this plan publishes. Re-run `compare-llama-cpp.sh`
  on all four sweep models and publish under `docs/perf-compare/<timestamp>-tokenizer-parity/` (named for what it
  measures; tier numbers stay out of directory names, README execution rule 7 check 6).

  **Threshold.** Token-count mismatch between Juno and the reference tokenizer is **<= 0 tokens**
  per parity-corpus line on every sweep model — exact equality, not a tolerance, since that is the
  whole point of the tier. **Measure it with the reference tool's tokenizer (`llama-tokenize`) on the
  parity corpus, not with `llama-bench`**: `llama-bench -p N` prefills N synthetic token IDs and never
  tokenizes text, so it has no token count to compare against, and the benchmark prompt's count is
  calibrated to `n_prompt` on the Juno side whatever the split does. Tier 01's precondition 7 should already have achieved this for the sweep
  models; if it did, this threshold is a regression check that passes on arrival, and saying so is the
  correct outcome rather than a sign the threshold is too weak. What is *not* already covered is
  item 4's stronger claim: the token **IDs** must match the reference for the parity corpus, not only
  the count, on every sweep model — equal counts with a different split is the failure this item
  exists to catch and is invisible to precondition 7's check.

  Throughput must not regress: Juno tg and pp t/s **>= 0.95x** the pre-tier build on every sweep
  model, from a same-hour interleaved A/B with pinned clocks against the pre-tier build (README, "No-regression gates tighter than the floor are Juno-against-Juno"). If any sweep model's token count moves
  *here* — which would mean Tier 01's implementation was incomplete for that family — state which model
  moved and why, mark the run as the new reference for it, and record it as a late measurement boundary
  under this tier's execution record, since the whole reason precondition 7 was hoisted into Tier 01
  was to avoid creating one at this point in the running order.

## Models needed

The four sweep models plus `Phi-3.5-mini` and `moondream2-q5_k.llamafile` cover the pre-types
likely to be present. No downloads expected, but step 1's enumeration decides that — if a declared
pre-type on disk has no reference tokenization available to check against, flag it to the user at
that point rather than asserting against Juno's own output and calling it parity.

## Exit criteria

- [ ] Tier 01's enumeration of `tokenizer.ggml.pre` values is re-run against current `models/` and the
      diff recorded in this file, with the reading that the key is read and dispatched on verified
      against current code rather than taken from Tier 01's record.
- [ ] Each enumerated pre-type has an implemented split with direct boundary tests — including the
      types Tier 01 did not need, which is this tier's own implementation list.
- [ ] The fail-closed path still rejects a still-unimplemented pre-type with an error naming it, the
      rejection set is demonstrably smaller than before this tier, and a file with no pre-type key
      still loads bit-identically.
- [ ] Token-ID parity evidence published per sweep model, with any remaining divergence documented
      per model rather than unstated.
- [ ] Juno's token count equals the reference tokenizer's (`llama-tokenize`, not `llama-bench`,
      which never tokenizes text) on every parity-corpus line for every sweep model.
- [ ] The LoRA-adapter question (checklist rows 9 and 10) is resolved and documented — re-train,
      version, or accept-and-warn — not left implicit, and covering both Tier 01's precondition-7
      change and any further change this tier made, since the adapter on disk may already be on the far
      side of the first one.
- [ ] Standing CPU and allocation gate met (README, "Test infrastructure"): CPU tg and pp
      >= 0.95x the pre-tier build, allocation per token <= 1.10x, in-span GC pause total <= 1.25x.
- [ ] Cross-surface checklist fully resolved.
- [ ] Perf gate published; the run is marked as the new reference for any model whose token count
      changed, and the shift is explained.
- [ ] Docs (`docs/howto.md`, `docs/agent-arch.txt`) updated, Juno-native language only.
- [ ] `CHANGELOG.md` entry added, noting that `usage.prompt_tokens` changes for affected models.
