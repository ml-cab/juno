# Tier 04B: Tokenizer fidelity

Status: not started
Gap analysis refs: none — the gap analysis has no tokenizer section, and no other tier in this plan
owns the `tokenizer` module. See "Why this tier, why now".

## Objective

Make Juno's tokenization match what a model's GGUF actually specifies, instead of applying one
BPE strategy to every file regardless of what the file says. Concretely: read
`tokenizer.ggml.pre`, implement the pre-tokenizer split each declared pre-type requires, fail
closed on a pre-type Juno does not implement, and prove per model family that Juno's token IDs for
a given string match the reference tokenization for that family.

## Why this tier, why now

Before this tier was written, the `tokenizer` module appeared in this plan exactly twice, both
times inside a `mvn test -pl ...` command. Nothing owned it. That is a gap in two directions at
once:

**Correctness.** `GgufTokenizer` pre-splits on special tokens and then runs BPE merges over the
whole remaining segment. It never reads `tokenizer.ggml.pre`, and a repo-wide grep finds no
reference to that key anywhere. Real GGUF files declare a pre-tokenizer type (`llama-bpe`,
`gpt-2`, `deepseek-llm`, `qwen2`, and others), and each one specifies a regex split applied
*before* merging — the split that decides whether `"1234"` is one token or four, whether `"don't"`
splits at the apostrophe, and how runs of punctuation and leading spaces group. Skipping it does
not throw; it silently produces a different token sequence from the one the model was trained
against. That is precisely the silent-degrade shape Tier 00 spent its effort removing from the
architecture path, still present one module over.

**Measurement.** Every llama.cpp-relative ratio this plan collects divides a Juno throughput by a
llama.cpp throughput for "the same prompt." If the two engines tokenize that prompt differently,
they are not processing the same number of tokens, and the ratio silently absorbs the difference.
The benchmark-parity preconditions in [`README.md`](README.md) require Juno's `prompt_tokens` to be
within 10% of `n_prompt`; that check is a guard against the symptom. This tier removes the cause.

Sequenced here because it needs nothing from Tiers 00-04 and nothing later depends on it, but it
must land before Tier 08 adds four new architecture families — each of which brings its own
pre-tokenizer type, and adding handlers for them while the pre-tokenizer is a single hardcoded
strategy would bake four more silent divergences in.

## Scope

### In scope

1. **Read `tokenizer.ggml.pre` and dispatch on it.** Add the metadata read and a pre-tokenizer
   abstraction with one implementation per supported pre-type. Start from the types the files on
   disk actually declare — run `./juno gguf-info` across `models/` and enumerate them before
   writing any code, rather than implementing a guessed list.
2. **Implement the regex split per supported pre-type**, applied between special-token splitting
   and BPE merging. The GPT-2/`llama-bpe` family split is the first target since it covers Qwen2,
   Phi-2 and the Llama-3 family.
3. **Fail closed on an unknown pre-type.** A file declaring a pre-type Juno does not implement must
   be rejected at load with an error naming the pre-type — the same treatment
   `ForwardPassHandlerLoader` now gives an unrecognized architecture, and for the same reason. A
   file with no `tokenizer.ggml.pre` key at all keeps today's behaviour, which is correct for the
   SentencePiece-style models that predate the key.
4. **Cross-engine token-ID parity evidence.** For each sweep model, record Juno's token IDs for a
   fixed corpus and compare against the reference tokenization for that model. Where a divergence
   remains after items 1-3, document it per model rather than leaving it unstated.

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
| 9 | LoRA training | **a real risk**: training tokenizes its corpus. A pre-tokenizer change alters the token sequence a trained adapter was fitted to, so an adapter trained before this tier may no longer compose correctly with a base model tokenized after it. Decide and document: re-train, version the `.lora` format with the pre-type it was trained under, or accept and warn. Do not leave this implicit |
| 10 | LoRA playback | same consideration; `tinyllama-...Q4_K_M.lora` on disk was trained under today's tokenizer |
| 11 | Vision | vision splices image tokens at `<image>` positions — confirm the pre-tokenizer split does not fragment or reorder that marker |
| 12 | OpenAI REST surface | `usage.prompt_tokens` changes for some models; that is the correct value, but it is a user-visible change and needs a `CHANGELOG.md` note |
| 13 | Native REST surface | same |
| 14 | CLI | no new flags expected; `./juno gguf-info` should report the file's declared pre-type |

## Implementation steps

1. Enumerate the `tokenizer.ggml.pre` values actually present across `models/` via
   `./juno gguf-info`, and record the table in this file. That table is the implementation list.
2. Write the token-ID parity corpus and its expected output per model family **first** — this is
   the test that must fail against today's code for the right reason before anything changes.
3. Add the metadata read and the pre-tokenizer dispatch, with today's behaviour as the no-key path.
4. Implement each split, one pre-type at a time, re-running the parity corpus after each.
5. Add the fail-closed path for unimplemented pre-types.
6. Resolve the LoRA question in row 9 explicitly before closing the tier.

## Tests to write/upgrade before implementation

- **New `GgufTokenizerPreTokenizerTest`**: one case per implemented pre-type, asserting the split
  boundaries directly (not just the final IDs), including the cases that distinguish the families —
  digit runs, contractions, punctuation runs, leading/trailing whitespace, CJK, emoji, and mixed
  scripts.
- **New token-ID parity test**: a fixed corpus tokenized per sweep model, asserted against recorded
  reference IDs. Where a reference is unobtainable, assert against a committed golden file and note
  in this tier's file that it is a Juno-derived golden, not an independent reference.
- **New fail-closed test**: a GGUF declaring an unimplemented pre-type is rejected with an error
  naming it, and a GGUF with no pre-type key still loads on today's path.
- **`ModelLiveRunnerIT`**: assert `usage.prompt_tokens` for a fixed prompt per model matches the
  parity corpus expectation, so a future regression in this area shows up against a real model.
- **New bash smoke script**: `scripts/performance-tests/smoke-tier04b-tokenizer.sh` — runs
  `./juno gguf-info` across `models/` reporting each file's declared pre-type, and drives a fixed
  prompt through `/v1/chat/completions` per sweep model asserting the expected `prompt_tokens`.
- **Perf gate (required)**: this is not a forward-pass change, but it changes token counts, which
  is the denominator of every pp and tg figure this plan publishes. Re-run `compare-llama-cpp.sh`
  on all four sweep models and publish under `docs/perf-compare/<timestamp>-tier04b-tokenizer/`.

  **Threshold.** Juno's `prompt_tokens` for the benchmark prompt must equal llama.cpp's token count
  for the same prompt on every sweep model — **exact equality, not a tolerance**, since that is the
  whole point of the tier. Throughput must not regress: tg ratio within 0.95x and pp ratio within
  0.95x of the pre-tier baseline, median of three per the README's noise-floor rule. Expect the
  published ratios to shift where token counts changed; state which models moved and why, and mark
  the run as the new reference for those models.

## Models needed

The four sweep models plus `Phi-3.5-mini` and `moondream2-q5_k.llamafile` cover the pre-types
likely to be present. No downloads expected, but step 1's enumeration decides that — if a declared
pre-type on disk has no reference tokenization available to check against, flag it to the user at
that point rather than asserting against Juno's own output and calling it parity.

## Exit criteria

- [ ] `tokenizer.ggml.pre` is read, and the enumeration of values present across `models/` is
      recorded in this file.
- [ ] Each enumerated pre-type has an implemented split with direct boundary tests.
- [ ] An unimplemented pre-type fails closed with an error naming it; a file with no pre-type key
      still loads on today's path, bit-identically.
- [ ] Token-ID parity evidence published per sweep model, with any remaining divergence documented
      per model rather than unstated.
- [ ] `prompt_tokens` equals llama.cpp's token count for the benchmark prompt on every sweep model.
- [ ] The LoRA-adapter question (checklist rows 9 and 10) is resolved and documented — re-train,
      version, or accept-and-warn — not left implicit.
- [ ] Cross-surface checklist fully resolved.
- [ ] Perf gate published; the run is marked as the new reference for any model whose token count
      changed, and the shift is explained.
- [ ] Docs (`docs/howto.md`, `docs/agent-arch.txt`) updated, Juno-native language only.
- [ ] `CHANGELOG.md` entry added, noting that `usage.prompt_tokens` changes for affected models.
