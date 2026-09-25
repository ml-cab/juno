# Tier 05: Sampling & grammar completeness

Status: not started
Gap analysis refs: §1.4

## Objective

Add the missing sampler strategies (min-p, typical-p, tail-free, mirostat v1/v2, DRY, XTC), make
the sampler chain configurable (order and step selection, not a hardcoded seven-step sequence),
implement true per-occurrence-count OpenAI `frequency_penalty` instead of faking it via repetition
penalty, and close the two grammar gaps (byte-only char classes → full Unicode codepoint ranges;
the arbitrary `max ≤ min + 8` bounded-repetition cap) plus meaningfully widen the JSON-Schema-to-
grammar subset (at minimum `$ref` for shared definitions, since that's flagged as "extremely
common" in real schemas).

## Why this tier, why now

This tier is independent of the GPU/architecture work in Tiers 01-04 — it's pure sampler/grammar
logic, CPU-side, model-architecture-agnostic. It's sequenced here (rather than earlier) because it
benefits from Tier 00's sampler-pipeline-order documentation fix landing first (this tier will
restructure `Sampler.create()` into a configurable chain, so starting from a doc-and-code-agreed
baseline avoids compounding the confusion Tier 00 fixed).

## Scope

### In scope

1. Configurable sampler chain: replace `Sampler.create()`'s hardcoded seven-step sequence with an
   ordered, selectable list of steps, exposed via CLI (a `--samplers` -style flag, Juno-native
   naming) and the OpenAI/native REST APIs (a new `x_juno_samplers`-style field, following the
   project's existing `x_juno_*` extension convention).
2. New sampler steps: min-p, typical-p (locally typical sampling), tail-free sampling, mirostat v1
   and v2, DRY (repeat-suppression), XTC. Each implemented as its own `SamplingStep`-family class,
   consistent with the existing per-step-class pattern (`TopKStep`, `TopPStep`, etc.).
3. Real `frequency_penalty`: a genuine per-occurrence-count-scaled penalty step, replacing
   `OpenAiAdapter`'s current conversion into the binary repetition-penalty step. The existing
   binary repetition-penalty step stays as-is for Juno's native `repetitionPenalty` parameter —
   this adds a second, correctly-semantics step for the OpenAI field rather than changing the
   existing one's behavior (which would be a breaking change for existing native-API callers).
4. GBNF: full Unicode codepoint char classes (not byte-only), and a real justification-or-removal
   of the `max ≤ min + 8` bounded-repetition cap (this tier removes it unless investigation turns
   up a real reason to keep a bound, in which case document why and pick a defensible number).
5. JSON-Schema-to-grammar: add `$ref`/`$defs` support (schema-local definitions, the most common
   real-world need), and re-evaluate `minLength`/`maxLength`/`minimum`/`maximum` numeric/string
   bounds as a stretch goal within this tier if `$ref` lands with room to spare — otherwise file
   those explicitly as follow-up scope, not silently dropped.

### Out of scope

- `pattern`/`format` (regex-based) JSON Schema constraints — meaningfully harder (requires
  regex-to-grammar compilation) and not required to unblock the common `$ref` case; explicitly
  deferred, tracked as follow-up scope in this tier's exit notes rather than silently dropped.
- `oneOf`/`anyOf`/`allOf`/`not`/`if-then-else` — same treatment, deferred.
- Any change to speculative decoding's grammar interaction beyond what already exists (Tier 06's
  job if it comes up).

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | sampler/grammar logic is backend-agnostic; primary correctness surface |
| 2 | CUDA GPU inference | confirm new sampler steps don't assume CPU-resident logits in a way that breaks once Tier 01's residency work reaches the LM head/sampling boundary — check for a dependency conflict with Tier 01 at implementation time |
| 3 | ROCm GPU inference | same check, N/A for kernel-level work since sampling is host-side |
| 4 | Static schedule | per-request sampler-chain configuration must work when multiple different chains are requested within the same micro-batch |
| 5 | Continuous schedule | same, across `ContinuousBatchEngine`'s per-slot sampler state |
| 6 | Single-node local mode | primary dev/test surface |
| 7 | Pipeline-parallel cluster | sampler runs on the coordinator side after logits return from the last shard — confirm chain configuration correctly reaches the coordinator regardless of which node produced final logits |
| 8 | Tensor-parallel cluster | same, post-AllReduce |
| 9 | LoRA training | N/A — training doesn't sample |
| 10 | LoRA playback | new sampler steps must compose correctly with LoRA-modified logits (no special interaction expected, but must be tested, not assumed) |
| 11 | Vision | `/v1/vision/chat` must accept the same sampler-chain configuration as text chat |
| 12 | OpenAI REST surface | new `frequency_penalty` semantics, new `x_juno_samplers` field, widened `json_schema`/`$ref` support |
| 13 | Native REST surface | same, via native field names |
| 14 | CLI | new `--samplers`-style flag; existing individual flags (`--temperature`, `--top-k`, etc.) keep working and compose with the new chain-selection flag rather than being replaced |
| 15 | JVM embedding facade | `JunoPlayer`/`LoraTrainer` build `SamplingParams` directly, so a configurable chain that is only reachable through the CLI flag and the REST field leaves the facade on the hardcoded default; expose chain selection there or document the limitation |

## Implementation steps

1. Refactor `Sampler`/`SamplingStep` into a configurable chain first (mechanical change, no new
   sampling behavior yet) — this is the riskiest part to get wrong silently, so it gets its own
   regression pass (identical output for the existing default chain) before any new step is added.
2. Add each new sampler step behind its own unit tests (known-input/known-output cases from
   published reference implementations of min-p/typical-p/tail-free/mirostat/DRY/XTC).
3. Implement real `frequency_penalty`.
4. Widen GBNF char classes to full Unicode; remove or properly justify the repetition cap.
5. Add `$ref`/`$defs` to `JsonSchemaToGbnf`.
6. Run the full cross-surface smoke matrix, including a mixed-chain static-batch case (two
   concurrent requests with different sampler configurations in the same micro-batch).

## Tests to write/upgrade before implementation

- **`SamplerTest`**: baseline regression — default chain produces bit-identical output to
  pre-refactor, for a fixed seed, before any new step lands.
- **New per-step unit tests**: `MinPStepTest`, `TypicalPStepTest`, `TailFreeStepTest`,
  `MirostatStepTest` (v1 and v2), `DryStepTest`, `XtcStepTest` — each validated against published
  reference behavior on small synthetic logit distributions.
- **`OpenAiAdapterTest`**: new cases for real `frequency_penalty` semantics (escalating penalty
  with repeat count, distinct from the existing binary repetition-penalty behavior).
- **`GbnfGrammarTest`**: Unicode codepoint char-class cases; bounded-repetition cases beyond the
  old `min+8` ceiling.
- **`JsonSchemaToGbnfTest`**: `$ref`/`$defs` resolution cases, including a schema that would have
  been rejected before this tier.
- **`ModelLiveRunnerIT`**: add a mixed-sampler-chain static-batch check.
- **New bash smoke script**: `scripts/performance-tests/smoke-tier05-sampling-grammar.sh` —
  exercises each new sampler step and the widened grammar support end-to-end via
  `/v1/chat/completions`, plus the existing `smoke-grammar.sh`/`smoke-tools.sh` scripts must still
  pass unmodified (regression guard for existing grammar/tools behavior).
- No mandatory perf gate rerun (this tier is not a forward-pass/MatVec/GPU-residency/batching/KV
  change per the existing `CLAUDE.md` rule) — confirm that classification still holds at
  implementation time before skipping the gate.

## Models needed

Existing models (`tinyllama`, `qwen2.5-3b`) are sufficient — this tier is sampler/grammar logic,
not model-architecture-dependent.

## Exit criteria

- [ ] Sampler chain is configurable (order + selection) via CLI and both REST surfaces, with the
      old default chain preserved as the literal default when nothing is configured.
- [ ] min-p, typical-p, tail-free, mirostat v1/v2, DRY, XTC implemented and unit-tested against
      reference behavior.
- [ ] Real per-occurrence `frequency_penalty` implemented, distinct from the existing repetition
      penalty step.
- [ ] GBNF supports full Unicode char classes; repetition cap removed or justified.
- [ ] `$ref`/`$defs` supported in JSON-Schema-to-grammar; remaining unsupported keywords
      (`pattern`/`format`/`oneOf`/etc.) still fail closed with a clear error, not silently ignored.
- [ ] Cross-surface checklist fully resolved.
- [ ] Existing `smoke-grammar.sh`/`smoke-tools.sh` still pass unmodified.
- [ ] `x_juno_samplers`, the new `frequency_penalty` semantics and the widened `json_schema` support
      are declared in `api/src/main/resources/openapi.yaml` and `juno-api.yaml` (README
      feature-complete rule).
- [ ] Docs updated (`docs/howto.md` sampler/grammar sections), Juno-native language only.
- [ ] `CHANGELOG.md` entry added.
