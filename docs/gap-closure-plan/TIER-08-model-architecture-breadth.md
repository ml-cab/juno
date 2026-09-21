# Tier 08: Model architecture breadth

Status: not started
Gap analysis refs: §1.9, and the real-file findings in [`INVENTORY.md`](INVENTORY.md)

## Objective

Build real, tested handlers for the architecture families Juno already has files for but can't
correctly run: `gemma4` (Gemma), `mistral3` (the newer Mistral architecture family used by
`Devstral`), `qwen35` (Qwen3.5), and `minimax-m2` (a real MoE architecture with 24 experts). Also
add broader MoE family support (Mixtral-shaped models: architecture string `llama` but with expert
tensors) so the exact failure mode described in gap analysis §1.9 — an MoE model silently
misrouted to the dense handler — cannot happen for any model Juno claims to support.

## Why this tier, why now

This is the tier where the four "falls through to the generic handler" files from
[`INVENTORY.md`](INVENTORY.md) get real support, not just a safe rejection (Tier 00 already made
sure the current behavior is at least safe). It's sequenced after quantization coverage (Tier 04)
because two of these files (`Devstral`, `minimax-m2.5-tiny`) need their quant formats implemented
before there's anything to test the architecture handler against. It's after speculative decoding
(Tier 06) because Tier 06 explicitly adds `forwardVerify` overrides per architecture, and doing that
work against a stable, final set of architecture handlers is less wasteful than adding verify
support to a handler that's about to be substantially rewritten here.

## Scope

### In scope

1. **Gemma handler** (`gemma4` architecture) — verify actual tensor-layout differences from Llama
   (Gemma has some real structural differences: different normalization placement, embedding
   scaling) and build a dedicated handler if the current silent fallback isn't actually correct
   (Tier 00's audit will have already determined whether it's silently wrong or coincidentally
   fine — this tier acts on that finding).
2. **Mistral3 handler** (or confirmed-safe extension of the existing Llama-family path) for
   `Devstral-Small` and any other `mistral3`-architecture file.
3. **Qwen3.5 handler** for `qwen35`, using the existing `Qwen3TransformerHandler` as a reference
   point — determine what's actually different between `qwen3` and `qwen35` metadata/tensors and
   either extend the existing handler or add a sibling.
4. **Minimax-M2 MoE handler**, modeled on `Qwen3MoeTransformerHandler`'s existing routed-expert
   design but adapted to `minimax-m2`'s actual tensor names/gating function
   (`expert_gating_func=2` per its GGUF metadata — confirm what that value means before assuming
   it matches Qwen3-MoE's gating).
5. **Generic Mixtral-shaped MoE detection**: independent of any specific named architecture, add a
   structural guard in `ForwardPassHandlerLoader` that inspects for `ffn_gate_exps`/`ffn_up_exps`/
   `ffn_down_exps` tensors regardless of the declared `general.architecture` string, and routes to
   an MoE-capable handler (or fails closed with a clear "MoE tensors present, architecture not
   recognized" error) rather than ever letting expert tensors silently pass through the dense
   handler unused.

### Out of scope

- DeepSeek-MoE-specific optimizations (e.g. its particular shared-expert design) beyond what the
  generic Mixtral-shaped detection in item 5 catches — a dedicated DeepSeek handler is additional
  scope only if a real DeepSeek file becomes available to test against.
- Any change to quantization support itself (that's Tier 04, a prerequisite here).

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | every new handler must work correctly on CPU first (correctness oracle) |
| 2 | CUDA GPU inference | new handlers should get GPU residency parity with existing dense handlers where feasible; MoE routing on GPU needs its own `matVecExpert`-equivalent kernel path, mirroring `Qwen3MoeTransformerHandler`'s existing (CPU-only per the gap analysis) expert slicing — extending it to GPU is in scope if time allows, otherwise explicitly deferred and documented |
| 3 | ROCm GPU inference | NEEDS-AMD-HARDWARE for any new GPU kernel work; CPU path must be correct regardless |
| 4 | Static schedule | new handlers must support `generateBatch()` correctly |
| 5 | Continuous schedule | new handlers must support `ContinuousBatchEngine` correctly |
| 6 | Single-node local mode | primary dev/test surface for every new handler |
| 7 | Pipeline-parallel cluster | new architectures must be shardable by layer range like existing ones |
| 8 | Tensor-parallel cluster | new architectures must work with the existing (broadcast+AllReduce) tensor-parallel path at minimum; real per-layer slicing for them is Tier 09's job |
| 9 | LoRA training | confirm new architectures can be LoRA-trained (`LoraTrainableHandler`-equivalent wiring) — if not feasible this tier, document as an explicit, fail-closed gap, not silent |
| 10 | LoRA playback | same |
| 11 | Vision | N/A unless any of these four turns out to be a vision-capable architecture (none currently known to be) |
| 12 | OpenAI REST surface | new architectures must work end-to-end via `/v1/chat/completions`, including correct chat-template detection (`ChatTemplate.java`'s existing mapping needs entries for any of these four that don't already have one) |
| 13 | Native REST surface | same |
| 14 | CLI | `./juno gguf-info` output should clearly identify each new architecture; `./juno local`/`cluster` need no new flags, just correct auto-dispatch |

## Implementation steps

1. For each of the four architectures, deep-dive the actual GGUF tensor names/shapes/metadata
   (via `./juno gguf-info` and direct hex/structure inspection if needed) and diff against the
   nearest already-supported family, before writing any handler code.
2. Build/extend handlers one architecture at a time (own sub-milestone within the tier, but the
   tier as a whole isn't complete until all four plus the generic MoE guard are done, per rule 1 —
   don't split this into a separate tier per architecture).
3. Add the generic Mixtral-shaped structural MoE detection last, once real MoE handler patterns
   (Qwen3-MoE, Minimax-M2) exist to route into.
4. Add chat-template entries for any of the four missing from `ChatTemplate.java`.
5. Run the full cross-surface smoke matrix per architecture.

## Tests to write/upgrade before implementation

- **New handler unit tests** per architecture (`Gemma4TransformerHandlerTest`,
  `Mistral3TransformerHandlerTest` or equivalent, `Qwen35TransformerHandlerTest`,
  `MinimaxM2TransformerHandlerTest`), each validating forward-pass output against known-correct
  reference values if obtainable, or at minimum internal consistency (e.g. logit distribution
  sanity, no NaNs, matches greedy-decode expectations for a simple prompt).
- **`ForwardPassHandlerLoaderTest`**: extend with the new architecture-string routing cases, and the
  new structural (tensor-name-based) MoE detection cases, including a synthetic "architecture says
  llama, tensors say MoE" fixture to directly test the Mixtral-shaped scenario without needing an
  actual Mixtral file.
- **`ChatTemplateTest`**: new template-detection cases for the four architectures.
- **`ModelLiveRunnerIT`**: add a load-and-generate check for each of the four real files
  (`gemma-4-E4B`, `Devstral-Small`, `Qwen3.5-0.8B`, `minimax-m2.5-tiny`), replacing whatever
  placeholder/rejection check Tier 00 added for them.
- **New bash smoke script**: `scripts/performance-tests/smoke-tier08-architecture-breadth.sh` —
  end-to-end chat completion against all four real files, asserting coherent (not just
  non-crashing) output.
- **Perf gate**: new forward-pass code is a hot-path change by definition — `compare-lora.sh` at
  minimum; a dedicated per-architecture microbenchmark if these models are large enough to matter
  (`Devstral` at 24B is the one most worth measuring for memory/GPU-layer-offload behavior).

## Models needed

All four target files are already present. If a true Mixtral-8x7B-shaped file or a working
`qwen3moe` file remains unavailable when this tier starts (see [`INVENTORY.md`](INVENTORY.md)'s
open gaps), ask the user then — the generic structural MoE guard can be tested with a synthetic
fixture in the meantime, but end-to-end `ModelLiveRunnerIT` coverage for genuine Mixtral and for the
*existing* `qwen3moe` handler needs real files.

## Exit criteria

- [ ] `gemma4`, `mistral3`, `qwen35`, `minimax-m2` each load and generate correct output via a
      real, tested handler (not a coincidental fallback).
- [ ] Generic structural MoE detection catches an architecture-string-lies-about-MoE case
      (synthetic fixture at minimum, real Mixtral file if available) and either routes correctly or
      fails closed with a clear error — never silently drops expert tensors.
- [ ] Chat templates correctly detected for all four architectures.
- [ ] Cross-surface checklist fully resolved, LoRA-trainability gap (if any) explicitly documented
      per architecture rather than silently absent.
- [ ] Perf gate published for at least the largest new model (`Devstral`, 24B).
- [ ] Docs (`docs/agent-arch.txt`'s architecture table, `docs/howto.md`) updated with the newly
      supported families, Juno-native language only.
- [ ] `CHANGELOG.md` entry added.
