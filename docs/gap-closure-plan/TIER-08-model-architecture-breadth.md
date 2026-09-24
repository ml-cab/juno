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
before there's anything to test the architecture handler against.

**This tier runs before Tier 06, not after it** — see the tier index in [`README.md`](README.md),
which is the running order. An earlier draft placed it after Tier 06 while giving a rationale that
argues for exactly the opposite: Tier 06 adds a `forwardVerify` override per architecture, so
running it first means adding verify support to four handlers and then having this tier introduce
four more that either need the same work again or silently lack it. Running this tier first means
Tier 06 covers every handler in one pass, against a stable and final set. The dependency direction
was right in the prose and wrong in the ordering; the ordering is now the one the prose implies.

This tier also carries an obligation handed over from Tier 02. Tier 02 ships sliding-window
attention validated against a synthetic windowed-metadata fixture, because the only real windowed
file on disk (`gemma-4-E4B`, patterned 512-token window per Tier 00's audit) is not loadable until
this tier's Gemma handler exists. Real-model validation of the windowed path is therefore one of
this tier's exit criteria, not Tier 02's.

## Scope

### In scope

1. **Gemma handler** (`gemma4` architecture). Tier 00's audit already established what this file
   needs, so build to that finding rather than re-deriving it: patterned sliding-window attention
   with a 512 window, final logit softcapping (30), per-layer input embeddings, KV-shared layers,
   and all-supported tensor types (Q4_0/F32). Tier 00 also recorded that a Gemma variant *without*
   KV sharing would have loaded and run silently wrong under the old fallback, which is why the
   architecture guard rejects it by name today.

   **This handler is where Tier 02's sliding-window mechanism gets its real-model validation.** Use
   the window metadata read Tier 02 added; do not implement a second windowing path here. If the
   window key Tier 02 chose is not the one this file declares, that is a Tier 02 defect surfacing
   late — fix it in the shared mechanism, not with a Gemma-local special case.

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
| 2 | CUDA GPU inference | new handlers should get GPU residency parity with existing dense handlers where feasible, **including GPU-resident attention** — Tier 01B made that the default for every architecture it was able to measure on a real model and built the capability-reporting mechanism for the rest, so a new handler that quietly resolves to the scalar path reopens exactly the silent-degrade gap 01B closed; each new architecture either supports it and says so, or fails to the documented fallback with an explicit notice. MoE routing on GPU needs its own `matVecExpert`-equivalent kernel path, mirroring `Qwen3MoeTransformerHandler`'s existing (CPU-only per the gap analysis) expert slicing — extending it to GPU is in scope if time allows, otherwise explicitly deferred and documented |
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
  (`Devstral` at 24B is the one most worth measuring for memory/GPU-layer-offload behavior), plus
  `compare-llama-cpp.sh` for a llama.cpp-relative reading on each newly-supported architecture (per
  README's llama.cpp-relative gate) — these are exactly the models where Juno previously couldn't
  even load, so this is the first llama.cpp-relative data point for each one.

  **Threshold.** These architectures have no prior Juno baseline, so the gate is relative to the
  nearest already-supported family rather than to their own history:
  - each new handler's tg must reach **>= 0.70x** the tg of the closest supported dense model of
    comparable parameter count and quant on the same host — a new handler an order slower than its
    nearest neighbour indicates a layout or dispatch mistake, not merely an unoptimized path;
  - no already-supported architecture regresses: tg and pp both within **0.95x** of the pre-tier
    baseline, median of three per the README's noise-floor rule, since the generic structural MoE
    detection in item 5 runs on every load;
  - `Devstral` (24B, IQ1_S) additionally reports peak RSS and GPU-layer-offload behaviour with no
    threshold attached — it is the memory-pressure data point, and this is its first measurement.

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
- [ ] **Sliding-window attention validated end to end on `gemma-4-E4B`**, using Tier 02's mechanism
      and window-metadata read — the real-model half of Tier 02's sliding-window work, handed over
      because this tier is what makes that file loadable. A non-windowed model's output stays
      bit-identical.
- [ ] Each new handler either supports GPU-resident attention and reports that capability through
      the mechanism Tier 01B built, or fails to the documented fallback with an explicit notice —
      never a silent resolution to the scalar path.
- [ ] Cross-surface checklist fully resolved, LoRA-trainability gap (if any) explicitly documented
      per architecture rather than silently absent.
- [ ] Perf gate published for at least the largest new model (`Devstral`, 24B).
- [ ] Docs (`docs/agent-arch.txt`'s architecture table, `docs/howto.md`) updated with the newly
      supported families, Juno-native language only.
- [ ] `CHANGELOG.md` entry added.
