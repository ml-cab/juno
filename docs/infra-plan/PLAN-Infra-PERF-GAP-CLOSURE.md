# Performance Gap Closure — 5-Item Plan (2026-09-17)

## Purpose

This doc records the five highest-leverage items identified from a fresh read of
[`PLAN-Infra-PERF-ANALYSIS.md`](PLAN-Infra-PERF-ANALYSIS.md), the latest bake-off
([`docs/perf-compare/20260916T043335Z/`](../perf-compare/20260916T043335Z/)), and the
Tier 13/17 JFR findings, and routes each one to the plan or prompt doc that owns it. It is a
**routing document, not a new Infra tier** — where an existing tier/prompt already owns an item,
this doc points there instead of duplicating it (ROADMAP §1: tier numbers identify features, do not
fork planning for work another doc already scopes).

Full context/root-cause writeup for why these five were chosen: see the chat analysis this session
started from (llama.cpp-vs-Juno gap: MatVec launch-overhead dominance in decode, unbatched-by-default
prefill, VRAM fit, unpublished CPU SIMD parity, no speculative decoding yet).

## The five items

| # | Item | Owning doc | New or existing | Status at time of writing |
|---|------|-----------|------------------|---------------------------|
| 1 | Reduce MatVec/decode per-call launch + FFI overhead (highest-leverage single lever; `juno.MatVec` is 71-93% of decode wall time across the current model set) | [`PROMPT-P0-Gate.md`](PROMPT-P0-Gate.md) | Existing — resumed, not forked | P0 gate unmet (Phi-3.5 0.33× vs 0.5× target); explicitly deferred 2026-09-15, not resumed since |
| 2 | Default-flag / benchmark-methodology gap — the published *default* bake-off understates what Juno already does with shipped `auto` flags (0.13-0.20× tg / 0.002-0.03× pp default vs up to 0.41× tg / 3.85× pp tuned, same hardware) | [`PLAN-Infra-Tier18.md`](PLAN-Infra-Tier18.md) | **New** (Tier 18) | **Feature complete** (2026-09-18) — bake-off [`20260918T024641Z`](../perf-compare/20260918T024641Z/), tuned/default tg 1.48-1.63× (TinyLlama/Qwen2.5-3B/Phi-3.5-mini), 35.6× (Mistral-7B) |
| 3 | CPU SIMD (`--vector 1`) path unverified against several sessions of GPU-side kernel work | [`PROMPT-Vector-SIMD-Refresh.md`](PROMPT-Vector-SIMD-Refresh.md) | **New** (prompt; parallel track, not a tier) | Not started — track itself is feature complete, this is a re-verification |
| 4 | Speculative decoding (1.2-2× effective tg at high acceptance, orthogonal to kernel work) | [`PLAN-Infra-Tier9.md`](PLAN-Infra-Tier9.md) → [`PLAN-Infra-Tier12.md`](PLAN-Infra-Tier12.md) | Existing — resumed, not forked | Tier 9 **feature complete** (2026-09-18) — a live smoke test measured 94.9% draft acceptance but only ~7% wall-clock tg gain on a maximally-repetitive workload (attention-dispatch batching is the saving; per-row verify GEMM cost offsets the launch-count reduction, see `docs/perf-compare/README.md`), below this row's 1.2-2× expectation; Tier 12 (draft-model speculation, Tier 9's hard prerequisite) still Pending |
| 5 | Re-verify the last published Phi-3.5 ratio (0.33×, 2026-09-11) still holds post-Tier-17 before spending further kernel effort | Folded into [`PROMPT-P0-Gate.md`](PROMPT-P0-Gate.md) step 1 | Existing — not a separate item | Addressed by this session's addendum to that prompt |

Item 5 is not a standalone doc: it was already the first step of `PROMPT-P0-Gate.md`'s suggested
approach ("Baseline: `compare-llama-cpp.sh --gpu --vector 0` — record Phi-3.5 and Mistral ratios"),
just never explicitly framed as "the old number may be stale." This session added that framing plus
the new Attention-JFR evidence directly to `PROMPT-P0-Gate.md` (see its "Why this exists" addendum)
rather than creating a sixth doc for a one-step re-verification.

## Why items 1 and 5 were not forked into a new tier

`PROMPT-P0-Gate.md` already is the decode-kernel-overhead gate: it names the same root cause
(`juno.MatVec` ≈ 93-96% of GPU decode), the same target metric (Phi-3.5 tg ≥ 0.5×), and carries an
explicit, dated deferral note in `PLAN-Infra-ROADMAP.md` ("P0 decode-kernel gap — explicit deferral,
2026-09-15") stating a future session should resume from that exact prompt plus Nsight Compute
profiling. Creating a parallel tier for the same gap would violate ROADMAP §1 (one tier at a time)
and fragment the JFR evidence trail. This session's contribution is the addendum recording new
evidence (the `juno.Attention` JFR span, landed 2026-09-16, showing attention is 64.2% of *decode*
wall time at ctx≈512 — a long-context finding the short-context Tier 13 Phase A analysis did not
have) and flagging `--gpu-attention` as an untested decode-side lever on top of the tile-kernel work
already scoped.

## Why item 4 was not forked

Tier 9 (ngram speculative decoding) and Tier 12 (draft-model speculative decoding, hard-depends on
Tier 9) already fully scope this lever, correctly phased (P4, after Tier 8, which is feature
complete). Nothing about this session's analysis changes their design — it confirms they are the
right next P4 step once P0 kernel work is paused or a second engineer is available (ROADMAP
"Parallel tracks" only exempts LoRA/model-E2E/Vision/Vector-SIMD from the one-tier-at-a-time rule;
P4 speculative decoding is a full Infra tier and must wait its turn under §1 unless explicitly run
as the next tier after the current P0 work reaches feature complete or is intentionally parked).

## Recommended sequencing

Per ROADMAP §1 (one Infra tier in flight at a time), only one of {item 1 (`PROMPT-P0-Gate.md`), item
2 (Tier 18), item 4 (Tier 9 -> 12)} can be actively implemented at once. Item 3
(`PROMPT-Vector-SIMD-Refresh.md`) is a parallel track and does not block or get blocked by the
others.

1. **Item 5** (re-baseline) — a few minutes, gates whether item 1's kernel work target has moved.
   Do this first regardless of which tier is picked up next; it is cheap and de-risks everything
   downstream of "is 0.33× still accurate."
2. **Item 3** (Vector SIMD refresh) — can run in parallel with anything else; cheap, closes an
   honesty gap in every GPU bake-off published since (all say `juno_use_vector: 0`).
3. **Item 2** (Tier 18) — cheapest of the remaining Infra-tier-gated items (script + docs, no kernel
   risk); recommended next if the team wants a quick, low-risk win that makes today's real
   capability visible before committing further engineering to item 1.
4. **Item 1** (resume `PROMPT-P0-Gate.md`) — highest ceiling, highest effort/risk (CUDA kernel
   tuning, Nsight Compute profiling); the program's stated primary lever. Pick this up when
   dedicated kernel-authorship time is available.
5. **Item 4** (Tier 9 -> 12) — next after item 1 reaches feature complete or is intentionally
   parked, per P4's phase ordering; orthogonal effective-tg win once the base decode kernel is as
   fast as it will get in the near term.

This ordering is a recommendation, not a gate — whoever picks up the next Infra tier should still
follow ROADMAP §1's single-tier rule and the phase table, not this doc's numbering, if they conflict.

## References

- [`PLAN-Infra-PERF-ANALYSIS.md`](PLAN-Infra-PERF-ANALYSIS.md) — root-cause JFR breakdown this plan
  is built on
- [`docs/perf-compare/20260916T043335Z/`](../perf-compare/20260916T043335Z/) — latest default-flags
  bake-off (the numbers cited in the table above)
- [`docs/performance.md:424-440`](../performance.md) — `--gpu-attention` prefill bake-off (3.85× pp)
- [`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) — "P0 decode-kernel gap — explicit deferral" and
  "Attention JFR span landed" notes (2026-09-15 / 2026-09-16)
