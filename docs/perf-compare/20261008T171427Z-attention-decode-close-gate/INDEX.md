# Attention and long context: closing gate re-run after the decode-width attention fix (owner run)

**Purpose.** The first closing gate (`20261008T054413Z-attention-close-gate`) missed one row: Qwen2.5-3B GPU
generation at 0.948x the pre-tier build, traced to the attention kernel at decode width. After the fix
(`20261008T163349Z-decode-attention-kernel`: decode rows read their keys straight from global memory, bit-identical
output), this re-runs the GPU no-regression A/B. The CPU gate and the 2048-token attention speedup are not re-run:
the fix reaches neither the CPU path nor a prefill window wider than one row, and their readings in the first gate
stand.

**Pinned: yes** (`clock_pinned: true` on all six runs).

**Builds.**

| Side | Source | Shaded jar sha256 (first 16) |
|---|---|---|
| baseline (A) | HEAD `05a17de`, the last build before the attention and long-context work | `d1f50669dfc23d55` |
| candidate (B) | HEAD `51fa5d4` plus the decode-width attention path and test-only changes | `43b22345f1175c8f` |

**Command.** `dist/attention-decode-close/run-gate.sh` part A: each invocation `compare-llama-cpp.sh --gpu --pin-clocks
--n-prompt 512 --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish --juno-jar <jar>`, four sweep
models, A B A B A B, medians of three per side.

## Results (gate: tg and pp B/A >= 0.95; allocation per token <= 1.10x; GC pause in the token span <= 1.25x, or <= 5 ms under a 5 ms baseline)

| Model | pp A | pp B | pp B/A | tg A | tg B | tg B/A | alloc/token A | B | B/A | GC ms A | B | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tinyllama-1.1b | 1173.41 | 2901.81 | 2.473 | 61.33 | 142.00 | 2.315 | 48.0M | 39.0M | 0.812 | 7.4 | 0.0 | met |
| qwen2.5-3b | 605.97 | 1182.20 | 1.951 | 29.01 | 28.99 | **1.000** | 142.9M | 142.7M | 0.998 | 0.0 | 0.0 | met |
| Phi-3.5-mini | 238.40 | 646.37 | 2.711 | 28.59 | 50.03 | 1.750 | 210.4M | 178.6M | 0.849 | 11.7 | 12.2 | met |
| mistral-7b | 294.60 | 502.72 | 1.706 | 20.32 | 35.80 | 1.762 | 223.0M | 193.6M | 0.868 | 14.8 | 0.0 | met |

Qwen2.5-3B tg per run: A 29.06 / 29.01 / 27.46, B 28.81 / 29.08 / 28.99 (first gate: B lower in all three pairs).

## How to read it

- Every row is met. Qwen2.5-3B, which the decode residency region declines, now generates at the pre-tier build's
  rate; the generation lane decodes from a short context, where attention is a small part of a step, so the
  decode-width fix shows there as the regression removed rather than as a gain.
- The closing sweeps of the same run: `20261008T173153Z` (128), `20261008T174605Z` (512), `20261008T180014Z` (2048,
  three models), `20261008T181417Z` (2048, Phi-3.5-mini at 8 GiB).
- Local paths are replaced by `<repo>/`.
