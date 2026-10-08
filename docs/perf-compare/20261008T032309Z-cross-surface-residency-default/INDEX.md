# Cross-surface readings on the decode-region default

**Purpose.** The vision and LoRA no-regression readings, and the residency and long-prompt smoke summaries,
taken for the closing cross-surface matrix after `--gpu-residency` became `auto` by default. The vision gate
(latency `<= 1.25x`, decode tps `>= 0.80x`) and the LoRA gates (train `<= 1.25x` time per pass, playback
`>= 0.80x`) are looser than 0.90x, so they are read unpinned, median of three, both builds in one session.

**Pinned: no.** Governor `schedutil`, turbo on; desktop session sharing the GTX 1080.

**Builds.**

| Side | Source | Shaded jar sha256 (first 16) |
|---|---|---|
| baseline | HEAD `05a17de`, the last build before the attention and long-context work (full-materialization GPU attention; decode region through RoPE only; `--gpu-residency` default `off`), from `git archive`, built with package only | `d1f50669dfc23d55` |
| candidate | HEAD `51fa5d4` plus test-only changes (working tree): `--gpu-residency` default `auto` | `fefd30371caf535a` |

2026-10-08 03:23Z to 05:41Z, one engine at a time.

**Commands.** Each build's own script from its own tree:
- `compare-vision.sh --gpu --skip-build --no-publish`, three runs per build, alternated base, candidate,
  base, candidate, base, candidate;
- `compare-lora.sh --gpu --reps 3 --skip-build --no-publish`, base then candidate;
- `smoke-long-prompt-prefill.sh` (defaults: TinyLlama and Mistral 7B, 128, 512 and 2048 prompt tokens, static
  and continuous) and `smoke-gpu-residency.sh --models tinyllama,mistral-7b,Phi-3.5-mini,Qwen3-1.7B,qwen2.5-3b`,
  then `--models tinyllama-1.1b-chat-v1.0.Q4_K_M` (see below), candidate only, unmodified scripts.

## Results

| Gate | Baseline (median of 3) | Candidate (median of 3) | Ratio | Threshold | Result |
|---|---|---|---|---|---|
| vision `latency_ms` | 493,896 | 485,444 | 0.983x | `<= 1.25x` | met |
| vision decode tps | 1.4370 | 1.4413 | 1.003x | `>= 0.80x` | met |
| LoRA playback, wall-clock tps | 13.645 | 12.797 | 0.938x | `>= 0.80x` | met |
| LoRA train ms per pass | 2,733 | 2,733 | 1.000x | `<= 1.25x` | met |

Per repetition: vision latency base 481,007 / 495,018 / 493,896, candidate 485,444 / 488,007 / 482,806 ms;
decode tps base 1.478 / 1.437 / 1.427, candidate 1.362 / 1.441 / 1.531. LoRA playback base 13.67 / 13.08 / 13.65,
candidate 13.33 / 12.80 / 12.77 t/s. Six vision captions are byte-identical across both builds. Every LoRA
repetition trained 15 passes to loss 1.1797 on both builds.

| Smoke (candidate) | Result |
|---|---|
| `smoke-long-prompt-prefill.sh` | 36 PASS, 0 FAIL: 12 cells (two models, two schedules, three lengths); TinyLlama's 2048 capped at 2024 by its 2048-token context, Mistral 7B at 2048 prompts 2044 tokens; streamed equals unstreamed on every cell (`smoke-long-prompt-results.json`) |
| `smoke-gpu-residency.sh`, five models | 17 PASS, 0 FAIL; region active and greedy identical on against off on Mistral 7B, Phi-3.5-mini and Qwen3-1.7B, memory flat in both modes, Qwen2.5-3B declined with the logged reason, both cluster modes answer with local mode's output and leave no node JVM. The key `tinyllama` matched `tinyllama-1.1b-chat-v1.0.Q2_K.gguf` first, whose Q/K/V are not K-quant, so the region was declined there (`smoke-gpu-residency-summary.md`) |
| `smoke-gpu-residency.sh`, TinyLlama Q4_K_M by its full name | 7 PASS, 0 FAIL: region active on 8 of 8 layers of the local node, memory flat, greedy identical on against off, both cluster modes (`smoke-gpu-residency-tinyllama-q4km-summary.md`) |

## How to read it

- **Vision does not reach the changed code.** moondream2's text half is Phi-2, which has no decode region and
  no device weight path, and the CLIP encoder runs on the CPU; the default change and the tiled attention kernel
  cannot touch it. Equal times and identical captions confirm it.
- **LoRA declines the region** under `auto` (training keeps FP32-resident frozen weights; playback multiplies row by
  row), so neither build runs it there. Playback reads 6% lower on the candidate with overlapping repetitions
  (12.77 to 13.33 against 13.08 to 13.67); this is within the 15% noise floor and well inside the gate.
- Files: `vision-<base|cand>-<n>.json`, `lora-<base|cand>.json` (median of three) and `lora-<base|cand>-rep<n>.json`.
  Local paths are replaced by `<repo>/` and `<baseline-tree>/`.
