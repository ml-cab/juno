# LoRA train and playback gate after the prefill-throughput work (2026-10-03)

Purpose: the closing LoRA regression gate for the prefill-throughput work. Gate: train time (total and per
pass) <= 1.25x and playback wall-clock tps >= 0.80x the baseline, median of three. **Unpinned**, agent-run;
a gate this loose may be read across unpinned runs, and both builds ran
back to back in one session.

Command, each side: `compare-lora.sh --gpu --skip-build --no-publish --reps 3`, TinyLlama 1.1B Q4_K_M, the
train-qa name-recall scenario (fresh adapter, train to the target loss, then playback). HEAD ran
2026-10-02 20:47 -0500, the baseline right after, each from its own tree.

Builds: HEAD `1ac490a` (shaded jar sha256 `66f02ee7c2908f78`). Baseline `ffd0ca7`, the build the
2026-09-30 reference sweeps measured, rebuilt from an export of that commit (its JSON reads
`git_commit: unknown`; jar `bd19a9306ea7901f`). `head/` and `base/` hold the per-repetition JSON and each
side's INDEX.

| Rep | HEAD train ms | HEAD ms/pass | HEAD playback t/s | Baseline train ms | Baseline ms/pass | Baseline playback t/s |
|---|---|---|---|---|---|---|
| 1 | 41,000 | 2,733 | 12.82 | 41,000 | 2,733 | 13.62 |
| 2 | 41,000 | 2,733 | 13.01 | 41,000 | 2,733 | 13.31 |
| 3 | 41,000 | 2,733 | 13.36 | 43,000 | 2,867 | 12.94 |
| **Median** | **41,000** | **2,733** | **13.01** | **41,000** | **2,733** | **13.31** |

**Gate met:** train 1.00x (limit <= 1.25x), playback 0.978x (limit >= 0.80x). Every repetition on both
builds trained 15 passes to the same final loss (1.1797) and recalled the name.

Why nothing moved: the LoRA handlers are their own classes, which do not use the prefill-window device
region, and keep scalar attention (the startup notice says so). The train timings come from the REPL log
at one-second resolution, so 41,000 against 43,000 ms is one tick.
