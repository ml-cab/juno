# Tiled attention kernel at `n_prompt=2048`: indicative readings

> **Indicative, not a gate reading.** One repetition per model (`--juno-reps 1 --juno-warmup 2 --reps 1`),
> **clocks not pinned**, `--device-spans`, default lane only, the candidate build only. No threshold is scored
> from this directory: the attention-speedup milestone is scored from a same-session A/B alternating the
> pre-change jar and the candidate, median of three per side, at the tier's close.

**Purpose.** First reading of the tiled, online-softmax GPU attention kernel (it streams K/V through shared
memory and keeps no score row) on a 2048-token prefill window, against the pre-change decomposition of the
same window (`20261005T003146Z/milestone-decomposition.md`; 2048 readings `20261005T004009Z`,
`20261005T005334Z`, jar `3306ea4261f84a49`).

**Builds.** HEAD `3153e44` plus the uncommitted kernel change (`juno_tree_dirty: true`).
- `20261005T043214Z/` (Mistral 7B): jar `ced50575c7c29675`, the kernel with the 128- and 256-wide head
  variants only.
- `20261005T043507Z/` (TinyLlama) and `20261005T043552Z/` (Phi-3.5-mini, `COMPARE_HEAP=8g`): jar
  `b12152497e2341c9`, which adds the 64-wide head variant. Mistral 7B's 128-wide heads run the same code in
  both jars.

**Commands** (one per model, from the repository root):

```
scripts/performance-tests/compare-llama-cpp.sh --gpu --device-spans --n-prompt 2048 --no-tuned-lane \
  --models <model> --juno-jar juno-player/target/juno-player-0.1.2-shaded.jar \
  --juno-reps 1 --juno-warmup 2 --reps 1 --no-publish
scripts/performance-tests/prefill-breakdown.sh <run dir>   # -> <run dir>/prefill-breakdown.md
```

Phi-3.5-mini with `COMPARE_HEAP=8g`, as in the reference sweep.

| Model | Prefill ms, before / now | Attention ms, before / now | Attention speedup (raw) | SM clock in window, before / now | Attention speedup, clock-normalised | pp ratio at 2048, before / now | tg ratio now |
|---|---|---|---|---|---|---|---|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 5060 / 855 | 4459 / 291 | 15.3x | not recorded / 1829 MHz | not computable (no pre-change clock) | 0.125x / 0.672x | 0.331x |
| mistral-7b-instruct-v0.1-q4_k_m | 23226 / 4375 | 19253 / 1141 | 16.9x | 1607 MHz / 1746 MHz | 15.5x | 0.144x / 0.729x | 0.611x |
| Phi-3.5-mini-instruct-Q4_K_M | 24334 / 6325 | 19900 / 1973 | 10.1x | not recorded / 1809 MHz | not computable (no pre-change clock) | 0.091x / 0.304x | 0.523x |

**How to read it.** Attention is kernel plus copies plus host part, as the decomposition read it:
`region_attention_compute + attention_compute + attention_copies + attention_host` from `prefill-breakdown.md`.
On TinyLlama and Mistral 7B attention runs inside the prefill region, so it is kernel time only. On
Phi-3.5-mini it runs outside the region: 1036 ms of kernel, 250 ms of copies and 687 ms of host work; moving it
into the region is a later item of the same tier. Clock-normalised speedup is (before ms x before MHz) / (now ms x now MHz).
Only Mistral 7B's pre-change window had its clock sampled (`20261005T012348Z/gpu-clocks.md`). The harness has
recorded the in-window clock since 2026-10-05, so the milestone A/B normalises every model. TinyLlama's
breakdown residue is 5.8% of its (now short) window, above `prefill-breakdown.sh`'s 5% bound, so that file
reports the flag; the attention term is a direct span, not the residue.

Qwen2.5-3B was not read here.
