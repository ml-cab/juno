# GPU prefill ratio at 128 and 512 tokens after the tiled attention kernel: indicative

> **Indicative, not a gate reading.** **Clocks not pinned**, harness default repetitions, default lane only, one
> sweep per length taken back to back (`n128/` 04:57Z, `n512/` 05:04Z, 2026-10-06). Read to decide whether the
> 512-over-128 milestone row still measures something before the tier's closing sweeps score it.

**Build.** HEAD `ffb9583` plus the uncommitted decode-region change (`juno_tree_dirty: true`), jar
`ace9e5a64b1192eb`: the tiled attention kernel and attention inside the decode region (decode-only; prefill
does not run it).

**Commands** (from the repository root):

```
scripts/performance-tests/compare-llama-cpp.sh --gpu --n-prompt 128 --no-tuned-lane \
  --models tinyllama-1.1b-chat-v1.0.Q4_K_M,qwen2.5-3b-instruct-q4_k_m,Phi-3.5-mini-instruct-Q4_K_M,mistral-7b-instruct-v0.1-q4_k_m \
  --juno-jar dist/gpu-residency-attention-ab/candidate-shaded.jar
# the same with --n-prompt 512
```

Every row scorable (dispersion within 15%, prompt parity 128/128 and 512/512).

| Model | pp ratio at 128 | pp ratio at 512 | 512 over 128 | Reference before the kernel (Tier 01C close) | Juno pp t/s 128 / 512 |
|---|---|---|---|---|---|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 0.623x | 0.730x | **1.171** | 0.606 | 2273 / 3030 |
| qwen2.5-3b-instruct-q4_k_m | 0.734x | 0.842x | **1.147** | 0.697 | 1089 / 1262 |
| Phi-3.5-mini-instruct-Q4_K_M | 0.303x | 0.307x | **1.015** | 0.743 | 354 / 369 |
| mistral-7b-instruct-v0.1-q4_k_m | 0.775x | 0.826x | **1.067** | 0.642 | 512 / 547 |

**How to read it.** The milestone row asks for 512 over 128 >= 1.00 on every sweep model. Unpinned, every model
reads above it: Juno's prefill throughput now rises from 128 to 512 tokens, as the reference tool's does, where
before the kernel it fell. Phi-3.5-mini clears it by 1.5%, inside the 15% noise floor; its attention still runs
outside the prefill region with host copies. The reference column is from the pinned closing sweeps
`20261004T113210Z`/`20261004T114812Z`, so compare it for direction only. Recordings and logs were not copied;
local paths are scrubbed.
