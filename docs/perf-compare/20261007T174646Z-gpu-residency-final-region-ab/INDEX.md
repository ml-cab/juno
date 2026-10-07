# Final decode residency region, `--gpu-residency` off against on (pinned) - 20261007T174646Z (gpu)

**Purpose.** The same-hour pinned A/B of the decode residency region on the final tree: the whole decode layer on
the device on the LLaMA-family, Phi-3 and Qwen3 handlers, the residual row kept on the device between layers, the
512-token KV mirror reserve under `--gpu-layers auto`, the scratch-growth fixes, and Qwen3's prefill norm, RoPE and
attention in the prefill region. It scores the region's throughput threshold: tg region on `>= 1.00x` region off on
every model where the region runs, `>= 0.95x` where it is declined.

**Pinned: yes**, on all six runs (`clock_pinned: true`); runs started 17:46Z to 18:03Z; owner run.

**Command.** `bash dist/gpu-residency-final-region-ab/run-gate.sh` (copied here as `run-gate.sh`): one jar, sha256
`533b1864d682bea5` (HEAD `d05d1fe` plus the uncommitted Qwen3 prefill change; decode code as at `d05d1fe`), the flag
alternated off, on, off, on, off, on; each run `compare-llama-cpp.sh --gpu --pin-clocks --models <five> --n-prompt 128
--juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish --gpu-residency off|on`. Re-scored from the run
files with the script's method (medians of three), since its console output was not kept. Each `off-N`/`on-N`
directory holds one run's JSON; `RUN-INDEX.md` is the harness's index. Local paths made repository-relative.

## Reading (Juno t/s, medians of three; every row scorable on all six runs)

| Model | Region | tg off (3 runs) | tg on (3 runs) | tg on/off | Threshold | pp on/off (not gated) | alloc/token off / on | GC max pause ms off / on | GPU tg ratio off / on |
|---|---|---|---|---|---|---|---|---|---|
| tinyllama-1.1b | whole layer | 61.05 (64.29, 61.05, 60.99) | 124.89 (124.89, 123.16, 126.36) | **2.046** | >= 1.00, met | 1.020 | 47.6M / 38.7M | 6 / 0 | 0.331x / 0.679x |
| mistral-7b | whole layer | 21.03 (22.14, 20.96, 21.03) | 35.09 (34.00, 36.40, 35.09) | **1.669** | >= 1.00, met | 0.968 | 223.9M / 193.6M | 13 / 0 | 0.642x / 1.002x |
| Phi-3.5-mini | whole layer | 28.84 (28.84, 28.57, 29.33) | 47.31 (47.31, 47.82, 47.13) | **1.641** | >= 1.00, met | 0.974 | 209.1M / 180.7M | 9 / 10 | 0.505x / 0.829x |
| Qwen3-1.7B | whole layer | 31.49 (31.45, 35.00, 31.49) | 60.54 (61.25, 58.02, 60.54) | **1.923** | >= 1.00, met | 1.036 | 111.1M / 97.2M | 0 / 17 | 0.289x / 0.551x |
| qwen2.5-3b | declined | 27.19 (27.19, 27.27, 27.14) | 27.67 (27.75, 27.67, 27.07) | **1.017** | >= 0.95, met | 0.967 | 143.6M / 143.6M | 0 / 0 | 0.404x / 0.416x |

**Gate met.** On every region model every on repetition is above every off repetition.

## How to read it

- Generation (64 tokens at a shallow context, `n_prompt=128`) is what the region changes; prefill does not run the
  decode region, and its on/off sits within 0.967 to 1.036.
- Greedy text over the 64 generated tokens is identical on against off on Phi-3.5-mini, Qwen3-1.7B and Qwen2.5-3B, and
  each mode is identical across its three runs on every model. On TinyLlama and Mistral 7B the on and off texts part
  within the 64 tokens, as at the earlier gates of this region (the region's norms sum in a different order than the
  default path's CPU norms); the 32-token on/off check of `smoke-gpu-residency.sh` holds on every handler.
- Qwen3-1.7B's GC max pause rises from 0 to 17 ms with the region on (one young collection inside the span while
  allocation per token falls 13%), as at the second part's gate; carried to the standing allocation gate.
- The GPU tg ratios are against the reference tool's reading in the same run, one repetition each; the closing sweeps
  score the end-of-plan targets.
