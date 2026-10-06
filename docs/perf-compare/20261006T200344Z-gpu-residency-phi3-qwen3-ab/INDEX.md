# Decode residency region on Phi-3 and Qwen3: pinned A/B - 20261006T200344Z (gpu, clocks pinned, owner run)

**Purpose.** The gate of the change that runs the whole decode layer inside the device residency region
on the Phi-3 and dense Qwen3 handlers as well as the LLaMA family: `--gpu-residency` off against on, one
jar. Thresholds, Juno tg t/s medians of three: on/off **>= 1.00** where the region runs (TinyLlama,
Mistral 7B, Phi-3.5-mini, Qwen3-1.7B) and **>= 0.95** everywhere (Qwen2.5-3B declines the region and
reads the flag's cost where it does nothing). pp on/off recorded, not gated (the region is decode-only).

**Pinned: yes**, on all six runs (`clock_pinned: true` in every `host.json`; CPU governor performance,
turbo off; the GPU clock is recorded, not locked, as on every run on this card). Owner run.

**Command.** `bash dist/gpu-residency-phi3-qwen3-ab/run-gate.sh` (copied here as `run-gate.sh`): the flag
alternated off, on, off, on, off, on from 20:03Z to 20:20Z, each run `compare-llama-cpp.sh --gpu --pin-clocks
--models <five models> --n-prompt 128 --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish
--gpu-residency off|on --juno-jar <jar>`, GPU attention at its default (on under CUDA). The console output
was not kept; the scores below were recomputed from the run files with the script's own method.

**Jar.** sha256 `9659ff62d527578b`, the working tree of the change (committed afterwards as 5e4c913), on
both sides.

## Results (Juno t/s, median of three; all readings)

| Model | Region | tg off | tg on | tg on/off | Threshold | pp on/off (not gated) | alloc/token off / on | GC max ms off / on | GPU tg ratio off / on |
|---|---|---|---|---|---|---|---|---|---|
| tinyllama-1.1b | whole layer | 64.00 (64.26, 64.00, 63.73) | 131.47 (131.47, 132.11, 129.25) | **2.054** | >= 1.00, met | 0.913 | 47.9M / 38.8M | 6-7 / 0 | 0.329x / 0.672x |
| mistral-7b | whole layer | 22.16 (21.85, 22.16, 22.38) | 36.54 (36.76, 36.54, 36.19) | **1.649** | >= 1.00, met | 0.990 | 223.8M / 191.7M | 13-14 / 0 | 0.596x / 0.982x |
| Phi-3.5-mini | whole layer | 31.22 (30.51, 31.32, 31.22) | 48.22 (48.40, 48.13, 48.22) | **1.545** | >= 1.00, met | 1.001 | 207.5M / 180.7M | 7-8 / 9 | 0.515x / 0.799x |
| Qwen3-1.7B | whole layer | 37.25 (37.44, 37.25, 37.17) | 63.18 (63.62, 62.82, 63.18) | **1.696** | >= 1.00, met | 0.972 | 111.8M / 97.7M | 0 / 15-19 | 0.320x / 0.546x |
| qwen2.5-3b | declined | 28.78 (28.94, 28.78, 28.77) | 28.95 (29.28, 28.55, 28.95) | **1.006** | >= 0.95, met | 0.989 | 143.3M / 143.2M | 0 / 0 | 0.404x / 0.407x |

Every row scorable on all six runs. On every region model every on repetition is above every off repetition.

Two readings, not gated: Qwen3-1.7B's GC max pause rises from 0 to 15-19 ms with the region on (one young
collection now inside the 64-token span while allocation per token falls 13%; the unpinned run showed the
same); TinyLlama's pp on/off reads 0.913 although prefill does not run the decode region (the whole-layer
gate read 0.901 the same way).

## How to read it

Each `off-N` / `on-N` directory is one harness run (JSON and Markdown only; local paths made
repository-relative, the reference tool's install directory replaced by `<reference-tool-bin>`). The t/s
figures are `prompt_eval_tps` and `token_gen_tps` in `<model>-juno.json`; the ratio, GC and allocation
columns are each run's `INDEX.md` row (allocation is per generated token in the generation run). The
ratios against the reference tool are from a gate A/B with one repetition per run at `n_prompt=128`, not
the closing sweeps the end-of-plan targets are scored on.
