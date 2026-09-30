# GPU attention on the Phi-3 and Qwen3 handlers: same-build flag A/B (preliminary, unpinned)

> **Superseded** by the pinned [`20260930T205554Z-tier01b-item0-ab`](../20260930T205554Z-tier01b-item0-ab/INDEX.md), which reads 6.55x and 6.10x at 512 tokens where this run read 6.5x to 6.9x and 5.8x. Kept as the record of the first reading.

**Not scorable against any gate.** Clocks were not pinned (`clock_pinned: false`, governor `schedutil`,
turbo on): the agent shell that took these runs had no prompt-free sudo. They show the direction and
size of the effect and that the kernel is taken; the gate reading for this change is a pinned run.

- Build: working tree on `ffd0ca7` (dirty: the change under test), shaded jar sha256 `d701031b0acf7d2d`.
- Harness: `compare-llama-cpp.sh --gpu --models Phi-3.5-mini,Qwen3-1.7B --n-prompt 512 --juno-warmup 2
  --juno-reps 3 --reps 3 --no-tuned-lane --no-publish --gpu-attention {off|on}`, four invocations
  alternating off, on, off, on, 2026-09-30 13:07 to 13:57 local. GTX 1080, JFR on the operating-system
  clock.
- The only difference between lanes is `--gpu-attention`: `off` is the scalar CPU attention these two
  handlers ran before this change; `on` is the GPU-resident attention kernel over an FP16 KV mirror.
- `gpu_attention_resolved` reads `unknown` in every lane, because none carries `--device-spans` and the
  console suppresses the handler's activation log line. The kernel being taken is shown by the parity
  tests (`GpuAttentionHandlerParityTest`) and by decode moving with it, below.
- Every row is scorable by the harness's own rule (repetitions within 15%), and every Juno prefill is
  512/512 tokens.

Juno medians of three, t/s; ratio against the reference tool's reading in the same invocation:

| Model | Lane | pp t/s (off / on) | pp gain | pp ratio (off / on) | tg t/s (off / on) | tg gain |
|---|---|---|---|---|---|---|
| Phi-3.5-mini Q4_K_M | pair 1 | 14.21 / 91.98 | 6.47x | 0.0118x / 0.0775x | 24.33 / 32.87 | 1.35x |
| Phi-3.5-mini Q4_K_M | pair 2 | 13.21 / 91.39 | 6.92x | 0.0115x / 0.0746x | 24.18 / 32.72 | 1.35x |
| Qwen3-1.7B Q4_K_M | pair 1 | 30.75 / 177.07 | 5.76x | 0.0110x / 0.0658x | 35.61 / 39.50 | 1.11x |
| Qwen3-1.7B Q4_K_M | pair 2 | 30.40 / 176.73 | 5.81x | 0.0109x / 0.0685x | 34.84 / 38.68 | 1.11x |

Runs: `1-off-20260930T180709Z/`, `2-on-20260930T182137Z/`, `3-off-20260930T182653Z/`,
`4-on-20260930T184140Z/` (JFR recordings not kept, as in other published runs).
