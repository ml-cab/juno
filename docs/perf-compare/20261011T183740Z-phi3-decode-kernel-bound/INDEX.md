# Phi-3.5-mini decode step on the current build: bound by device kernels - 20261011T183740Z (gpu)

**Purpose.** Item 4's reading on the build that carries this tier's changes (jar `921ab742ba3f362a`: `198467ad` plus Qwen2's
decode region, host KV reuse and the direct K/V write). By owner decision 1 (a), item 4 closes on the measured step:
whether a non-kernel term is left to remove, or the step is bound by device kernels. Same method and scripts as
[`20261011T055415Z-gpu-host-overhead-decomposition`](../20261011T055415Z-gpu-host-overhead-decomposition/INDEX.md)
(`tools/`): an untraced run for the step and the host terms, a CUDA-traced run for device-busy time, 128 generated
tokens, median of 127 steps.

**Pinned: no.** A reading, not a gate.

| Prompt tokens | Step ms (t/s) | Device-busy ms | Share | In-forward host ms | Between passes ms | Pre-tier step / device-busy share |
|---|---|---|---|---|---|---|
| 55 | 17.25 (58.0) | 16.32 (kernels 16.15, copies 0.17) | **94.6%** | 0.42 | 0.51 | 17.48 / 93.7% |
| 503 | 18.76 (53.3) | 17.56 (kernels 17.39, copies 0.17) | **93.6%** | 0.65 | 0.55 | 18.43 / 94.5% |

**The step is bound by device kernels.** At 55 prompt tokens, of the 16.15 ms of kernels: `q4k_gemv` 6.98 ms (43%),
`q5k_gemv` 4.16 (26%), `q6k_gemv` 2.09 (13%), the LM head's FP16 GEMV (cuBLAS `gemv2T_kernel_val`) 0.93 (5.8%), attention
0.61 (3.8%; 1.81 ms at 503 tokens). What is not device time is about 6% of the step, in two terms of about 0.5 ms each:
the host's turn at each of the 32 per-layer downloads (one stream sync per layer), and the time between passes, where the
full-vocabulary softmax that greedy sampling does not read is the most-sampled frame (left to the sampling work by owner
decision). Removing every non-kernel term would bring the step to about 16.3 ms, at most about 1.06x.

The step moved within run-to-run variation against the pre-tier build (17.48 to 17.25 ms at 55 tokens, 18.43 to 18.76 at
503): nothing in this tier changes Phi-3.5-mini's decode path.

Files: `phi3-c*-trace.json` (CUDA trace per step), `phi3-c*-untraced-step.json` (JFR), `phi3-c*-untraced-response.json`.
