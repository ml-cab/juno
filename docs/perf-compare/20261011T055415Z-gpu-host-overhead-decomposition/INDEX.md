# GPU decode step and prefill window: where the non-kernel time goes - 20261011T055415Z (gpu)

**Purpose.** The decomposition published before any change to the GPU host overhead work: (a) a Phi-3.5-mini and
a Qwen2.5-3B decode step split into device-busy time and the host time around it, at prompts of about 64 and 512
tokens; (b) the Phi-3.5-mini 512- and 2048-token prefill window split by `prefill-breakdown.sh`, with its
unattributed residue located; (c) for each raised end-of-plan GPU row, the ratio this work can reach by the
arithmetic of (a) and (b) with everything else held.

**Pinned: no.** Every figure is a share or a duration read inside one run, not a gate. Unpinned (schedutil, turbo on).
Ratios below are projected onto the pinned reference column, never read from these runs.

**Build.** Jar `6ae135b0e21839cf` (HEAD `198467ad`, clean tree): the pre-work build, saved as
`dist/gpu-host-overhead/baseline-shaded.jar` for the later A/B.

## Method

**Decode (a): CUDA trace, not `--device-spans`.** `juno.DeviceCompute` and `juno.DeviceStaging` count decode-width
launches and copies but do not time them (the event's own documentation, and `timed_count = 0` on every decode site
of every earlier span run, e.g. `20261007T054314Z-gpu-residency-final-region-spans`). So the device side of a decode
step comes from a CUDA activity trace (Nsight Systems 2022.4, `-t cuda`, CPU sampling unavailable on this host), and
the host side from JFR. `tools/nsys-decode.sh` launches Juno with the harness's GPU-lane flags (`--threads 12`, fixed
heap, temperature 0), sends two warm-up requests (the second under a throwaway JFR recording), then one untraced
request, then one measured request under JFR (`juno-perf.jfc` plus `juno-perf-spans.jfc`) and, in the traced run, the
CUDA trace. Each configuration runs twice, untraced and traced, each request 128 generated tokens (`min_tokens` 128).

- `tools/decode_trace.py` splits the trace into decode steps at each logits download (Phi-3.5-mini 128,256 B,
  Qwen2.5-3B 607,744 B) and reports per step, median of 127 steps: wall, device-busy (union of kernel and copy
  intervals), kernel time by kernel, launches and copies, and the host's CUDA API time by call.
- `tools/decode_jfr.py` reads the untraced step from JFR: median `juno.ForwardPass` duration of the decode passes,
  and the median interval between `juno.TokenProduced` events (the whole step, sampling included).
- Tracing inflates host time (about 9 us per traced `cuLaunchKernel` on TinyLlama), not kernel durations. So the
  host terms are read untraced: **in-forward host = untraced forward pass - traced device-busy**, **between passes =
  untraced token interval - untraced forward pass**.

**Prefill (b).** `compare-llama-cpp.sh --gpu --device-spans --juno-reps 3 --no-tuned-lane --n-prompt 512` (and 2048 at
`COMPARE_HEAP=8g`), `--models Phi-3.5-mini-instruct-Q4_K_M --juno-jar dist/gpu-host-overhead/baseline-shaded.jar`, then
`prefill-breakdown.sh`. The residue is located by `tools/residue_gaps.py`, which takes the window's
`juno.PrefillBatch` event, subtracts every top-level span on the same thread, and reports the uncovered intervals by
position, the GC pauses inside them, and the allocation samples taken inside them.

## (a) Decode step, untraced, median of 127 steps

| Model, prompt tokens | Step ms (t/s) | Forward pass ms | Device-busy ms (kernels, copies) | In-forward host ms | Between passes ms | Launches / copies per step | Stream syncs per step |
|---|---|---|---|---|---|---|---|
| Phi-3.5-mini, 55 | 17.48 (57.2) | 16.97 | **16.37** (16.19, 0.17) | 0.60 (3.4%) | 0.52 (3.0%) | 705 / 35 | 33 |
| Phi-3.5-mini, 503 | 18.43 (54.3) | 17.93 | **17.42** (17.23, 0.18) | 0.51 (2.8%) | 0.50 (2.7%) | 705 / 35 | 33 |
| Qwen2.5-3B, 54 | 36.17 (27.7) | 33.90 | 14.35 (13.28, 1.07) | **19.56 (54.1%)** | 2.26 (6.2%) | 434 / 650 | 145 + 252 blocking copies |
| Qwen2.5-3B, 502 | 35.59 (28.1) | 33.04 | 14.75 (13.69, 1.07) | **18.29 (51.4%)** | 2.55 (7.2%) | 434 / 650 | 145 + 252 blocking copies |

Shares are of the step. Device-busy is from the traced run of the same configuration (`decode/*-trace.json`); step and
forward pass from the untraced run (`decode/*-untraced-step.json`).

**Phi-3.5-mini is bound by device kernels: 93.7% of the step at 55 tokens, 94.5% at 503.** Per layer: 22 launches,
one stream sync (the per-layer K, V and attention-output download), 512 us of device time. Kernels at 55 tokens
(16.19 ms): `q4k_gemv` 6.98 ms (43%), `q5k_gemv` 4.16 (26%), `q6k_gemv` 2.09 (13%), the LM head 0.93 (5.8%; cuBLAS
`gemv2T_kernel_val` over the FP16 output matrix, `Phi3TransformerHandler.outputProjection`), `gqa_attention_d128` 0.63
(3.9%; 1.82 ms at 503 tokens), `rms_norm` 0.40, `rope_ext_split_half` 0.27, `quantize_q8_1` 0.26, `swiglu` 0.13,
`to_half` 0.11. The largest term that is not device time is the in-forward host time, 0.60 ms: the gap the device
waits through at each of the 32 per-layer syncs while the host takes the download and launches the next layer.
Between passes, 0.52 ms: sampling and token handling; `SoftmaxStep.apply` is the most-sampled frame there (5 of 14
samples), a softmax over the full 32,064-entry vocabulary that greedy sampling does not read.

**Qwen2.5-3B runs outside the decode region, and the host holds more than half of its step.** 650 copies per token
(361 up, 289 down), 252 of them blocking `cudaMemcpy` and 399 `cudaMemcpyAsync` calls the host waits in. Of the
untraced execution samples, 87 of about 130 are the host SwiGLU loop (`LlamaTransformerHandler.ffn`, `silu(gate) * up`
over 11,008 elements, 36 layers per token) and 25 are `SoftmaxStep.apply` over the 151,936-entry vocabulary, which is
most of the 2.26 ms between passes. The device kernels themselves take 13.28 ms (`q4k_gemv` 8.35, `q6k_gemv` 4.18,
attention 0.43, `quantize_q8_1` 0.30).

## (b) Phi-3.5-mini prefill window, unpinned, median of three, `--device-spans`

| Term | 512 tokens, ms | Share | 2048 tokens, ms | Share |
|---|---|---|---|---|
| Window | 767.4 | | 3628.2 | |
| GEMM compute | 333.6 | 43.5% | 1376.5 | 37.9% |
| Region attention | 76.5 | 10.0% | 1099.2 | 30.3% |
| Device elementwise | 70.1 | 9.1% | 237.9 | 6.6% |
| Region host side (`projection_and_region_host`) | 84.4 | 11.0% | 263.7 | 7.3% |
| Host KV write (`kv_host_write`) | 69.3 | 9.0% | 249.8 | 6.9% |
| Matmul staging | 42.3 | 5.5% | 164.1 | 4.5% |
| Embed and LM head | 2.7 | 0.3% | 6.3 | 0.2% |
| **Residue** | **84.3** | **11.0%** | **226.4** | **6.2%** |

`prefill-breakdown.sh` flags both residues (limit 5%); `prefill512/`, `prefill2048/`.

**The residue is attributed.** In every repetition at both lengths, 96% to 98% of it is one interval: from the end of
the `embed` span to the start of the first `device_layer` span (80.8 to 89.2 ms at 512 tokens, 200.3 to 227.6 ms at
2048; `prefill*/residue-gaps-rep*.txt`). The rest is about 1.7 ms of span boundaries spread over 64 layers. That
interval is `Phi3TransformerHandler.runLayersBatch` before its first layer:

- **Host KV growth.** Each request's 64 host KV tensors (32 layers, K and V; Phi-3.5-mini has 32 KV heads, so
  `kvDim` 3072) start at 64 positions, and `ensureCapacity` grows each by doubling with a copy (`DenseKvTensor.grow`)
  to the window's length. Every allocation sample inside the interval is `DenseKvTensor.grow` or
  `DenseKvTensor.allocate` (120 to 160 MB sampled weight at 512 tokens; 1.6 to 1.7 GB at 2048, where the final tensors
  alone are 64 x 2048 x 3072 x 4 B = 1.61 GB), plus the `BatchWorkspace` arrays at 2048.
- **GC pauses** that this allocation triggers: 24.2 to 28.3 ms at 512 tokens and 37.3 to 56.2 ms at 2048, every one
  of them inside the interval.

Mistral 7B (8 KV heads, a 1024-float row) allocates a third of this and Qwen2.5-3B (2 KV heads, 36 layers) about a tenth, which is why their residues read 3.3%
and 3.7% at Tier 02's close.

## (c) What each raised row can reach

Projected onto the pinned reference column (`20261008T173153Z`, `20261008T174605Z`, `20261008T181417Z`), with Tier
02C's pinned GPU A/B (`20261011T052951Z-cpu-hot-path-close-gate`, Part A) applied where it moved the reading, and the
fraction of Juno's time this work removes taken from (a) and (b).

| Row | Reference | Needed | Projection | Reachable |
|---|---|---|---|---|
| GPU tg >= 0.70x, every sweep model (Qwen2.5-3B binding) | 0.416x (x 1.020 after Tier 02C: about 0.42x); 34.9 ms per token pinned | 20.7 ms per token | Qwen2 in the decode region (item 2): device 13.3 ms of kernels + about 1.1 ms of region elementwise kernels (scaled from Phi-3.5-mini's 1.4 ms) + about 0.7 ms in-forward host (Phi-3.5-mini's 0.60 per 32 layers, at 36) + 2.3 ms between passes = **about 17.4 ms, 0.80x to 0.83x** | **yes, by item 2 alone**. Other models: TinyLlama 0.770x (0.79x after 02C), Phi-3.5-mini and Mistral 7B above 0.95x |
| GPU tg >= 0.90x, Phi-3.5-mini | 0.858x | | **already met before this work, by projection**: 0.858x x 1.127 (Tier 02C's pinned A/B) = **0.967x**. The two prefill runs here, unpinned, same session, read Phi-3.5-mini tg at **0.977x** and **0.955x** against the reference tool with span recording on | met; see the decision this raises. Bound if every non-device term were removed: 17.48 to 16.37 ms, about 1.03x |
| GPU pp >= 0.70x at 512, every sweep model (Phi-3.5-mini binding) | 0.594x (x 1.001 after 02C) | window -15.1% | residue alone (11.0%): 0.667x. Residue plus half of region host, host KV write and staging (12.8%): **0.78x**. All four terms: 0.94x | **yes, but not by the residue alone**: about 4% more has to come from the other three terms. Other models already 0.75x to 0.82x |
| GPU pp >= 0.60x at 2048, every sweep model (Phi-3.5-mini binding) | 0.540x | window -10.0% | residue alone (6.2%): 0.576x. Residue plus half of the other three (9.4%): **0.64x**. All four: 0.72x | **yes, but not by the residue alone**. Other models already 0.68x to 0.71x |

The prefill shares are unpinned. Pinned, the host terms are a larger share (turbo off slows the host and not the
device), so the projections are conservative for the host terms.

**Ranking for the work that follows.** Item 2 (Qwen2 in the region) first: it moves the binding generation row from
about 0.42x to about 0.8x. Item 3 second: both prefill rows need the residue (host KV growth and the GC it causes)
removed and then about 4% more from the region host side, the host KV write and staging. Item 4 last: the Phi-3.5-mini
decode step is kernel-bound at 93.7%, its largest non-kernel term is 0.60 ms (3.4%), and its row is already met.

## Files

- `decode/`: per configuration the trace summary (`*-trace.json`, `*-trace.txt`), the untraced and traced step from
  JFR (`*-untraced-step.json`, `*-traced-step.json`), the measured responses, and the untraced top execution frames.
- `prefill512/`, `prefill2048/`: the harness's JSON and `RUN-INDEX.md` (its own index of the run, with the
  same-session reference-tool rows), `pp*-breakdown.md`/`.json`, `residue-gaps-rep*.txt`.
- `tools/`: the four scripts above. Traces and JFR recordings are not kept (hundreds of MB); the scripts regenerate them.

Local paths are replaced by `<run dir 512>`, `<run dir 2048>`, `<scratch>` and `<reference-tool-bin>`, and the
repository root is stripped.
