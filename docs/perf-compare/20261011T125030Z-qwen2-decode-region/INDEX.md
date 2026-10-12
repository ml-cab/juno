# Qwen2.5-3B decode in the device-resident region - 20261011T125030Z (gpu)

**Purpose.** The re-measurement after Qwen2 joined the decode region (split-half RoPE and the Q, K and V biases on the
device), against the decomposition taken on the build before it
([`20261011T055415Z-gpu-host-overhead-decomposition`](../20261011T055415Z-gpu-host-overhead-decomposition/INDEX.md)),
same method, same scripts (that directory's `tools/`), same configurations. Plus the residency smoke on the new build.

**Pinned: no.** A reading, not a gate. The no-regression gate and the ratio against the reference engine are the tier's
pinned closing runs.

**Builds.** Before: jar `6ae135b0e21839cf` (HEAD `198467ad`). After: jar `662d602e5491a67a` (`198467ad` plus the region
change, uncommitted tree).

## Decode step, untraced, median of 127 steps, 128 generated tokens

| Prompt tokens | Step ms before | Step ms after | Speedup | Forward pass before / after | Device-busy before / after | Copies per token before / after |
|---|---|---|---|---|---|---|
| 54 | 36.17 (27.7 t/s) | **17.11 (58.5 t/s)** | **2.11x** | 33.90 / 15.12 | 14.35 / 14.50 | 361 up, 289 down / **2 up, 37 down** |
| 502 | 35.59 (28.1 t/s) | **17.46 (57.3 t/s)** | **2.04x** | 33.04 / 15.41 | 14.75 / 15.10 | 361 up, 289 down / **2 up, 37 down** |

The decomposition projected about 17.4 ms per token. Device kernels are now 14.36 ms of the 17.11 (`q4k_gemv` 8.05,
`q6k_gemv` 4.18, attention 0.51, `rms_norm` 0.37, `quantize_q8_1` 0.29, `rope_split_half` 0.22, `add_bias` 0.21,
`swiglu` 0.17). What remains outside them: about 0.6 ms in the forward pass (one stream sync per layer) and 2.0 ms between
passes, where the full-vocabulary softmax that greedy sampling does not read is the most-sampled frame (left to the
sampling work, owner decision).

Read against the reference engine's pinned 68.89 t/s (`20261008T173153Z`), 58.5 t/s would be about 0.85x; that mixes an
unpinned reading with a pinned one and is indicative only.

## Greedy output

- `smoke-gpu-residency.sh` (unmodified, `--models tinyllama-1.1b-chat-v1.0.Q4_K_M,qwen2.5-3b-instruct-q4_k_m`):
  Qwen2.5-3B reports the region active on every layer, greedy output identical on and off over 32 tokens, no device
  memory growth across requests; TinyLlama and its pipeline and tensor cluster runs pass as before
  (`smoke-gpu-residency-summary.md`).
- `LlamaTransformerHandlerGpuResidencyTest`: on against off over the model's own 19-token greedy continuation of a
  5-token prompt, the same token at every position, logits within 0.51.
- On the two "x x x ..." prompts above, the 128-token replies part from the earlier build's after about 25 tokens (54-token
  prompt) and after about 4 (502-token prompt); both replies are coherent. The region normalizes on the GPU and the
  op-at-a-time path on the CPU (a different summation order), which moves Q8_1 roundings, the same class of difference
  the region already shows on TinyLlama and Mistral 7B.

## Files

`qwen2-c*-after-trace.json` (CUDA trace per step), `qwen2-c*-after-untraced-step.json` and `-traced-step.json` (JFR),
`qwen2-c*-after-untraced-response.json`, `smoke-gpu-residency-summary.md`. Local paths removed.
