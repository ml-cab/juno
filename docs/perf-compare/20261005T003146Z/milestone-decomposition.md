# 2048-over-512 prefill milestone: decomposition (2026-10-05)

Purpose: before a tiled attention kernel is built, read off whether attention alone can move the GPU
prefill ratio at `n_prompt=2048` to at least 0.90 of the ratio at `n_prompt=512` on every sweep model.

Sources: the 512 run in this directory and the 2048 runs
[`20261005T004009Z`](../20261005T004009Z/INDEX.md) (TinyLlama, Qwen2.5-3B, Mistral 7B) and
[`20261005T005334Z`](../20261005T005334Z/INDEX.md) (Phi-3.5-mini, 8 GiB heap). All three: jar
`3306ea4261f84a49`, `--device-spans`, default lane, clocks not pinned, median of three repetitions
per term (`prefill-breakdown.sh`). Ratios below are this run's own Juno over reference-tool pp, so the
spans overhead sits on both lengths; the pinned references (`20261004T114812Z`, `20261004T220015Z`,
`20261004T222758Z`) give 2048 over 512 of 0.329 to 0.441, and the readings here (0.307 to 0.452) agree
within the noise floor.

Attention = `region_attention_compute` + `attention_compute` + `attention_copies` + `attention_host`
(kernel, copies and host part). On the three LLaMA-family models attention runs inside the prefill
region, where its host dispatch is not separable and stays in `projection_and_region_host` (0.4% to 3.4%
of the window); on Phi-3.5-mini it runs outside the region and all three parts are separate.

| Model | Prefill ms 512 / 2048 | Attention ms 512 / 2048 | Attention share 512 / 2048 | Non-attention ms per token 512 / 2048 (growth) | pp ratio 512 / 2048 (2048 over 512) | Attention speedup at 2048 needed, 512 held | Speedup needed at both lengths | 2048 over 512 with attention at zero |
|---|---|---|---|---|---|---|---|---|
| Phi-3.5-mini-instruct-Q4_K_M | 2072 / 24334 | 943 / 19900 | 45.5% / 81.8% | 2.206 / 2.165 (-1.9%) | 0.200x / 0.091x (0.452) | 2.56x | 6.44x | 1.352 |
| mistral-7b-instruct-v0.1-q4_k_m | 1591 / 23226 | 757 / 19253 | 47.6% / 82.9% | 1.628 / 1.940 (+19.2%) | 0.468x / 0.144x (0.307) | 4.86x | 84.35x | 0.942 |
| qwen2.5-3b-instruct-q4_k_m | 765 / 8933 | 408 / 7417 | 53.3% / 83.0% | 0.697 / 0.740 (+6.1%) | 0.415x / 0.169x (0.407) | 2.94x | 14.24x | 1.119 |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 400 / 5060 | 248 / 4459 | 62.1% / 88.1% | 0.296 / 0.294 (-0.7%) | 0.330x / 0.125x (0.378) | 2.93x | 15.48x | 1.204 |

How to read the three speedup columns:

- **Attention speedup at 2048 needed, 512 held**: the factor by which attention at 2048 must get faster
  for 2048 over 512 to reach 0.90 when nothing else moves, the 512 reading included. This is the figure
  the step's 4x escalation rule reads.
- **Speedup needed at both lengths**: the same, but the faster kernel also runs at 512 (it does), which
  raises the 512 ratio the milestone divides by. "unreachable" would mean no finite factor suffices.
- **2048 over 512 with attention at zero**: the ceiling of the milestone ratio if attention cost nothing at
  either length. On Mistral 7B it is 0.942, barely above 0.90, because its non-attention time per token
  grows 19.2% from 512 to 2048.

Reference-tool retention for context (its own pp t/s at 2048 over 512, this run): TinyLlama 0.836,
Qwen2.5-3B 0.842, Phi-3.5-mini 0.754, Mistral 7B 0.891. Juno's: 0.316, 0.342, 0.341, 0.274. The milestone
asks Juno to keep at least 0.90 of the reference tool's retention.

**Mistral 7B's non-attention growth.** Both lengths run one prefill window with the same launch counts
(224 packed K-quant GEMMs, 32 attention calls), yet GEMM compute grows 650 ms to 3,242 ms (4.98x for 4x the
rows) and every elementwise kernel 4x to 6x. TinyLlama shows no such growth (-0.7% per token). The cause is
not established by these runs: the GPU clock falling over a 23-second window (clocks unpinned; this card
refuses clock locking anyway) and the GEMM's behaviour at 2048 rows are both consistent with it.
