# Final decode region: copies per token and the prefill window - 20261007T054314Z (gpu)

**Purpose.** Reads, on the final decode residency region (`--gpu-residency on`: the whole decode layer on
the device on the LLaMA-family, Phi-3 and Qwen3 handlers, the residual row kept on the device between
layers), how many host-device copies one decoded token makes, on every model where the region runs, and
re-verifies on all four sweep models that a 512-token prefill window moves no activation to the host apart
from the K and V rows for the host KV cache and the window's final residual. Also the breakdown of the
512-token window's time.

**Pinned: no.** Copy counts and bytes do not depend on clocks (identical on all three repetitions of every
model). The throughput figures are unpinned, taken with span recording on, and are a reading, not a gate.

**Command.**

```
scripts/performance-tests/compare-llama-cpp.sh --gpu --device-spans --juno-reps 3 --no-tuned-lane \
  --n-prompt 512 --gpu-residency on \
  --models tinyllama-1.1b-chat-v1.0.Q4_K_M,qwen2.5-3b-instruct-q4_k_m,Phi-3.5-mini-instruct-Q4_K_M,mistral-7b-instruct-v0.1-q4_k_m,Qwen3-1.7B-Q4_K_M \
  --juno-jar dist/gpu-residency-final-region-ab/candidate-shaded.jar
scripts/performance-tests/prefill-breakdown.sh <run dir> --json prefill-breakdown.json > prefill-breakdown.md
```

The jar is HEAD `d05d1fe` built with `mvn -q package -DskipTests` on a clean tree, sha256 `652bf47e12609973`.
Because `--juno-jar` was given, the harness did not publish; this directory was copied from its output by
hand, JSON and Markdown only, with local paths made repository-relative and the reference tool's install
directory replaced by `<reference-tool-bin>`. `RUN-INDEX.md` is the harness's own index of the run.

## Decode copies per generated token (generate lane, `juno.DeviceStaging`, decode phase)

Threshold (scope item 8): host-to-device copies `<= layers`, device-to-host `<= layers + 1` (the `+ 1` is
the logits).

| Model | Layers | Region | Raw H2D / D2H per decode pass | Excluded (not decode work) | **H2D per token** | **D2H per token** | Threshold |
|---|---|---|---|---|---|---|---|
| tinyllama-1.1b | 22 | whole layer | 2.016 / 23.016 | the prefill pass's LM head (1 pair) | **2** | **23** | <= 22 / <= 23, met |
| Phi-3.5-mini | 32 | whole layer | 2.016 / 33.016 | the prefill pass's LM head (1 pair) | **2** | **33** | <= 32 / <= 33, met |
| Qwen3-1.7B | 28 | whole layer | 2.016 / 29.016 | the prefill pass's LM head (1 pair) | **2** | **29** | <= 28 / <= 29, met |
| mistral-7b | 32 | whole layer | 30.016 / 61.016 | the prefill pass's LM head (1 pair) and the 8-row host prefill window (1,792 pairs) | **2** | **33** | <= 32 / <= 33, met |
| qwen2.5-3b | 36 | declined (split-half RoPE, Q/K/V biases; announced) | 392.5 / 320.5 | | 392.5 | 320.5 | not applicable: the region does not run |

Each figure is identical on all three repetitions. 64 decode passes per request.

**Two exclusions, both from the rule that assigns a copy its phase.** A copy is tagged with the row count of
the forward call that issued it: more than one row is prefill, one row is decode.

- The prefill window computes logits for its last row only, so its LM head (one upload of the last hidden row,
  one download of the logits) is a one-row call and lands in the decode bucket. Every model shows 65 LM-head
  copy pairs against 64 decode passes, and none in the prefill phase.
- The generation request's chat-template prompt is short (9 tokens on Mistral 7B and Qwen2.5-3B, 10 on
  Phi-3.5-mini, 13 on Qwen3-1.7B, 19 on TinyLlama). Its prefill window is the prompt less its last token, and
  a window of 8 rows or fewer runs on the host path, one row at a time through the packed GEMV
  (`PrefillWindowRegion.MAX_HOST_WINDOW`); those calls are one-row calls and land in the decode bucket too.
  Among the region models only Mistral 7B's window is that short (8 rows). The
  excess is exactly 8 rows x 32 layers x 7 projections = 1,792 copy pairs, and its bytes match to the byte:
  per row and layer the downloads are q 16,384 + k 4,096 + v 4,096 + output 16,384 + gate 57,344 + up 57,344
  + down 16,384 = 172,032 B (x 256 = 44,040,192 B measured), the uploads 6 x 16,384 + 57,344 = 155,648 B (x 256
  = 39,845,888 B measured).

The region sites themselves are exact on every region model: `upload(decode region input)` 64 (once per
token), `materialize(decode region k v attention layer output)` 64 x layers (once per layer per token).

## Bytes per 512-token prefill window (prefill lane, prefill phase)

| Model | H2D MB | D2H MB | What crosses to the host |
|---|---|---|---|
| tinyllama-1.1b | 4.4 | 27.2 | K and V rows (11.5 MB each), the final residual (4.2 MB) |
| qwen2.5-3b | 4.6 | 41.9 | K and V rows (18.8 MB each), the final residual (4.2 MB) |
| Phi-3.5-mini | 6.6 | 408.1 | K and V rows (200.9 MB each), the final residual (6.3 MB) |
| mistral-7b | 8.7 | 142.3 | K and V rows (67.0 MB each), the final residual (8.4 MB) |
| Qwen3-1.7B | 297.5 | 355.8 | Q rows down, attention output up, the attention kernel's own window copies, K and V rows, the final residual |

MB is 10^6 bytes; identical on all three repetitions. Uploads on the four sweep models are the residual once
and the attention pointer tables (0.2 to 0.4 MB). On the four sweep models no activation leaves the device
inside the window apart from the K and V rows for the host KV cache, which stays the source of truth, and the
window's final residual. Qwen3-1.7B runs RoPE and attention on the host side of the prefill region, as
documented (its per-head Q/K norm is in the decode region, not the prefill region); it is not a sweep model.
The `memcpy(prefill residual layer input d2d)` site is a device-to-device copy.

## Prefill breakdown (median of 3, 512-token window)

See `prefill-breakdown.md`. Attention's share of the window (on the region models, `region_attention_compute`):

| Model | Prefill ms | Attention ms | Share | Share at the pre-tier decomposition (`20261005T003146Z`) |
|---|---|---|---|---|
| tinyllama-1.1b | 179.5 | 22.7 | 12.6% | 62.1% |
| qwen2.5-3b | 433.8 | 48.9 | 11.3% | 53.3% |
| Phi-3.5-mini | 753.3 | 75.6 | 10.0% | 45.5% |
| mistral-7b | 984.1 | 87.8 | 8.9% | 47.6% |
| Qwen3-1.7B | 680.3 | 157.3 (compute 40.9, copies 52.4, host 64.0) | 23.1% | not decomposed |

The breakdown script exits 1 on its residue flag: Phi-3.5-mini 10.8% (81.5 ms; 10.9% at
`20261006T184154Z-phi3-prefill-region-attention`) and TinyLlama 6.8% (12.2 ms), against 5%. Reported; the flag
was not relaxed.

## Throughput (unpinned, span recording on, a reading only)

| Model | Juno pp t/s | Juno tg t/s | Juno/ref pp at 512 | Juno/ref tg |
|---|---|---|---|---|
| tinyllama-1.1b | 2852.7 | 123.0 | 0.736x | 0.679x |
| qwen2.5-3b | 1180.2 | 28.1 | 0.766x | 0.410x |
| Phi-3.5-mini | 679.7 | 46.6 | 0.604x | 0.843x |
| mistral-7b | 520.3 | 34.7 | 0.793x | 1.000x |
| Qwen3-1.7B | 752.6 | 60.5 | 0.316x | 0.630x |

Unpinned and with `--device-spans` on, which costs a few percent of prefill; not comparable with the pinned
sweeps the program targets are scored on. Mistral 7B's GPU clock reached thermal slowdown (low 1607 MHz).

## How to read it

Every file is one repetition's harness output (`*-prefill-rep*`, `*-generate-rep*`) or its aggregate. Decode
copies: `juno.DeviceStaging.{H2D,D2H}.decode.count` over `juno.ForwardPass.decode.count` in
`*-generate-rep*-juno-jfr.json`, per site from `juno.DeviceStaging.site.<site>.decode.{count,bytes}`. Window
bytes: `juno.DeviceStaging.{H2D,D2H}.prefill.bytes` and the per-site `...prefill.bytes` keys of
`*-prefill-rep*-juno-jfr.json`.
