# Qwen3 prefill window: per-head norm, RoPE and attention inside the device region - 20261007T065243Z (gpu)

**Purpose.** Reads what a 512-token Qwen3-1.7B prefill window moves across the bus after the prefill-window
device region took over Qwen3's per-head Q and K RMS norm, its RoPE and its attention. Before, the region
handed Q, K and V back to the host every layer, the host normalized each head, rotated and attended, and the
attention output was uploaded again. Also the breakdown of the window's time.

**Pinned: no.** The bytes figure does not depend on clocks (identical on all three repetitions). The
throughput figures are unpinned and are a reading, not a gate.

**Command.**

```
scripts/performance-tests/compare-llama-cpp.sh --gpu --device-spans --juno-reps 3 --no-tuned-lane \
  --n-prompt 512 --models Qwen3-1.7B-Q4_K_M --juno-jar dist/qwen3-prefill-region-ab/candidate-shaded.jar
scripts/performance-tests/prefill-breakdown.sh <run dir> --json prefill-breakdown.json > prefill-breakdown.md
```

The jar is this working tree's build (`juno_commit` d05d1fe plus the uncommitted change, `juno_tree_dirty:
true`), sha256 `533b1864d682bea5`. Copied by hand from the harness output, JSON and Markdown only, local paths
made repository-relative, the reference tool's install directory replaced by `<reference-tool-bin>`.
`RUN-INDEX.md` is the harness's own index. Default flags (`--gpu-residency` off), so the generate lane's
decode copies are the op-at-a-time path's and are not what this run reads.

## Bytes per 512-token window (prefill phase, `juno.DeviceStaging`)

| Reading | H2D MB | D2H MB | H2D + D2H MB |
|---|---|---|---|
| Before (region with the head norm, RoPE and attention on the host; `20261007T054314Z-gpu-residency-final-region-spans`) | 297.5 | 355.8 | 653.3 |
| **This run** (each of 3 repetitions) | **4.5** | **121.4** | **125.9** (-80.7%) |

MB is 10^6 bytes. Qwen3-1.7B is not a sweep model and carries no bytes threshold. What still crosses: the K and V
rows of every layer for the host KV cache (`materialize(prefill k)`, `materialize(prefill v)`, 58.6 MB each), the
residual once each way (4.2 MB each) and the attention tables (0.3 MB). No Q row, attention output, or copy of the
host-fed attention call remains.

## Prefill breakdown (median of 3, 512-token window)

See `prefill-breakdown.md`. Prefill 306.8 ms (680.3 ms in the run above, which had span recording on as well). Host
norm, host RoPE, host attention and attention copies all 0; attention runs as `region_attention_compute` (38.0 ms,
12.4%; was 157.3 ms, 23.1%, for compute, copies and host part). The breakdown script exits 1 on its residue flag
(8.5%, 26.0 ms, against 5%); reported, not relaxed.

## Throughput (unpinned, span recording on, a reading only)

| Model | Juno pp t/s | Juno tg t/s | Juno/ref pp at 512 | Juno/ref tg |
|---|---|---|---|---|
| Qwen3-1.7B | 1668.9 | 37.0 | 0.595x (0.316x before) | 0.321x (region off) |

The before/after A/B on the same jars is `20261007T065442Z-qwen3-prefill-region-ab-unpinned`.

## How to read it

Bytes: `models[0].metrics["juno.DeviceStaging.{H2D,D2H}.prefill.bytes"]` in `*-prefill-rep*-juno-jfr.json`, per
site from the `juno.DeviceStaging.site.*.prefill.bytes` keys of the same files.
