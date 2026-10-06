# Phi-3 prefill window: RoPE and attention inside the device region - 20261006T184154Z (gpu)

**Purpose.** Reads what a 512-token Phi-3.5-mini prefill window moves across the bus after the
prefill-window device region took over Phi-3's RoPE and attention: the fused Q/K/V rows are split on
the device, rotated by the extended-RoPE kernel, cast into the attention KV mirror and attended there.
Before, the region handed Q, K and V back to the host every layer and took the attention output up
again. Also the breakdown of the window's time.

**Pinned: no.** The bytes figure does not depend on clocks (identical on all three repetitions). The
throughput figures are unpinned and are a reading, not a gate.

**Command.**

```
scripts/performance-tests/compare-llama-cpp.sh --gpu --device-spans --juno-reps 3 --no-tuned-lane \
  --n-prompt 512 --models Phi-3.5-mini --juno-jar juno-player/target/juno-player-0.1.2-shaded.jar
scripts/performance-tests/prefill-breakdown.sh <run dir> --json prefill-breakdown.json > prefill-breakdown.md
```

The jar is this working tree's build (`juno_commit` 5e4c913 plus the uncommitted change,
`juno_tree_dirty: true`), sha256 `326cbb9ff2bd86e2`. Because `--juno-jar` was given, the harness
did not publish; this directory was copied from its output by hand, JSON and Markdown only, with
local paths made repository-relative and the reference tool's install directory replaced by
`<reference-tool-bin>`. `RUN-INDEX.md` is the harness's own index of the run.

## Bytes per 512-token window (prefill phase, `juno.DeviceStaging`)

| Reading | H2D MB | D2H MB | H2D + D2H MB | Against 2,645 MB |
|---|---|---|---|---|
| Before the region (2026-09-30 baseline) | | | 2,645 | |
| Region, RoPE and attention on the host (`20261002T050741Z`) | | | 1,419 | -46.4% |
| **This run** (each of 3 repetitions) | **6.6** | **408.1** | **414.8** | **-84.3%** |

MB is 10^6 bytes. Threshold: at least 70% below 2,645 MB, that is at most 793.5 MB. **Met.**

What still crosses, per window: the K and V rows of every layer for the host KV cache, which stays the
source of truth (`materialize(prefill k)` and `materialize(prefill v)`, 200.9 MB each), the residual
stream once up and once down (`upload(prefill residual)`, `materialize(prefill residual)`, 6.3 MB
each), and the attention pointer tables (`memcpy(region attention tables H2D)`, 0.3 MB). No Q, fused
Q/K/V or attention-output copy remains, and no activation leaves the device inside the window apart
from the K and V rows and the window's final residual.

## Prefill breakdown (median of 3, ms per 512-token window)

See `prefill-breakdown.md`. Prefill 763.7 ms. Host RoPE, host attention, attention copies and KV
mirror copies all read 0; attention runs as `region_attention_compute` (73.7 ms, 9.7%).

The breakdown script exits 1: residue 10.9% against its 5% flag. The residue is not new time: in
absolute terms it is 83.6 ms, against 88.7 ms (2.9% of 3,012 ms) at `20261002T050741Z`. Its share
grew because the window around it became about four times faster (packed matmul since then, and this
change). It is time inside the forward pass outside every `juno.WindowStep` span; it is reported here,
the flag was not relaxed.

## Throughput (unpinned, a reading only)

| Model | Juno pp t/s | Juno tg t/s | Juno/ref pp | Juno/ref tg |
|---|---|---|---|---|
| Phi-3.5-mini | 670.39 | 29.43 | 0.574x | 0.515x |

Unpinned and taken with `--device-spans` recording on; not comparable with the pinned sweeps the
program targets are scored on.

## How to read it

Every file is one repetition's harness output (`*-prefill-rep*`, `*-generate-rep*`) or its aggregate.
The bytes come from `models[0].metrics["juno.DeviceStaging.{H2D,D2H}.prefill.bytes"]` in
`*-prefill-rep*-juno-jfr.json`, per-site figures from the `juno.DeviceStaging.site.*.prefill.bytes`
keys of the same files.
