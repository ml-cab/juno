# Vision perf compare — 20261003T011854Z

Model: `moondream2-q5_k.llamafile` · backend=gpu · prefill=`single` · prompt: *What is in this image?* · max_tokens=32 · JFR `30m`

| ref | commit | prompt tok | latency ms | prefill ms | decode tps | status |
|-----|--------|----------:|-----------:|-----------:|-----------:|:------:|
| HEAD | unknown | 741 | 475779 | 820.145925 | 1.4413042745911844 | success |


Decode tps prefers `juno.TokenProduced.tps` from JFR when present; latency from `x_juno_latency_ms`.
Prefill: `juno.ForwardPass.prefill.total_ms` (vision+text). Default launch uses `--prefill single` for caption quality.
