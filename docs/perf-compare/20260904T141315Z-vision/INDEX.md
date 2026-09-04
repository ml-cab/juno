# Vision perf compare — 20260904T141315Z

Model: `moondream2-q5_k.llamafile` · backend=gpu · prompt: *What is in this image?* · max_tokens=32 · JFR `30m`

| ref | commit | prompt tok | latency ms | prefill ms | decode tps | status |
|-----|--------|----------:|-----------:|-----------:|-----------:|:------:|
| 47-vision | a5255d3 | 741 | 502863 | 0 | 1.4127585116052728 | success |
| 47-vision | a5255d3 | 741 | 502863 | 0 | 1.4127585116052728 | success |


Decode tps prefers `juno.TokenProduced.tps` from JFR when present; latency from `x_juno_latency_ms`.
Prefill: `juno.ForwardPass.prefill.total_ms` (vision+text).
