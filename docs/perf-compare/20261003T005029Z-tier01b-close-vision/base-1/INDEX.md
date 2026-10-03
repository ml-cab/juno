# Vision perf compare — 20261003T010002Z

Model: `moondream2-q5_k.llamafile` · backend=gpu · prefill=`single` · prompt: *What is in this image?* · max_tokens=32 · JFR `30m`

| ref | commit | prompt tok | latency ms | prefill ms | decode tps | status |
|-----|--------|----------:|-----------:|-----------:|-----------:|:------:|
| HEAD | unknown | 741 | 479501 | 853.045674 | 1.4669575210250527 | success |


Decode tps prefers `juno.TokenProduced.tps` from JFR when present; latency from `x_juno_latency_ms`.
Prefill: `juno.ForwardPass.prefill.total_ms` (vision+text). Default launch uses `--prefill single` for caption quality.
