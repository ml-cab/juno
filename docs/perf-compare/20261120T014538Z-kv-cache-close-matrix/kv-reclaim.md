# KV page reclaim pause

Budget: a reclaim pass adds at most 5.0 ms to any decode step it overlaps (added_ms: the longest overlapping step less the median step alone).

| model | layers | ctx | batch | page | kv_dim | idle_pages | pass_ms median | pass_ms max | hold_us | step_ms p50 | step_ms max | overlapping steps | step_ms p50, overlapping | step_ms max, overlapping | added_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tinyllama | 22 | 8192 | 8 | 16 | 16 | 180224 | 2.004 | 13.131 | 2.8 | 0.046 | 0.363 | 56 | 0.134 | 2.373 | 2.328 |
| qwen2.5-3b | 36 | 8192 | 8 | 16 | 16 | 294912 | 3.002 | 3.933 | 2.6 | 0.069 | 1.615 | 56 | 0.280 | 0.821 | 0.752 |
| phi-3.5-mini | 32 | 8192 | 8 | 16 | 16 | 262144 | 4.371 | 4.500 | 4.3 | 0.060 | 0.601 | 60 | 0.377 | 0.937 | 0.877 |
| mistral-7b | 32 | 8192 | 8 | 16 | 16 | 262144 | 4.255 | 4.630 | 4.2 | 0.044 | 1.423 | 82 | 0.290 | 0.494 | 0.449 |
