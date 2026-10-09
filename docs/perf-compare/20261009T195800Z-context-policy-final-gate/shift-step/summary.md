# Context-shift step latency

- Jar: `candidate-shaded.jar` sha256 `16f8794d833b1dff`; commit 0f87603 (dirty tree)
- Clocks: cpu governor performance (was schedutil), turbo off, gpu clock recorded, not fixed
- Depth 32768, keep 32, 3 repetitions, heap 16g; gate: median ratio <= 3.0

| Model | Backend | Median ratio | Min | Max | Result | Log |
|---|---|---|---|---|---|---|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | cpu | 0.541 | 0.539 | 0.542 | met | `tinyllama-1.1b-chat-v1.0.Q4_K_M-cpu.log` |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | gpu | 2.362 | 2.361 | 2.620 | met | `tinyllama-1.1b-chat-v1.0.Q4_K_M-gpu.log` |
| mistral-7b-instruct-v0.1-q4_k_m | cpu | 0.581 | 0.580 | 0.586 | met | `mistral-7b-instruct-v0.1-q4_k_m-cpu.log` |
| mistral-7b-instruct-v0.1-q4_k_m | gpu | 0.630 | 0.630 | 0.631 | met | `mistral-7b-instruct-v0.1-q4_k_m-gpu.log` |
