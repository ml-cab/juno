# --gpu-residency smoke — 20261120T161026Z


## tinyllama-1.1b-chat-v1.0.Q4_K_M

- tinyllama-1.1b-chat-v1.0.Q4_K_M: GPU MiB after each request, off: 938 938 938 938
- tinyllama-1.1b-chat-v1.0.Q4_K_M: GPU MiB after each request, on:  938 938 938 938
- PASS: tinyllama-1.1b-chat-v1.0.Q4_K_M: GPU-resident decode region active (gpu-residency=on) on 8 of 8 layers
- PASS: tinyllama-1.1b-chat-v1.0.Q4_K_M: GPU memory growth over requests 2..4: on 0 MiB, off 0 MiB (limit 16 MiB)
- FAIL: tinyllama-1.1b-chat-v1.0.Q4_K_M: greedy output differs on vs off within the first 32 tokens, from word 21

Failures: 1
