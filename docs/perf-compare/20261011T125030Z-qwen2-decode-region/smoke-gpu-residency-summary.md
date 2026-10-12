# --gpu-residency smoke — 20261011T124827Z


## tinyllama-1.1b-chat-v1.0.Q4_K_M

- tinyllama-1.1b-chat-v1.0.Q4_K_M: GPU MiB after each request, off: 938 938 938 938
- tinyllama-1.1b-chat-v1.0.Q4_K_M: GPU MiB after each request, on:  938 938 938 938
- PASS: tinyllama-1.1b-chat-v1.0.Q4_K_M: GPU-resident decode region active (gpu-residency=on) on 8 of 8 layers
- PASS: tinyllama-1.1b-chat-v1.0.Q4_K_M: GPU memory growth over requests 2..4: on 0 MiB, off 0 MiB (limit 16 MiB)
- PASS: tinyllama-1.1b-chat-v1.0.Q4_K_M: greedy output identical on vs off (32 tokens)

## cluster (tinyllama-1.1b-chat-v1.0.Q4_K_M)

- PASS: cluster pipeline on: answers, output equals local mode's
- PASS: cluster pipeline: no node JVM left
- PASS: cluster tensor on: answers, output equals local mode's
- PASS: cluster tensor: no node JVM left

## qwen2.5-3b-instruct-q4_k_m

- qwen2.5-3b-instruct-q4_k_m: GPU MiB after each request, off: 2346 2346 2346 2346
- qwen2.5-3b-instruct-q4_k_m: GPU MiB after each request, on:  2346 2346 2346 2346
- PASS: qwen2.5-3b-instruct-q4_k_m: GPU-resident decode region active (gpu-residency=on) on 12 of 12 layers
- PASS: qwen2.5-3b-instruct-q4_k_m: GPU memory growth over requests 2..4: on 0 MiB, off 0 MiB (limit 16 MiB)
- PASS: qwen2.5-3b-instruct-q4_k_m: greedy output identical on vs off (32 tokens)

Failures: 0
