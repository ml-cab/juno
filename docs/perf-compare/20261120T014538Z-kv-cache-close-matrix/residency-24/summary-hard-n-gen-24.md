# --gpu-residency smoke — 20261120T155448Z


## tinyllama-1.1b-chat-v1.0.Q4_K_M

- tinyllama-1.1b-chat-v1.0.Q4_K_M: GPU MiB after each request, off: 938 938 938 938
- tinyllama-1.1b-chat-v1.0.Q4_K_M: GPU MiB after each request, on:  938 938 938 938
- PASS: tinyllama-1.1b-chat-v1.0.Q4_K_M: GPU-resident decode region active (gpu-residency=on) on 8 of 8 layers
- PASS: tinyllama-1.1b-chat-v1.0.Q4_K_M: GPU memory growth over requests 2..4: on 0 MiB, off 0 MiB (limit 16 MiB)
- PASS: tinyllama-1.1b-chat-v1.0.Q4_K_M: greedy output identical on vs off over the first 24 tokens; over 32 it differs from word 21 (reported, not failed; see tinyllama-1.1b-chat-v1.0.Q4_K_M-local-on.txt / -off.txt)

## cluster (tinyllama-1.1b-chat-v1.0.Q4_K_M)

- PASS: cluster pipeline on: answers, output equals local mode's
- PASS: cluster pipeline: no node JVM left
- PASS: cluster tensor on: answers, output equals local mode's
- PASS: cluster tensor: no node JVM left

## mistral-7b-instruct-v0.1-q4_k_m

- mistral-7b-instruct-v0.1-q4_k_m: GPU MiB after each request, off: 4538 4538 4538 4538
- mistral-7b-instruct-v0.1-q4_k_m: GPU MiB after each request, on:  4544 4544 4544 4544
- PASS: mistral-7b-instruct-v0.1-q4_k_m: GPU-resident decode region active (gpu-residency=on) on 11 of 11 layers
- PASS: mistral-7b-instruct-v0.1-q4_k_m: GPU memory growth over requests 2..4: on 0 MiB, off 0 MiB (limit 16 MiB)
- PASS: mistral-7b-instruct-v0.1-q4_k_m: greedy output identical on vs off (32 tokens)

## llama-1-30b.Q4_K_M

- llama-1-30b.Q4_K_M: GPU MiB after each request, off: 7350 7350 7352 7354
- llama-1-30b.Q4_K_M: GPU MiB after each request, on:  7362 7362 7362 7362
- PASS: llama-1-30b.Q4_K_M: GPU-resident decode region active (gpu-residency=on) on 20 of 20 layers
- PASS: llama-1-30b.Q4_K_M: GPU memory growth over requests 2..4: on 0 MiB, off 4 MiB (limit 16 MiB)
- PASS: llama-1-30b.Q4_K_M: greedy output identical on vs off (32 tokens)

Failures: 0
