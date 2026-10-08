# --gpu-residency smoke — 20261008T042110Z


## tinyllama

- tinyllama: GPU MiB after each request, off: 2048 2048 2048 2048
- tinyllama: GPU MiB after each request, on:  2048 2048 2048 2048
- tinyllama: declined - requested, but this shard has no layer whose Q/K/V projections are all K-quant MMQ matrices on the device; using the existing path there
- PASS: tinyllama: GPU memory growth over requests 2..4: on 0 MiB, off 0 MiB (limit 16 MiB)
- PASS: tinyllama: greedy output identical on vs off (32 tokens)

## cluster (tinyllama)

- PASS: cluster pipeline on: answers, output equals local mode's
- PASS: cluster pipeline: no node JVM left
- PASS: cluster tensor on: answers, output equals local mode's
- PASS: cluster tensor: no node JVM left

## mistral-7b

- mistral-7b: GPU MiB after each request, off: 4538 4538 4538 4538
- mistral-7b: GPU MiB after each request, on:  4544 4544 4544 4544
- PASS: mistral-7b: GPU-resident decode region active (gpu-residency=on) on 11 of 11 layers
- PASS: mistral-7b: GPU memory growth over requests 2..4: on 0 MiB, off 0 MiB (limit 16 MiB)
- PASS: mistral-7b: greedy output identical on vs off (32 tokens)

## Phi-3.5-mini

- Phi-3.5-mini: GPU MiB after each request, off: 2664 2664 2664 2664
- Phi-3.5-mini: GPU MiB after each request, on:  2672 2672 2672 2672
- PASS: Phi-3.5-mini: GPU-resident decode region active (gpu-residency=on) on 11 of 11 layers
- PASS: Phi-3.5-mini: GPU memory growth over requests 2..4: on 0 MiB, off 0 MiB (limit 16 MiB)
- PASS: Phi-3.5-mini: greedy output identical on vs off (32 tokens)

## Qwen3-1.7B

- Qwen3-1.7B: GPU MiB after each request, off: 1788 1788 1788 1788
- Qwen3-1.7B: GPU MiB after each request, on:  1788 1788 1788 1788
- PASS: Qwen3-1.7B: GPU-resident decode region active (gpu-residency=on) on 10 of 10 layers
- PASS: Qwen3-1.7B: GPU memory growth over requests 2..4: on 0 MiB, off 0 MiB (limit 16 MiB)
- PASS: Qwen3-1.7B: greedy output identical on vs off (32 tokens)

## qwen2.5-3b

- qwen2.5-3b: GPU MiB after each request, off: 2346 2346 2346 2346
- qwen2.5-3b: GPU MiB after each request, on:  2346 2346 2346 2346
- qwen2.5-3b: declined - requested, but architecture qwen2 uses the split-half RoPE layout, which the device RoPE kernel does not implement; using the existing path there
- PASS: qwen2.5-3b: GPU memory growth over requests 2..4: on 0 MiB, off 0 MiB (limit 16 MiB)
- PASS: qwen2.5-3b: greedy output identical on vs off (32 tokens)

Failures: 0
