# CPU dispatch: common pool against worker pool (worker threads=12, common pool parallelism=11, reps=5)

| Case | Common pool ms | Worker pool, static ms | Static / common | Worker pool, chunked ms | Chunked / common |
|---|---|---|---|---|---|
| decode 12 256x2048 | 0.084 | 0.096 | 1.141 | 0.112 | 1.329 |
| decode 12 2048x2048 | 0.353 | 0.432 | 1.225 | 0.364 | 1.032 |
| decode 14 2048x5632 | 3.053 | 3.354 | 1.098 | 2.939 | 0.962 |
| decode 12 1024x4096 | 0.366 | 0.453 | 1.239 | 0.372 | 1.017 |
| decode 12 14336x4096 | 4.354 | 4.846 | 1.113 | 4.277 | 0.982 |
| decode 14 4096x14336 | 14.711 | 16.143 | 1.097 | 14.450 | 0.982 |
| decode layer, TinyLlama shapes (7 matmuls) | 5.848 | 6.701 | 1.146 | 5.891 | 1.007 |
| decode layer, Mistral 7B shapes (7 matmuls) | 27.660 | 30.546 | 1.104 | 27.127 | 0.981 |
| prefill B=741, Q4_K 2048x2048 | 95.807 | 100.868 | 1.053 | 96.476 | 1.007 |
| prefill B=32, Q4_K 4096x4096 | 18.104 | 18.907 | 1.044 | 18.127 | 1.001 |
[cpu-dispatch] done: dispatch2
