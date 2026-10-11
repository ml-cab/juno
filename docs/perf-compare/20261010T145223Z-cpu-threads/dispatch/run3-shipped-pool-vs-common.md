# CPU dispatch: common pool against worker pool (worker threads=12, common pool parallelism=11, reps=5)

| Case | Common pool ms | Worker pool ms | Worker / common | Common min to max | Worker min to max |
|---|---|---|---|---|---|
| decode 12 256x2048 | 0.081 | 0.113 | 1.393 | 0.079 to 0.081 | 0.094 to 0.124 |
| decode 12 2048x2048 | 0.359 | 0.370 | 1.031 | 0.350 to 0.384 | 0.352 to 0.387 |
| decode 14 2048x5632 | 3.058 | 2.962 | 0.969 | 2.970 to 3.061 | 2.856 to 3.247 |
| decode 12 1024x4096 | 0.359 | 0.380 | 1.057 | 0.354 to 0.372 | 0.351 to 0.432 |
| decode 12 14336x4096 | 4.332 | 4.279 | 0.988 | 4.315 to 4.512 | 4.237 to 4.383 |
| decode 14 4096x14336 | 14.743 | 14.503 | 0.984 | 14.506 to 15.294 | 14.154 to 14.937 |
| decode layer, TinyLlama shapes (7 matmuls) | 6.034 | 5.875 | 0.974 | 5.703 to 6.244 | 5.758 to 6.382 |
| decode layer, Mistral 7B shapes (7 matmuls) | 27.449 | 27.573 | 1.004 | 26.656 to 28.422 | 26.428 to 29.853 |
| prefill B=741, Q4_K 2048x2048 | 96.674 | 96.666 | 1.000 | 94.595 to 99.983 | 95.091 to 106.465 |
| prefill B=32, Q4_K 4096x4096 | 18.119 | 17.977 | 0.992 | 17.463 to 19.106 | 17.625 to 18.620 |
