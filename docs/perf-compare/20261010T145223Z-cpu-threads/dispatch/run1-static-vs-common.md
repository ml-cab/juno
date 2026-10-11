# CPU dispatch: common pool against worker pool (worker threads=12, common pool parallelism=11, reps=5)

| Case | Common pool ms | Worker pool ms | Worker / common | Common min to max | Worker min to max |
|---|---|---|---|---|---|
| decode 12 256x2048 | 0.082 | 0.083 | 1.023 | 0.081 to 0.082 | 0.072 to 0.092 |
| decode 12 2048x2048 | 0.354 | 0.456 | 1.288 | 0.345 to 0.392 | 0.405 to 0.484 |
| decode 14 2048x5632 | 3.008 | 3.518 | 1.170 | 2.956 to 3.066 | 3.082 to 4.032 |
| decode 12 1024x4096 | 0.357 | 0.454 | 1.270 | 0.349 to 0.406 | 0.445 to 0.470 |
| decode 12 14336x4096 | 4.397 | 4.890 | 1.112 | 4.218 to 4.435 | 4.743 to 5.347 |
| decode 14 4096x14336 | 14.713 | 16.444 | 1.118 | 14.457 to 15.631 | 15.305 to 20.019 |
| decode layer, TinyLlama shapes (7 matmuls) | 5.796 | 6.917 | 1.193 | 5.763 to 6.123 | 6.136 to 7.744 |
| decode layer, Mistral 7B shapes (7 matmuls) | 27.443 | 31.164 | 1.136 | 27.021 to 28.980 | 27.978 to 34.608 |
| prefill B=741, Q4_K 2048x2048 | 98.186 | 99.657 | 1.015 | 96.700 to 98.569 | 96.501 to 107.554 |
| prefill B=32, Q4_K 4096x4096 | 18.477 | 19.092 | 1.033 | 17.680 to 18.493 | 18.163 to 22.078 |
