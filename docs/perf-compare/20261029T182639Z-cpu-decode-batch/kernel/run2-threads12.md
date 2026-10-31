# CPU decode-batch matmuls (threads=12, integer width 128 bits, reps=3, target 600 ms per rep)

| Shape | Batch | Variant | Median ms/call | Min to max | Time vs gemm | Rows bit-identical to single |
|---|---|---|---|---|---|---|
| Q6_K 2048x5632 | 4 | gemm | 3.336 | 3.284 to 3.992 | 1.000 | no |
| Q6_K 2048x5632 | 4 | decode-batch | 5.057 | 5.033 to 5.537 | 1.516 | yes |
| Q6_K 2048x5632 | 4 | single-calls | 5.579 | 5.198 to 5.644 | 1.672 | yes |
| Q6_K 2048x5632 | 16 | gemm | 8.014 | 7.340 to 8.207 | 1.000 | no |
| Q6_K 2048x5632 | 16 | decode-batch | 19.398 | 18.888 to 20.915 | 2.421 | yes |
| Q6_K 2048x5632 | 16 | single-calls | 21.336 | 19.293 to 22.017 | 2.662 | yes |
| Q6_K 2048x5632 | 64 | gemm | 25.068 | 24.978 to 25.180 | 1.000 | no |
| Q6_K 2048x5632 | 64 | decode-batch | 75.248 | 74.644 to 75.304 | 3.002 | yes |
| Q6_K 2048x5632 | 64 | single-calls | 79.026 | 77.049 to 79.134 | 3.152 | yes |
| Q4_K 2048x5632 | 4 | gemm | 2.130 | 2.100 to 2.180 | 1.000 | no |
| Q4_K 2048x5632 | 4 | decode-batch | 2.585 | 2.463 to 2.606 | 1.214 | yes |
| Q4_K 2048x5632 | 4 | single-calls | 2.555 | 2.546 to 2.592 | 1.199 | yes |
| Q4_K 2048x5632 | 16 | gemm | 5.005 | 4.975 to 5.069 | 1.000 | no |
| Q4_K 2048x5632 | 16 | decode-batch | 9.693 | 9.690 to 9.957 | 1.937 | yes |
| Q4_K 2048x5632 | 16 | single-calls | 10.425 | 10.152 to 10.570 | 2.083 | yes |
| Q4_K 2048x5632 | 64 | gemm | 16.453 | 16.372 to 17.090 | 1.000 | no |
| Q4_K 2048x5632 | 64 | decode-batch | 38.510 | 38.483 to 38.777 | 2.341 | yes |
| Q4_K 2048x5632 | 64 | single-calls | 41.949 | 40.586 to 42.042 | 2.550 | yes |
