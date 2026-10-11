# CPU decode kernel variants (threads=12, reps=5, target 1000 ms per rep)

| Shape | Variant | Median ms/call | Min to max | Weight GB/s | Time vs a-single | Max rel. diff vs a-single |
|---|---|---|---|---|---|---|
| Q4_K 14336x4096 | production | 13.686 | 13.565 to 14.114 | 2.41 | 0.995 | 0.00e+00 |
| Q4_K 14336x4096 | a-single | 13.748 | 13.457 to 13.894 | 2.40 | 1.000 | 0.00e+00 |
| Q4_K 14336x4096 | b-split4 | 13.632 | 13.543 to 13.804 | 2.42 | 0.992 | 2.05e-06 |
| Q4_K 14336x4096 | b-split8 | 15.744 | 15.290 to 15.856 | 2.10 | 1.145 | 2.12e-06 |
| Q6_K 4096x14336 | production | 23.394 | 23.200 to 24.038 | 2.06 | 1.016 | 0.00e+00 |
| Q6_K 4096x14336 | a-single | 23.016 | 22.812 to 23.572 | 2.09 | 1.000 | 0.00e+00 |
| Q6_K 4096x14336 | b-split4 | 23.417 | 23.407 to 23.957 | 2.06 | 1.017 | 4.15e-06 |
| Q6_K 4096x14336 | b-split8 | 19.310 | 19.056 to 19.800 | 2.49 | 0.839 | 3.67e-06 |
