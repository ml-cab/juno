# CPU decode kernel variants (threads=12, reps=5, target 1000 ms per rep)

| Shape | Variant | Median ms/call | Min to max | Weight GB/s | Time vs a-single | Max rel. diff vs a-single |
|---|---|---|---|---|---|---|
| Q4_K 14336x4096 | production | 13.913 | 13.629 to 14.250 | 2.37 | 1.024 | 0.00e+00 |
| Q4_K 14336x4096 | a-single | 13.590 | 13.480 to 14.098 | 2.43 | 1.000 | 0.00e+00 |
| Q4_K 14336x4096 | b-split4 | 13.627 | 13.510 to 14.110 | 2.42 | 1.003 | 2.05e-06 |
| Q4_K 14336x4096 | b-split8 | 15.478 | 15.161 to 16.970 | 2.13 | 1.139 | 2.12e-06 |
| Q6_K 4096x14336 | production | 25.207 | 24.662 to 25.808 | 1.91 | 1.073 | 0.00e+00 |
| Q6_K 4096x14336 | a-single | 23.483 | 22.973 to 23.972 | 2.05 | 1.000 | 0.00e+00 |
| Q6_K 4096x14336 | b-split4 | 23.699 | 23.347 to 24.131 | 2.03 | 1.009 | 4.15e-06 |
| Q6_K 4096x14336 | b-split8 | 19.651 | 18.988 to 19.934 | 2.45 | 0.837 | 3.67e-06 |
