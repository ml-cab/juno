# CPU decode kernel variants (threads=1; printed as threads=2 by the label bug fixed before run 3, reps=5, target 1000 ms per rep)

| Shape | Variant | Median ms/call | Min to max | Weight GB/s | Time vs a-single | Max rel. diff vs a-single |
|---|---|---|---|---|---|---|
| Q4_K 14336x4096 | production | 80.530 | 79.719 to 83.703 | 0.41 | 1.000 | 0.00e+00 |
| Q4_K 14336x4096 | a-single | 80.545 | 79.859 to 81.499 | 0.41 | 1.000 | 0.00e+00 |
| Q4_K 14336x4096 | b-split4 | 82.065 | 81.154 to 83.113 | 0.40 | 1.019 | 2.05e-06 |
| Q4_K 14336x4096 | b-split8 | 82.747 | 81.196 to 84.684 | 0.40 | 1.027 | 2.12e-06 |
| Q6_K 4096x14336 | production | 128.179 | 124.762 to 129.390 | 0.38 | 1.017 | 0.00e+00 |
| Q6_K 4096x14336 | a-single | 126.035 | 125.058 to 127.204 | 0.38 | 1.000 | 0.00e+00 |
| Q6_K 4096x14336 | b-split4 | 127.280 | 126.032 to 135.193 | 0.38 | 1.010 | 4.15e-06 |
| Q6_K 4096x14336 | b-split8 | 102.937 | 102.392 to 106.379 | 0.47 | 0.817 | 3.67e-06 |
