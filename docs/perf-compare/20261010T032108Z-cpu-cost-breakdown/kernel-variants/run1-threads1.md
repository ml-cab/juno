# CPU decode kernel variants (threads=1; printed as threads=2 by the label bug fixed before run 3, reps=5, target 1000 ms per rep)

| Shape | Variant | Median ms/call | Min to max | Weight GB/s | Time vs a-single | Max rel. diff vs a-single |
|---|---|---|---|---|---|---|
| Q4_K 14336x4096 | production | 80.086 | 79.885 to 80.684 | 0.41 | 0.997 | 0.00e+00 |
| Q4_K 14336x4096 | a-single | 80.334 | 80.166 to 82.048 | 0.41 | 1.000 | 0.00e+00 |
| Q4_K 14336x4096 | b-split4 | 82.360 | 81.600 to 83.958 | 0.40 | 1.025 | 2.05e-06 |
| Q4_K 14336x4096 | b-split8 | 83.068 | 81.909 to 84.101 | 0.40 | 1.034 | 2.12e-06 |
| Q6_K 4096x14336 | production | 127.051 | 125.086 to 129.934 | 0.38 | 0.992 | 0.00e+00 |
| Q6_K 4096x14336 | a-single | 128.082 | 126.548 to 128.965 | 0.38 | 1.000 | 0.00e+00 |
| Q6_K 4096x14336 | b-split4 | 128.021 | 126.917 to 128.371 | 0.38 | 1.000 | 4.15e-06 |
| Q6_K 4096x14336 | b-split8 | 103.985 | 103.120 to 106.382 | 0.46 | 0.812 | 3.67e-06 |
