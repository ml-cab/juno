# CPU decode kernel variants (threads=12, reps=5, target 1000 ms per rep)

| Shape | Variant | Median ms/call | Min to max | Weight GB/s | Time vs a-single | Max rel. diff vs a-single |
|---|---|---|---|---|---|---|
| Q4_K 14336x4096 | production | 13.583 | 13.456 to 14.031 | 2.43 | 0.998 | 0.00e+00 |
| Q4_K 14336x4096 | a-single | 13.610 | 13.459 to 13.856 | 2.43 | 1.000 | 0.00e+00 |
| Q4_K 14336x4096 | b-split4 | 13.604 | 13.342 to 14.289 | 2.43 | 1.000 | 2.05e-06 |
| Q4_K 14336x4096 | b-split8 | 15.099 | 15.038 to 15.366 | 2.19 | 1.109 | 2.12e-06 |
| Q6_K 4096x14336 | production | 25.272 | 25.076 to 25.811 | 1.91 | 1.009 | 0.00e+00 |
| Q6_K 4096x14336 | a-single | 25.035 | 24.881 to 27.302 | 1.92 | 1.000 | 0.00e+00 |
| Q6_K 4096x14336 | b-split4 | 23.687 | 23.345 to 24.106 | 2.03 | 0.946 | 4.15e-06 |
| Q6_K 4096x14336 | b-split8 | 19.576 | 18.932 to 19.922 | 2.46 | 0.782 | 3.67e-06 |
