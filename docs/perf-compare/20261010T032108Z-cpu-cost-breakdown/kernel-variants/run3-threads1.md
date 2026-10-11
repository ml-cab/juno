# CPU decode kernel variants (threads=1, reps=5, target 1000 ms per rep)

| Shape | Variant | Median ms/call | Min to max | Weight GB/s | Time vs a-single | Max rel. diff vs a-single |
|---|---|---|---|---|---|---|
| Q4_K 14336x4096 | production | 79.755 | 79.283 to 83.596 | 0.41 | 1.002 | 0.00e+00 |
| Q4_K 14336x4096 | a-single | 79.631 | 79.435 to 81.577 | 0.41 | 1.000 | 0.00e+00 |
| Q4_K 14336x4096 | b-split4 | 81.882 | 80.472 to 82.981 | 0.40 | 1.028 | 2.05e-06 |
| Q4_K 14336x4096 | b-split8 | 82.218 | 81.626 to 83.686 | 0.40 | 1.032 | 2.12e-06 |
| Q6_K 4096x14336 | production | 133.649 | 129.031 to 141.562 | 0.36 | 1.004 | 0.00e+00 |
| Q6_K 4096x14336 | a-single | 133.125 | 130.512 to 139.840 | 0.36 | 1.000 | 0.00e+00 |
| Q6_K 4096x14336 | b-split4 | 131.757 | 130.543 to 139.608 | 0.37 | 0.990 | 4.15e-06 |
| Q6_K 4096x14336 | b-split8 | 109.597 | 105.550 to 117.317 | 0.44 | 0.823 | 3.67e-06 |
