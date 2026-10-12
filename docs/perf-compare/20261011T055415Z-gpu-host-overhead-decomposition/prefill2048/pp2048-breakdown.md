| Model | Prefill ms | Reps | Term | ms | Share |
|---|---|---|---|---|---|
| Phi-3.5-mini-instruct-Q4_K_M | 3628.2 | 3 | gemm_compute | 1376.5 | 37.9% |
| Phi-3.5-mini-instruct-Q4_K_M | 3628.2 | 3 | weight_dequant | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3628.2 | 3 | matmul_staging | 164.1 | 4.5% |
| Phi-3.5-mini-instruct-Q4_K_M | 3628.2 | 3 | host_fp16_pack | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3628.2 | 3 | projection_and_region_host | 263.7 | 7.3% |
| Phi-3.5-mini-instruct-Q4_K_M | 3628.2 | 3 | device_elementwise | 237.9 | 6.6% |
| Phi-3.5-mini-instruct-Q4_K_M | 3628.2 | 3 | region_attention_compute | 1099.2 | 30.3% |
| Phi-3.5-mini-instruct-Q4_K_M | 3628.2 | 3 | attention_compute | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3628.2 | 3 | attention_copies | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3628.2 | 3 | attention_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3628.2 | 3 | kv_mirror_copies | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3628.2 | 3 | kv_host_write | 249.8 | 6.9% |
| Phi-3.5-mini-instruct-Q4_K_M | 3628.2 | 3 | swiglu_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3628.2 | 3 | rmsnorm_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3628.2 | 3 | rope_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3628.2 | 3 | residual_add_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3628.2 | 3 | bias_add_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3628.2 | 3 | embed | 4.6 | 0.1% |
| Phi-3.5-mini-instruct-Q4_K_M | 3628.2 | 3 | lm_head | 1.7 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3628.2 | 3 | residue | 226.4 | 6.2% |
