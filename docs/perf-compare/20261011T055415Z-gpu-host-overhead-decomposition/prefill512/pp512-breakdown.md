| Model | Prefill ms | Reps | Term | ms | Share |
|---|---|---|---|---|---|
| Phi-3.5-mini-instruct-Q4_K_M | 767.4 | 3 | gemm_compute | 333.6 | 43.5% |
| Phi-3.5-mini-instruct-Q4_K_M | 767.4 | 3 | weight_dequant | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 767.4 | 3 | matmul_staging | 42.3 | 5.5% |
| Phi-3.5-mini-instruct-Q4_K_M | 767.4 | 3 | host_fp16_pack | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 767.4 | 3 | projection_and_region_host | 84.4 | 11% |
| Phi-3.5-mini-instruct-Q4_K_M | 767.4 | 3 | device_elementwise | 70.1 | 9.1% |
| Phi-3.5-mini-instruct-Q4_K_M | 767.4 | 3 | region_attention_compute | 76.5 | 10% |
| Phi-3.5-mini-instruct-Q4_K_M | 767.4 | 3 | attention_compute | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 767.4 | 3 | attention_copies | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 767.4 | 3 | attention_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 767.4 | 3 | kv_mirror_copies | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 767.4 | 3 | kv_host_write | 69.3 | 9% |
| Phi-3.5-mini-instruct-Q4_K_M | 767.4 | 3 | swiglu_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 767.4 | 3 | rmsnorm_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 767.4 | 3 | rope_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 767.4 | 3 | residual_add_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 767.4 | 3 | bias_add_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 767.4 | 3 | embed | 1 | 0.1% |
| Phi-3.5-mini-instruct-Q4_K_M | 767.4 | 3 | lm_head | 1.7 | 0.2% |
| Phi-3.5-mini-instruct-Q4_K_M | 767.4 | 3 | residue | 84.3 | 11% |
