| Model | Prefill ms | Reps | Term | ms | Share |
|---|---|---|---|---|---|
| Phi-3.5-mini-instruct-Q4_K_M | 602.2 | 3 | gemm_compute | 324.8 | 53.9% |
| Phi-3.5-mini-instruct-Q4_K_M | 602.2 | 3 | weight_dequant | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 602.2 | 3 | matmul_staging | 44 | 7.3% |
| Phi-3.5-mini-instruct-Q4_K_M | 602.2 | 3 | host_fp16_pack | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 602.2 | 3 | projection_and_region_host | 72.8 | 12.1% |
| Phi-3.5-mini-instruct-Q4_K_M | 602.2 | 3 | device_elementwise | 69.2 | 11.5% |
| Phi-3.5-mini-instruct-Q4_K_M | 602.2 | 3 | region_attention_compute | 73.7 | 12.2% |
| Phi-3.5-mini-instruct-Q4_K_M | 602.2 | 3 | attention_compute | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 602.2 | 3 | attention_copies | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 602.2 | 3 | attention_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 602.2 | 3 | kv_mirror_copies | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 602.2 | 3 | kv_host_write | 0.1 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 602.2 | 3 | swiglu_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 602.2 | 3 | rmsnorm_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 602.2 | 3 | rope_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 602.2 | 3 | residual_add_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 602.2 | 3 | bias_add_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 602.2 | 3 | embed | 1 | 0.2% |
| Phi-3.5-mini-instruct-Q4_K_M | 602.2 | 3 | lm_head | 1.7 | 0.3% |
| Phi-3.5-mini-instruct-Q4_K_M | 602.2 | 3 | residue | 19 | 3.2% |
