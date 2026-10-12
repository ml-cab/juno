| Model | Prefill ms | Reps | Term | ms | Share |
|---|---|---|---|---|---|
| Phi-3.5-mini-instruct-Q4_K_M | 3096.5 | 3 | gemm_compute | 1324.7 | 42.8% |
| Phi-3.5-mini-instruct-Q4_K_M | 3096.5 | 3 | weight_dequant | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3096.5 | 3 | matmul_staging | 164.3 | 5.3% |
| Phi-3.5-mini-instruct-Q4_K_M | 3096.5 | 3 | host_fp16_pack | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3096.5 | 3 | projection_and_region_host | 260.6 | 8.4% |
| Phi-3.5-mini-instruct-Q4_K_M | 3096.5 | 3 | device_elementwise | 234.5 | 7.6% |
| Phi-3.5-mini-instruct-Q4_K_M | 3096.5 | 3 | region_attention_compute | 1052.2 | 34% |
| Phi-3.5-mini-instruct-Q4_K_M | 3096.5 | 3 | attention_compute | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3096.5 | 3 | attention_copies | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3096.5 | 3 | attention_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3096.5 | 3 | kv_mirror_copies | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3096.5 | 3 | kv_host_write | 0.1 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3096.5 | 3 | swiglu_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3096.5 | 3 | rmsnorm_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3096.5 | 3 | rope_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3096.5 | 3 | residual_add_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3096.5 | 3 | bias_add_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3096.5 | 3 | embed | 4.7 | 0.2% |
| Phi-3.5-mini-instruct-Q4_K_M | 3096.5 | 3 | lm_head | 1.6 | 0.1% |
| Phi-3.5-mini-instruct-Q4_K_M | 3096.5 | 3 | residue | 49.6 | 1.6% |
