| Model | Prefill ms | Reps | Term | ms | Share |
|---|---|---|---|---|---|
| Phi-3.5-mini-instruct-Q4_K_M | 3348.5 | 3 | gemm_compute | 1318.1 | 39.4% |
| Phi-3.5-mini-instruct-Q4_K_M | 3348.5 | 3 | weight_dequant | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3348.5 | 3 | matmul_staging | 163.7 | 4.9% |
| Phi-3.5-mini-instruct-Q4_K_M | 3348.5 | 3 | host_fp16_pack | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3348.5 | 3 | projection_and_region_host | 270.3 | 8.1% |
| Phi-3.5-mini-instruct-Q4_K_M | 3348.5 | 3 | device_elementwise | 230.4 | 6.9% |
| Phi-3.5-mini-instruct-Q4_K_M | 3348.5 | 3 | region_attention_compute | 1051.6 | 31.4% |
| Phi-3.5-mini-instruct-Q4_K_M | 3348.5 | 3 | attention_compute | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3348.5 | 3 | attention_copies | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3348.5 | 3 | attention_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3348.5 | 3 | kv_mirror_copies | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3348.5 | 3 | kv_host_write | 260.9 | 7.8% |
| Phi-3.5-mini-instruct-Q4_K_M | 3348.5 | 3 | swiglu_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3348.5 | 3 | rmsnorm_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3348.5 | 3 | rope_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3348.5 | 3 | residual_add_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3348.5 | 3 | bias_add_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3348.5 | 3 | embed | 4.6 | 0.1% |
| Phi-3.5-mini-instruct-Q4_K_M | 3348.5 | 3 | lm_head | 1.6 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 3348.5 | 3 | residue | 50.8 | 1.5% |
