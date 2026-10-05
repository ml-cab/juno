| Model | Prefill ms | Reps | Term | ms | Share |
|---|---|---|---|---|---|
| Phi-3.5-mini-instruct-Q4_K_M | 24333.7 | 3 | gemm_compute | 1448.7 | 6% |
| Phi-3.5-mini-instruct-Q4_K_M | 24333.7 | 3 | weight_dequant | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 24333.7 | 3 | matmul_staging | 316 | 1.3% |
| Phi-3.5-mini-instruct-Q4_K_M | 24333.7 | 3 | host_fp16_pack | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 24333.7 | 3 | projection_and_region_host | 888.2 | 3.7% |
| Phi-3.5-mini-instruct-Q4_K_M | 24333.7 | 3 | device_elementwise | 167.9 | 0.7% |
| Phi-3.5-mini-instruct-Q4_K_M | 24333.7 | 3 | region_attention_compute | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 24333.7 | 3 | attention_compute | 19045.5 | 78.3% |
| Phi-3.5-mini-instruct-Q4_K_M | 24333.7 | 3 | attention_copies | 251.9 | 1% |
| Phi-3.5-mini-instruct-Q4_K_M | 24333.7 | 3 | attention_host | 602.5 | 2.5% |
| Phi-3.5-mini-instruct-Q4_K_M | 24333.7 | 3 | kv_mirror_copies | 129 | 0.5% |
| Phi-3.5-mini-instruct-Q4_K_M | 24333.7 | 3 | kv_host_write | 644.9 | 2.7% |
| Phi-3.5-mini-instruct-Q4_K_M | 24333.7 | 3 | swiglu_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 24333.7 | 3 | rmsnorm_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 24333.7 | 3 | rope_host | 650.2 | 2.7% |
| Phi-3.5-mini-instruct-Q4_K_M | 24333.7 | 3 | residual_add_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 24333.7 | 3 | bias_add_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 24333.7 | 3 | embed | 4.4 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 24333.7 | 3 | lm_head | 1.5 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 24333.7 | 3 | residue | 228.3 | 0.9% |
