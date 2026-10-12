| Model | Prefill ms | Reps | Term | ms | Share |
|---|---|---|---|---|---|
| Phi-3.5-mini-instruct-Q4_K_M | 659.6 | 3 | gemm_compute | 321.4 | 48.7% |
| Phi-3.5-mini-instruct-Q4_K_M | 659.6 | 3 | weight_dequant | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 659.6 | 3 | matmul_staging | 43.8 | 6.6% |
| Phi-3.5-mini-instruct-Q4_K_M | 659.6 | 3 | host_fp16_pack | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 659.6 | 3 | projection_and_region_host | 75.1 | 11.4% |
| Phi-3.5-mini-instruct-Q4_K_M | 659.6 | 3 | device_elementwise | 62.3 | 9.4% |
| Phi-3.5-mini-instruct-Q4_K_M | 659.6 | 3 | region_attention_compute | 73.1 | 11.1% |
| Phi-3.5-mini-instruct-Q4_K_M | 659.6 | 3 | attention_compute | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 659.6 | 3 | attention_copies | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 659.6 | 3 | attention_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 659.6 | 3 | kv_mirror_copies | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 659.6 | 3 | kv_host_write | 62 | 9.4% |
| Phi-3.5-mini-instruct-Q4_K_M | 659.6 | 3 | swiglu_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 659.6 | 3 | rmsnorm_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 659.6 | 3 | rope_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 659.6 | 3 | residual_add_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 659.6 | 3 | bias_add_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 659.6 | 3 | embed | 1 | 0.1% |
| Phi-3.5-mini-instruct-Q4_K_M | 659.6 | 3 | lm_head | 1.6 | 0.2% |
| Phi-3.5-mini-instruct-Q4_K_M | 659.6 | 3 | residue | 15 | 2.3% |
