| Model | Prefill ms | Reps | Term | ms | Share |
|---|---|---|---|---|---|
| Phi-3.5-mini-instruct-Q4_K_M | 6324.5 | 1 | gemm_compute | 1292.3 | 20.4% |
| Phi-3.5-mini-instruct-Q4_K_M | 6324.5 | 1 | weight_dequant | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 6324.5 | 1 | matmul_staging | 317.6 | 5% |
| Phi-3.5-mini-instruct-Q4_K_M | 6324.5 | 1 | host_fp16_pack | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 6324.5 | 1 | projection_and_region_host | 898.2 | 14.2% |
| Phi-3.5-mini-instruct-Q4_K_M | 6324.5 | 1 | device_elementwise | 157.1 | 2.5% |
| Phi-3.5-mini-instruct-Q4_K_M | 6324.5 | 1 | region_attention_compute | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 6324.5 | 1 | attention_compute | 1036.3 | 16.4% |
| Phi-3.5-mini-instruct-Q4_K_M | 6324.5 | 1 | attention_copies | 249.8 | 3.9% |
| Phi-3.5-mini-instruct-Q4_K_M | 6324.5 | 1 | attention_host | 686.9 | 10.9% |
| Phi-3.5-mini-instruct-Q4_K_M | 6324.5 | 1 | kv_mirror_copies | 129.4 | 2% |
| Phi-3.5-mini-instruct-Q4_K_M | 6324.5 | 1 | kv_host_write | 646.7 | 10.2% |
| Phi-3.5-mini-instruct-Q4_K_M | 6324.5 | 1 | swiglu_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 6324.5 | 1 | rmsnorm_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 6324.5 | 1 | rope_host | 665.9 | 10.5% |
| Phi-3.5-mini-instruct-Q4_K_M | 6324.5 | 1 | residual_add_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 6324.5 | 1 | bias_add_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 6324.5 | 1 | embed | 4.3 | 0.1% |
| Phi-3.5-mini-instruct-Q4_K_M | 6324.5 | 1 | lm_head | 1.5 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 6324.5 | 1 | residue | 238.5 | 3.8% |
