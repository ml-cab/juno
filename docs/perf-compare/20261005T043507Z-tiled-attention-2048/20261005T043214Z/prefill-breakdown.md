| Model | Prefill ms | Reps | Term | ms | Share |
|---|---|---|---|---|---|
| mistral-7b-instruct-v0.1-q4_k_m | 4375.2 | 1 | gemm_compute | 2587.5 | 59.1% |
| mistral-7b-instruct-v0.1-q4_k_m | 4375.2 | 1 | weight_dequant | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 4375.2 | 1 | matmul_staging | 67.2 | 1.5% |
| mistral-7b-instruct-v0.1-q4_k_m | 4375.2 | 1 | host_fp16_pack | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 4375.2 | 1 | projection_and_region_host | 99.5 | 2.3% |
| mistral-7b-instruct-v0.1-q4_k_m | 4375.2 | 1 | device_elementwise | 279.7 | 6.4% |
| mistral-7b-instruct-v0.1-q4_k_m | 4375.2 | 1 | region_attention_compute | 1140.9 | 26.1% |
| mistral-7b-instruct-v0.1-q4_k_m | 4375.2 | 1 | attention_compute | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 4375.2 | 1 | attention_copies | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 4375.2 | 1 | attention_host | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 4375.2 | 1 | kv_mirror_copies | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 4375.2 | 1 | kv_host_write | 85.4 | 2% |
| mistral-7b-instruct-v0.1-q4_k_m | 4375.2 | 1 | swiglu_host | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 4375.2 | 1 | rmsnorm_host | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 4375.2 | 1 | rope_host | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 4375.2 | 1 | residual_add_host | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 4375.2 | 1 | bias_add_host | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 4375.2 | 1 | embed | 6.3 | 0.1% |
| mistral-7b-instruct-v0.1-q4_k_m | 4375.2 | 1 | lm_head | 1.1 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 4375.2 | 1 | residue | 107.6 | 2.5% |
