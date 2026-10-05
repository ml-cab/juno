| Model | Prefill ms | Reps | Term | ms | Share |
|---|---|---|---|---|---|
| mistral-7b-instruct-v0.1-q4_k_m | 1497.4 | 3 | gemm_compute | 614.2 | 41% |
| mistral-7b-instruct-v0.1-q4_k_m | 1497.4 | 3 | weight_dequant | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 1497.4 | 3 | matmul_staging | 17.8 | 1.2% |
| mistral-7b-instruct-v0.1-q4_k_m | 1497.4 | 3 | host_fp16_pack | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 1497.4 | 3 | projection_and_region_host | 35.7 | 2.4% |
| mistral-7b-instruct-v0.1-q4_k_m | 1497.4 | 3 | device_elementwise | 71.9 | 4.8% |
| mistral-7b-instruct-v0.1-q4_k_m | 1497.4 | 3 | region_attention_compute | 707.1 | 47.2% |
| mistral-7b-instruct-v0.1-q4_k_m | 1497.4 | 3 | attention_compute | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 1497.4 | 3 | attention_copies | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 1497.4 | 3 | attention_host | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 1497.4 | 3 | kv_mirror_copies | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 1497.4 | 3 | kv_host_write | 17.2 | 1.2% |
| mistral-7b-instruct-v0.1-q4_k_m | 1497.4 | 3 | swiglu_host | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 1497.4 | 3 | rmsnorm_host | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 1497.4 | 3 | rope_host | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 1497.4 | 3 | residual_add_host | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 1497.4 | 3 | bias_add_host | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 1497.4 | 3 | embed | 1.3 | 0.1% |
| mistral-7b-instruct-v0.1-q4_k_m | 1497.4 | 3 | lm_head | 1.2 | 0.1% |
| mistral-7b-instruct-v0.1-q4_k_m | 1497.4 | 3 | residue | 35.1 | 2.3% |
