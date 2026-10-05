| Model | Prefill ms | Reps | Term | ms | Share |
|---|---|---|---|---|---|
| mistral-7b-instruct-v0.1-q4_k_m | 21041.4 | 3 | gemm_compute | 2795.2 | 13.3% |
| mistral-7b-instruct-v0.1-q4_k_m | 21041.4 | 3 | weight_dequant | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 21041.4 | 3 | matmul_staging | 66.9 | 0.3% |
| mistral-7b-instruct-v0.1-q4_k_m | 21041.4 | 3 | host_fp16_pack | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 21041.4 | 3 | projection_and_region_host | 96.6 | 0.5% |
| mistral-7b-instruct-v0.1-q4_k_m | 21041.4 | 3 | device_elementwise | 296.8 | 1.4% |
| mistral-7b-instruct-v0.1-q4_k_m | 21041.4 | 3 | region_attention_compute | 17584.7 | 83.6% |
| mistral-7b-instruct-v0.1-q4_k_m | 21041.4 | 3 | attention_compute | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 21041.4 | 3 | attention_copies | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 21041.4 | 3 | attention_host | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 21041.4 | 3 | kv_mirror_copies | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 21041.4 | 3 | kv_host_write | 84 | 0.4% |
| mistral-7b-instruct-v0.1-q4_k_m | 21041.4 | 3 | swiglu_host | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 21041.4 | 3 | rmsnorm_host | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 21041.4 | 3 | rope_host | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 21041.4 | 3 | residual_add_host | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 21041.4 | 3 | bias_add_host | 0 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 21041.4 | 3 | embed | 6.1 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 21041.4 | 3 | lm_head | 1.3 | 0% |
| mistral-7b-instruct-v0.1-q4_k_m | 21041.4 | 3 | residue | 108.5 | 0.5% |
