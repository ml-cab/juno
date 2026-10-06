| Model | Prefill ms | Reps | Term | ms | Share |
|---|---|---|---|---|---|
| Phi-3.5-mini-instruct-Q4_K_M | 763.7 | 3 | gemm_compute | 334.2 | 43.8% |
| Phi-3.5-mini-instruct-Q4_K_M | 763.7 | 3 | weight_dequant | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 763.7 | 3 | matmul_staging | 47.5 | 6.2% |
| Phi-3.5-mini-instruct-Q4_K_M | 763.7 | 3 | host_fp16_pack | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 763.7 | 3 | projection_and_region_host | 77.8 | 10.2% |
| Phi-3.5-mini-instruct-Q4_K_M | 763.7 | 3 | device_elementwise | 70.9 | 9.3% |
| Phi-3.5-mini-instruct-Q4_K_M | 763.7 | 3 | region_attention_compute | 73.7 | 9.7% |
| Phi-3.5-mini-instruct-Q4_K_M | 763.7 | 3 | attention_compute | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 763.7 | 3 | attention_copies | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 763.7 | 3 | attention_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 763.7 | 3 | kv_mirror_copies | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 763.7 | 3 | kv_host_write | 64.7 | 8.5% |
| Phi-3.5-mini-instruct-Q4_K_M | 763.7 | 3 | swiglu_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 763.7 | 3 | rmsnorm_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 763.7 | 3 | rope_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 763.7 | 3 | residual_add_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 763.7 | 3 | bias_add_host | 0 | 0% |
| Phi-3.5-mini-instruct-Q4_K_M | 763.7 | 3 | embed | 0.9 | 0.1% |
| Phi-3.5-mini-instruct-Q4_K_M | 763.7 | 3 | lm_head | 1.6 | 0.2% |
| Phi-3.5-mini-instruct-Q4_K_M | 763.7 | 3 | residue | 83.6 | 10.9% |
RESIDUE Phi-3.5-mini-instruct-Q4_K_M: 10.9% of prefill > 5%
