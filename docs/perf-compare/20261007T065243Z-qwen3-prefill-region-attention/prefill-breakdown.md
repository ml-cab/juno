| Model | Prefill ms | Reps | Term | ms | Share |
|---|---|---|---|---|---|
| Qwen3-1.7B-Q4_K_M | 306.8 | 3 | gemm_compute | 135.5 | 44.2% |
| Qwen3-1.7B-Q4_K_M | 306.8 | 3 | weight_dequant | 0 | 0% |
| Qwen3-1.7B-Q4_K_M | 306.8 | 3 | matmul_staging | 14.1 | 4.6% |
| Qwen3-1.7B-Q4_K_M | 306.8 | 3 | host_fp16_pack | 0 | 0% |
| Qwen3-1.7B-Q4_K_M | 306.8 | 3 | projection_and_region_host | 31 | 10.1% |
| Qwen3-1.7B-Q4_K_M | 306.8 | 3 | device_elementwise | 39.2 | 12.8% |
| Qwen3-1.7B-Q4_K_M | 306.8 | 3 | region_attention_compute | 38 | 12.4% |
| Qwen3-1.7B-Q4_K_M | 306.8 | 3 | attention_compute | 0 | 0% |
| Qwen3-1.7B-Q4_K_M | 306.8 | 3 | attention_copies | 0 | 0% |
| Qwen3-1.7B-Q4_K_M | 306.8 | 3 | attention_host | 0 | 0% |
| Qwen3-1.7B-Q4_K_M | 306.8 | 3 | kv_mirror_copies | 0 | 0% |
| Qwen3-1.7B-Q4_K_M | 306.8 | 3 | kv_host_write | 15.1 | 4.9% |
| Qwen3-1.7B-Q4_K_M | 306.8 | 3 | swiglu_host | 0 | 0% |
| Qwen3-1.7B-Q4_K_M | 306.8 | 3 | rmsnorm_host | 0 | 0% |
| Qwen3-1.7B-Q4_K_M | 306.8 | 3 | rope_host | 0 | 0% |
| Qwen3-1.7B-Q4_K_M | 306.8 | 3 | residual_add_host | 0 | 0% |
| Qwen3-1.7B-Q4_K_M | 306.8 | 3 | bias_add_host | 0 | 0% |
| Qwen3-1.7B-Q4_K_M | 306.8 | 3 | embed | 0.7 | 0.2% |
| Qwen3-1.7B-Q4_K_M | 306.8 | 3 | lm_head | 3.6 | 1.2% |
| Qwen3-1.7B-Q4_K_M | 306.8 | 3 | residue | 26 | 8.5% |
