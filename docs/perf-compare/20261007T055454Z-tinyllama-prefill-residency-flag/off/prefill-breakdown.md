| Model | Prefill ms | Reps | Term | ms | Share |
|---|---|---|---|---|---|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 61 | 3 | gemm_compute | 33 | 54.1% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 61 | 3 | weight_dequant | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 61 | 3 | matmul_staging | 1.5 | 2.5% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 61 | 3 | host_fp16_pack | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 61 | 3 | projection_and_region_host | 4.8 | 7.9% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 61 | 3 | device_elementwise | 8.9 | 14.6% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 61 | 3 | region_attention_compute | 4.2 | 6.9% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 61 | 3 | attention_compute | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 61 | 3 | attention_copies | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 61 | 3 | attention_host | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 61 | 3 | kv_mirror_copies | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 61 | 3 | kv_host_write | 0.8 | 1.3% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 61 | 3 | swiglu_host | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 61 | 3 | rmsnorm_host | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 61 | 3 | rope_host | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 61 | 3 | residual_add_host | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 61 | 3 | bias_add_host | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 61 | 3 | embed | 0.2 | 0.3% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 61 | 3 | lm_head | 1 | 1.6% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 61 | 3 | residue | 7.1 | 11.7% |
RESIDUE tinyllama-1.1b-chat-v1.0.Q4_K_M: 11.7% of prefill > 5%
