| Model | Prefill ms | Reps | Term | ms | Share |
|---|---|---|---|---|---|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 57.3 | 3 | gemm_compute | 29.5 | 51.5% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 57.3 | 3 | weight_dequant | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 57.3 | 3 | matmul_staging | 1.6 | 2.9% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 57.3 | 3 | host_fp16_pack | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 57.3 | 3 | projection_and_region_host | 5.2 | 9.1% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 57.3 | 3 | device_elementwise | 8.7 | 15.2% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 57.3 | 3 | region_attention_compute | 4 | 6.9% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 57.3 | 3 | attention_compute | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 57.3 | 3 | attention_copies | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 57.3 | 3 | attention_host | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 57.3 | 3 | kv_mirror_copies | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 57.3 | 3 | kv_host_write | 0.8 | 1.4% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 57.3 | 3 | swiglu_host | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 57.3 | 3 | rmsnorm_host | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 57.3 | 3 | rope_host | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 57.3 | 3 | residual_add_host | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 57.3 | 3 | bias_add_host | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 57.3 | 3 | embed | 0.2 | 0.4% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 57.3 | 3 | lm_head | 1.2 | 2.1% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 57.3 | 3 | residue | 5.7 | 10% |
RESIDUE tinyllama-1.1b-chat-v1.0.Q4_K_M: 10% of prefill > 5%
