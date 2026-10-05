| Model | Prefill ms | Reps | Term | ms | Share |
|---|---|---|---|---|---|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 855.2 | 1 | gemm_compute | 370.9 | 43.4% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 855.2 | 1 | weight_dequant | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 855.2 | 1 | matmul_staging | 15.7 | 1.8% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 855.2 | 1 | host_fp16_pack | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 855.2 | 1 | projection_and_region_host | 24.8 | 2.9% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 855.2 | 1 | device_elementwise | 85.6 | 10% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 855.2 | 1 | region_attention_compute | 290.9 | 34% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 855.2 | 1 | attention_compute | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 855.2 | 1 | attention_copies | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 855.2 | 1 | attention_host | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 855.2 | 1 | kv_mirror_copies | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 855.2 | 1 | kv_host_write | 13.2 | 1.5% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 855.2 | 1 | swiglu_host | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 855.2 | 1 | rmsnorm_host | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 855.2 | 1 | rope_host | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 855.2 | 1 | residual_add_host | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 855.2 | 1 | bias_add_host | 0 | 0% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 855.2 | 1 | embed | 3.4 | 0.4% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 855.2 | 1 | lm_head | 1.1 | 0.1% |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 855.2 | 1 | residue | 49.6 | 5.8% |
RESIDUE tinyllama-1.1b-chat-v1.0.Q4_K_M: 5.8% of prefill > 5%
