# GPU clock during the measured prefill windows (Mistral 7B, 512 and 2048)

Purpose: attribute Mistral 7B's non-attention prefill time per token growing from 512 to 2048 tokens
(+19.2% in `20261005T003146Z`/`20261005T004009Z`): GPU clock under sustained load, or the packed GEMM's
behaviour at 2048 rows.

Method: `nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,clocks_event_reasons.active -lms 100`
sampled for the whole of [`20261005T012110Z`](../20261005T012110Z/INDEX.md) (512) and this run (2048), both
`--device-spans`, default lane, unpinned, jar `3306ea4261f84a49`. Each measured repetition's prefill window is
its `juno.PrefillBatch` event; the samples inside it are in [`gpu-clocks-in-window.csv`](gpu-clocks-in-window.csv).
SM clock median is over samples with GPU utilization above 50%. Clock event reason `0x20` is software thermal
slowdown; `0x4` is the software power cap (corrected 2026-10-05: first published as the applications-clock setting, which is `0x2`).

```
mistral-7b-instruct-v0.1-q4_k_m-prefill- rows=512 window=    1523ms gemm=  616.0ms gemm/row=1.203 samples=15 sm_mhz med=1860 min=1835 max=1860 temp=52->61C power~160W reasons=['0x0000000000000000', '0x0000000000000004']
mistral-7b-instruct-v0.1-q4_k_m-prefill- rows=512 window=    1497ms gemm=  614.2ms gemm/row=1.200 samples=15 sm_mhz med=1841.0 min=1822 max=1860 temp=55->64C power~157W reasons=['0x0000000000000000', '0x0000000000000004']
mistral-7b-instruct-v0.1-q4_k_m-prefill- rows=512 window=    1497ms gemm=  611.2ms gemm/row=1.194 samples=15 sm_mhz med=1835 min=1809 max=1860 temp=58->67C power~162W reasons=['0x0000000000000000', '0x0000000000000004']
mistral-7b-instruct-v0.1-q4_k_m-prefill- rows=2048 window=   20999ms gemm= 2767.4ms gemm/row=1.351 samples=202 sm_mhz med=1607 min=1607 max=1809 temp=79->83C power~110W reasons=['0x0000000000000000', '0x0000000000000020']
mistral-7b-instruct-v0.1-q4_k_m-prefill- rows=2048 window=   21041ms gemm= 2795.2ms gemm/row=1.365 samples=199 sm_mhz med=1607 min=1607 max=1695 temp=84->90C power~110W reasons=['0x0000000000000000', '0x0000000000000020']
mistral-7b-instruct-v0.1-q4_k_m-prefill- rows=2048 window=   21072ms gemm= 2801.7ms gemm/row=1.368 samples=200 sm_mhz med=1607.0 min=1354 max=1708 temp=88->93C power~112W reasons=['0x0000000000000000', '0x0000000000000020']
```

**Reading.** At 512 the window runs at 1809 to 1860 MHz (median about 1845), 52 to 67 C, about 160 W. At 2048
the 21-second window heats the card to 79 to 93 C, the driver applies software thermal slowdown, and the SM clock
sits at 1607 MHz (low 1354). GEMM time per row rises 13.3% (1.199 to 1.361 ms median), against a 14.8% clock
drop, so clock-normalised GEMM time per row is flat (within 1.3%). The non-attention growth is the card's thermal
clock, not the GEMM at 2048 rows. The same clock drop inflates the 2048 attention time by a similar share.
