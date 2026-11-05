# Kit reference run: linux-amd64-xeon-e5-1650-v2-geforce-gtx-1080 (20261104T173924Z)

**Written by `juno-kit ingest --reference-run`; reviewed 2026-11-04 (notes below).** Unpinned. This is the reference
that later ingests at Juno `80ec949e` compare their correctness readings with; its readings are not a throughput
gate. Run with `dist/kit-reference/run-reference.sh` on a clean tree (tracked files unchanged).

**Host.** Linux 7.0.0-42-generic amd64; Intel(R) Xeon(R) CPU E5-1650 v2 @ 3.50GHz, 6 cores, 12 threads, 62 GiB RAM; JVM vectors: integer 128-bit, float 128-bit; GPU: NVIDIA GeForce GTX 1080

**Build.** Juno `80ec949ebc9796be4c45feda2865396f44d511a3`; kit 0.1.2, kit jar sha256 `bba50dfb33c23ca591423b0573c4bf532ea00c9534c77a796fe17ad52b073393`.

**Run.** Suite `all`, 20261104T043255Z to 20261104T173924Z; 72 probes: 30 DONE, 42 PASS.

## Review notes (2026-11-04)

- **Every runnable probe passed or completed**, first attempt each except calibration (attempt 1 re-run: other processes
  used 6%). Not runnable here, as expected: `cpu.kernel-width` (this JVM prefers 128-bit integer vectors) and
  `rocm.unit-tests` (no AMD GPU). The CPU unit tests read 2101 tests, 0 failures, 25 skipped; the CUDA group 367 tests,
  0 failures, 7 skipped.
- **Estimate.** First estimate 692 to 978 min (point 48,878 s; calibration 1.00x) against 47,189 s from start to
  finish (13:06:29): point over actual **1.04x**, inside the range. The large probes ran within a few percent of their
  catalog costs (for example `context-policy.phi-3.5-mini` 11,098 s against 11,250, `long-prompt.mistral-7b` 3,864
  against 3,855, `quality.phi-3.5-mini` 4,188 against 4,170). This is the `all` suite; the kit's estimate criterion is
  scored on `quick`.
- **`cpu.live.tinyllama` took 294 s** against a 250 s cost: check 9's 573-token window read 34.1 s and check 10's
  chunk-64 window 10.5 s (16.5 s and 5.1 s on 2026-10-28). Only the ten Q6_K layers were slow, the pattern of the JIT
  compiling the Q6_K prefill kernel with boxed vectors; see `20261104T191116Z-cpu-prefill-jit/`. The checks are
  unaffected (10 of 10 pass), and the timed readings in this run show no sign of it (small spreads).
- **Coverage table.** The ROCm row counts the GPU probes that run on whichever backend loads; on this host they all ran
  on CUDA, so the row says nothing about ROCm here. The Qwen3 mixture-of-experts row is the handler's synthetic-model
  unit tests only (no real file in the kit's catalog).
- **Kit output to fix later:** the calibration note ends in "`;.`"; `README-row.md` has two cells where the
  perf-compare README's table has four (the README row was written by hand); the coverage table does not say which
  GPU backend its GPU rows ran on. None changes a result.
- **Checked by ingest:** a later zip from this host at the same commit (`20261104T183707Z`, `cpu.live.tinyllama` alone)
  ingested against this run reads "10 items compared: 10 equal or within the bound". That test draft was not published.
- **Files** also include `estimate.json` (the first estimate), `skipped.json` (the two probes not runnable here) and
  `README-row.md` (the draft row).

## Probes

| Probe | Status | Time | Attempts | Other CPU | Note |
|---|---|---|---|---|---|
| `calibration` | DONE | 0:17 | 2 | 3% | attempt 1 re-run: other processes used 6% of the CPU;. |
| `env` | DONE | 0:01 | 1 | 0% |  |
| `cpu.live.tinyllama` | PASS | 4:54 | 1 | 5% |  |
| `cpu.throughput.tinyllama.128` | DONE | 0:59 | 1 | 3% |  |
| `cpu.throughput.tinyllama.512` | DONE | 2:08 | 1 | 2% |  |
| `gpu.live.tinyllama` | PASS | 1:38 | 1 | 2% |  |
| `gpu.throughput.tinyllama.512` | DONE | 0:11 | 1 | 2% |  |
| `gpu.throughput.tinyllama.2048` | DONE | 0:16 | 1 | 2% |  |
| `gpu.residency.tinyllama` | PASS | 0:46 | 1 | 2% |  |
| `gpu.packed-kquant.tinyllama.static` | PASS | 0:15 | 1 | 1% |  |
| `gpu.packed-kquant.tinyllama.continuous` | PASS | 1:14 | 1 | 2% |  |
| `cpu.live.qwen2.5-3b` | PASS | 8:26 | 1 | 2% |  |
| `cpu.live.phi-3.5-mini` | PASS | 14:27 | 1 | 2% |  |
| `cpu.live.mistral-7b` | PASS | 19:47 | 1 | 2% |  |
| `cpu.live.qwen3-1.7b` | PASS | 5:39 | 1 | 2% |  |
| `gpu.live.qwen2.5-3b` | PASS | 2:29 | 1 | 2% |  |
| `gpu.live.phi-3.5-mini` | PASS | 3:25 | 1 | 2% |  |
| `gpu.live.mistral-7b` | PASS | 5:02 | 1 | 2% |  |
| `gpu.live.qwen3-1.7b` | PASS | 2:09 | 1 | 2% |  |
| `consistency.tinyllama` | PASS | 1:37 | 1 | 3% |  |
| `gpu.consistency.tinyllama` | PASS | 0:13 | 1 | 2% |  |
| `grammar.tinyllama` | PASS | 1:37 | 1 | 2% |  |
| `tools.qwen2.5-3b` | PASS | 3:09 | 1 | 2% |  |
| `context-policy.phi-3.5-mini` | PASS | 3:04:58 | 1 | 2% |  |
| `gpu.context-policy.phi-3.5-mini` | PASS | 4:14 | 1 | 2% |  |
| `long-prompt.tinyllama` | PASS | 10:26 | 1 | 2% |  |
| `long-prompt.mistral-7b` | PASS | 1:04:24 | 1 | 2% |  |
| `gpu.long-prompt.tinyllama` | PASS | 0:44 | 1 | 2% |  |
| `gpu.long-prompt.mistral-7b` | PASS | 1:33 | 1 | 2% |  |
| `cpu.hot-path.tinyllama-1.1b` | PASS | 1:39 | 1 | 2% |  |
| `cpu.hot-path.qwen2.5-3b` | PASS | 4:07 | 1 | 2% |  |
| `cpu.hot-path.phi-3.5-mini` | PASS | 5:29 | 1 | 2% |  |
| `cpu.hot-path.mistral-7b` | PASS | 9:06 | 1 | 2% |  |
| `quality.tinyllama-1.1b` | PASS | 7:37 | 1 | 2% |  |
| `quality.qwen2.5-3b` | PASS | 23:47 | 1 | 2% |  |
| `quality.phi-3.5-mini` | PASS | 1:09:48 | 1 | 2% |  |
| `quality.mistral-7b` | PASS | 47:06 | 1 | 2% |  |
| `cpu.throughput.tinyllama-1.1b.2048` | DONE | 7:38 | 1 | 2% |  |
| `gpu.throughput.tinyllama-1.1b.128` | DONE | 0:10 | 1 | 1% |  |
| `cpu.throughput.qwen2.5-3b.128` | DONE | 2:33 | 1 | 2% |  |
| `cpu.throughput.qwen2.5-3b.512` | DONE | 5:44 | 1 | 2% |  |
| `cpu.throughput.qwen2.5-3b.2048` | DONE | 19:31 | 1 | 2% |  |
| `cpu.throughput.phi-3.5-mini.128` | DONE | 3:15 | 1 | 2% |  |
| `cpu.throughput.phi-3.5-mini.512` | DONE | 7:14 | 1 | 2% |  |
| `cpu.throughput.phi-3.5-mini.2048` | DONE | 29:31 | 1 | 3% |  |
| `cpu.throughput.mistral-7b.128` | DONE | 5:11 | 1 | 2% |  |
| `cpu.throughput.mistral-7b.512` | DONE | 12:34 | 1 | 2% |  |
| `cpu.throughput.mistral-7b.2048` | DONE | 45:02 | 1 | 2% |  |
| `gpu.throughput.qwen2.5-3b.128` | DONE | 0:21 | 1 | 3% |  |
| `gpu.throughput.qwen2.5-3b.512` | DONE | 0:23 | 1 | 2% |  |
| `gpu.throughput.qwen2.5-3b.2048` | DONE | 0:33 | 1 | 4% |  |
| `gpu.throughput.phi-3.5-mini.128` | DONE | 0:16 | 1 | 1% |  |
| `gpu.throughput.phi-3.5-mini.512` | DONE | 0:19 | 1 | 1% |  |
| `gpu.throughput.phi-3.5-mini.2048` | DONE | 0:36 | 1 | 2% |  |
| `gpu.throughput.mistral-7b.128` | DONE | 0:23 | 1 | 2% |  |
| `gpu.throughput.mistral-7b.512` | DONE | 0:27 | 1 | 2% |  |
| `gpu.throughput.mistral-7b.2048` | DONE | 0:51 | 1 | 2% |  |
| `parallel.tinyllama` | DONE | 14:11 | 1 | 3% |  |
| `gpu.parallel.tinyllama` | DONE | 2:18 | 1 | 2% |  |
| `schedule.tinyllama` | PASS | 5:07 | 1 | 2% |  |
| `gpu.schedule.tinyllama` | PASS | 1:03 | 1 | 2% |  |
| `mixed-prefill.tinyllama` | PASS | 0:53 | 1 | 2% |  |
| `gpu.mixed-prefill.tinyllama` | PASS | 0:24 | 1 | 2% |  |
| `prefill-batch.tinyllama` | DONE | 4:20 | 1 | 2% |  |
| `gpu.prefill-batch.tinyllama` | DONE | 1:11 | 1 | 3% |  |
| `lora.tinyllama` | PASS | 12:51 | 1 | 3% |  |
| `gpu.lora.tinyllama` | PASS | 1:09 | 1 | 2% |  |
| `vision.moondream2` | PASS | 17:49 | 1 | 2% |  |
| `gpu.vision.moondream2` | PASS | 7:10 | 1 | 2% |  |
| `launcher.tinyllama` | PASS | 0:13 | 1 | 4% |  |
| `cpu.unit-tests` | PASS | 35:57 | 1 | 2% |  |
| `gpu.unit-tests` | PASS | 23:42 | 1 | 2% |  |

## Readings

Each reading's repetitions on this host. They are not compared with the reference host.

| Probe | Reading | Unit | Median | Min | Max | n | Status |
|---|---|---|---|---|---|---|---|
| `calibration` | q4k_decode_ms_per_call | ms | 0.906532 | 0.864428 | 0.931245 | 5 | DONE |
| `cpu.throughput.tinyllama.128` | prefill t/s | t/s | 35.1292 | 34.5557 | 35.1877 | 3 | DONE |
| `cpu.throughput.tinyllama.128` | generation t/s | t/s | 12.4736 | 12.3713 | 12.6429 | 3 | DONE |
| `cpu.throughput.tinyllama.128` | generation allocation per token in recording | B/token | 407753 | 390209 | 453483 | 3 | DONE |
| `cpu.throughput.tinyllama.128` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.tinyllama.128` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.tinyllama.512` | prefill t/s | t/s | 33.6685 | 32.423 | 33.9573 | 3 | DONE |
| `cpu.throughput.tinyllama.512` | generation t/s | t/s | 12.6218 | 12.3517 | 12.7276 | 3 | DONE |
| `cpu.throughput.tinyllama.512` | generation allocation per token in recording | B/token | 418040 | 400491 | 448405 | 3 | DONE |
| `cpu.throughput.tinyllama.512` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.tinyllama.512` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.tinyllama.512` | prefill t/s | t/s | 2801.51 | 2693.54 | 2819.96 | 3 | DONE |
| `gpu.throughput.tinyllama.512` | generation t/s | t/s | 168.509 | 167.403 | 173.128 | 3 | DONE |
| `gpu.throughput.tinyllama.512` | generation allocation per token in recording | B/token | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.tinyllama.512` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.tinyllama.512` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.tinyllama.2048` | prefill t/s | t/s | 2377.6 | 2334.15 | 2378.12 | 3 | DONE |
| `gpu.throughput.tinyllama.2048` | generation t/s | t/s | 168.936 | 167.627 | 173.84 | 3 | DONE |
| `gpu.throughput.tinyllama.2048` | generation allocation per token in recording | B/token | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.tinyllama.2048` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.tinyllama.2048` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `quality.tinyllama-1.1b` | mean NLL, integer kernels | nats | 2.45609 | 2.45609 | 2.45609 | 1 | PASS |
| `quality.tinyllama-1.1b` | mean NLL, float-activation kernels | nats | 2.45566 | 2.45566 | 2.45566 | 1 | PASS |
| `quality.tinyllama-1.1b` | argmax agreement, first 512 positions | % | 99.0234 | 99.0234 | 99.0234 | 1 | PASS |
| `quality.qwen2.5-3b` | mean NLL, integer kernels | nats | 1.69656 | 1.69656 | 1.69656 | 1 | PASS |
| `quality.qwen2.5-3b` | mean NLL, float-activation kernels | nats | 1.69766 | 1.69766 | 1.69766 | 1 | PASS |
| `quality.qwen2.5-3b` | argmax agreement, first 512 positions | % | 100 | 100 | 100 | 1 | PASS |
| `quality.phi-3.5-mini` | mean NLL, integer kernels | nats | 1.3611 | 1.3611 | 1.3611 | 1 | PASS |
| `quality.phi-3.5-mini` | mean NLL, float-activation kernels | nats | 1.35826 | 1.35826 | 1.35826 | 1 | PASS |
| `quality.phi-3.5-mini` | argmax agreement, first 512 positions | % | 99.4141 | 99.4141 | 99.4141 | 1 | PASS |
| `quality.mistral-7b` | mean NLL, integer kernels | nats | 1.53013 | 1.53013 | 1.53013 | 1 | PASS |
| `quality.mistral-7b` | mean NLL, float-activation kernels | nats | 1.52891 | 1.52891 | 1.52891 | 1 | PASS |
| `quality.mistral-7b` | argmax agreement, first 512 positions | % | 99.6094 | 99.6094 | 99.6094 | 1 | PASS |
| `cpu.throughput.tinyllama-1.1b.2048` | prefill t/s | t/s | 29.54 | 28.8638 | 29.5527 | 3 | DONE |
| `cpu.throughput.tinyllama-1.1b.2048` | generation t/s | t/s | 11.3813 | 11.2539 | 11.7358 | 3 | DONE |
| `cpu.throughput.tinyllama-1.1b.2048` | generation allocation per token in recording | B/token | 418187 | 417523 | 447461 | 3 | DONE |
| `cpu.throughput.tinyllama-1.1b.2048` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.tinyllama-1.1b.2048` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.tinyllama-1.1b.128` | prefill t/s | t/s | 1936.15 | 1838.12 | 2023.05 | 3 | DONE |
| `gpu.throughput.tinyllama-1.1b.128` | generation t/s | t/s | 169.121 | 166.373 | 172.59 | 3 | DONE |
| `gpu.throughput.tinyllama-1.1b.128` | generation allocation per token in recording | B/token | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.tinyllama-1.1b.128` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.tinyllama-1.1b.128` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.qwen2.5-3b.128` | prefill t/s | t/s | 12.4724 | 12.4565 | 12.5731 | 3 | DONE |
| `cpu.throughput.qwen2.5-3b.128` | generation t/s | t/s | 4.43615 | 4.37249 | 4.46036 | 3 | DONE |
| `cpu.throughput.qwen2.5-3b.128` | generation allocation per token in recording | B/token | 1365680 | 1357440 | 1391070 | 3 | DONE |
| `cpu.throughput.qwen2.5-3b.128` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.qwen2.5-3b.128` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.qwen2.5-3b.512` | prefill t/s | t/s | 12.2692 | 12.0159 | 12.3202 | 3 | DONE |
| `cpu.throughput.qwen2.5-3b.512` | generation t/s | t/s | 4.43887 | 4.05581 | 4.48165 | 3 | DONE |
| `cpu.throughput.qwen2.5-3b.512` | generation allocation per token in recording | B/token | 1360260 | 1343160 | 1371710 | 3 | DONE |
| `cpu.throughput.qwen2.5-3b.512` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.qwen2.5-3b.512` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.qwen2.5-3b.2048` | prefill t/s | t/s | 11.1219 | 11.0667 | 11.458 | 3 | DONE |
| `cpu.throughput.qwen2.5-3b.2048` | generation t/s | t/s | 4.49548 | 4.48146 | 4.52832 | 3 | DONE |
| `cpu.throughput.qwen2.5-3b.2048` | generation allocation per token in recording | B/token | 1358970 | 1335760 | 1389330 | 3 | DONE |
| `cpu.throughput.qwen2.5-3b.2048` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.qwen2.5-3b.2048` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.phi-3.5-mini.128` | prefill t/s | t/s | 10.2314 | 10.154 | 10.5648 | 3 | DONE |
| `cpu.throughput.phi-3.5-mini.128` | generation t/s | t/s | 3.14205 | 3.12941 | 3.14808 | 3 | DONE |
| `cpu.throughput.phi-3.5-mini.128` | generation allocation per token in recording | B/token | 458941 | 451486 | 495287 | 3 | DONE |
| `cpu.throughput.phi-3.5-mini.128` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.phi-3.5-mini.128` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.phi-3.5-mini.512` | prefill t/s | t/s | 10.1072 | 10.0154 | 10.1157 | 3 | DONE |
| `cpu.throughput.phi-3.5-mini.512` | generation t/s | t/s | 2.90245 | 2.87044 | 3.07448 | 3 | DONE |
| `cpu.throughput.phi-3.5-mini.512` | generation allocation per token in recording | B/token | 459093 | 452073 | 495444 | 3 | DONE |
| `cpu.throughput.phi-3.5-mini.512` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.phi-3.5-mini.512` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.phi-3.5-mini.2048` | prefill t/s | t/s | 7.65206 | 7.31118 | 7.66638 | 3 | DONE |
| `cpu.throughput.phi-3.5-mini.2048` | generation t/s | t/s | 3.17321 | 3.1484 | 3.1825 | 3 | DONE |
| `cpu.throughput.phi-3.5-mini.2048` | generation allocation per token in recording | B/token | 464088 | 451822 | 489450 | 3 | DONE |
| `cpu.throughput.phi-3.5-mini.2048` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.phi-3.5-mini.2048` | generation GC pause in token span | ms | 0 | 0 | 6.93014 | 3 | DONE |
| `cpu.throughput.mistral-7b.128` | prefill t/s | t/s | 5.3891 | 5.38244 | 5.41221 | 3 | DONE |
| `cpu.throughput.mistral-7b.128` | generation t/s | t/s | 2.1877 | 2.17649 | 2.2115 | 3 | DONE |
| `cpu.throughput.mistral-7b.128` | generation allocation per token in recording | B/token | 495601 | 493837 | 540989 | 3 | DONE |
| `cpu.throughput.mistral-7b.128` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.mistral-7b.128` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.mistral-7b.512` | prefill t/s | t/s | 5.11324 | 5.08635 | 5.37166 | 3 | DONE |
| `cpu.throughput.mistral-7b.512` | generation t/s | t/s | 2.13369 | 2.02923 | 2.17496 | 3 | DONE |
| `cpu.throughput.mistral-7b.512` | generation allocation per token in recording | B/token | 495366 | 492007 | 539616 | 3 | DONE |
| `cpu.throughput.mistral-7b.512` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.mistral-7b.512` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.mistral-7b.2048` | prefill t/s | t/s | 4.82273 | 4.70161 | 4.85627 | 3 | DONE |
| `cpu.throughput.mistral-7b.2048` | generation t/s | t/s | 2.15023 | 2.07312 | 2.16281 | 3 | DONE |
| `cpu.throughput.mistral-7b.2048` | generation allocation per token in recording | B/token | 498563 | 495584 | 545682 | 3 | DONE |
| `cpu.throughput.mistral-7b.2048` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `cpu.throughput.mistral-7b.2048` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.qwen2.5-3b.128` | prefill t/s | t/s | 951.72 | 903.125 | 968.558 | 3 | DONE |
| `gpu.throughput.qwen2.5-3b.128` | generation t/s | t/s | 69.0303 | 67.2765 | 69.0724 | 3 | DONE |
| `gpu.throughput.qwen2.5-3b.128` | generation allocation per token in recording | B/token | 1524120 | 1365340 | 1695340 | 3 | DONE |
| `gpu.throughput.qwen2.5-3b.128` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.qwen2.5-3b.128` | generation GC pause in token span | ms | 0 | 0 | 10.1495 | 3 | DONE |
| `gpu.throughput.qwen2.5-3b.512` | prefill t/s | t/s | 1227.78 | 1166.23 | 1235.09 | 3 | DONE |
| `gpu.throughput.qwen2.5-3b.512` | generation t/s | t/s | 67.5167 | 67.2356 | 68.4962 | 3 | DONE |
| `gpu.throughput.qwen2.5-3b.512` | generation allocation per token in recording | B/token | 1534170 | 1362480 | 1637420 | 3 | DONE |
| `gpu.throughput.qwen2.5-3b.512` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.qwen2.5-3b.512` | generation GC pause in token span | ms | 0 | 0 | 2.8785 | 3 | DONE |
| `gpu.throughput.qwen2.5-3b.2048` | prefill t/s | t/s | 994.29 | 987.037 | 997.044 | 3 | DONE |
| `gpu.throughput.qwen2.5-3b.2048` | generation t/s | t/s | 68.3775 | 67.7809 | 68.5034 | 3 | DONE |
| `gpu.throughput.qwen2.5-3b.2048` | generation allocation per token in recording | B/token | 1540830 | 1364520 | 1639010 | 3 | DONE |
| `gpu.throughput.qwen2.5-3b.2048` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.qwen2.5-3b.2048` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.phi-3.5-mini.128` | prefill t/s | t/s | 792.559 | 759.04 | 812.994 | 3 | DONE |
| `gpu.throughput.phi-3.5-mini.128` | generation t/s | t/s | 60.4122 | 59.9624 | 60.5323 | 3 | DONE |
| `gpu.throughput.phi-3.5-mini.128` | generation allocation per token in recording | B/token | 349959 | 348547 | 783227 | 3 | DONE |
| `gpu.throughput.phi-3.5-mini.128` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.phi-3.5-mini.128` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.phi-3.5-mini.512` | prefill t/s | t/s | 848.545 | 835.7 | 851.56 | 3 | DONE |
| `gpu.throughput.phi-3.5-mini.512` | generation t/s | t/s | 60.7754 | 60.3392 | 60.9052 | 3 | DONE |
| `gpu.throughput.phi-3.5-mini.512` | generation allocation per token in recording | B/token | 353825 | 351817 | 790180 | 3 | DONE |
| `gpu.throughput.phi-3.5-mini.512` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.phi-3.5-mini.512` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.phi-3.5-mini.2048` | prefill t/s | t/s | 658.717 | 657.639 | 661.494 | 3 | DONE |
| `gpu.throughput.phi-3.5-mini.2048` | generation t/s | t/s | 60.1012 | 59.8484 | 60.2249 | 3 | DONE |
| `gpu.throughput.phi-3.5-mini.2048` | generation allocation per token in recording | B/token | 698005 | 353901 | 785863 | 3 | DONE |
| `gpu.throughput.phi-3.5-mini.2048` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.phi-3.5-mini.2048` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.mistral-7b.128` | prefill t/s | t/s | 508.694 | 506.147 | 509.838 | 3 | DONE |
| `gpu.throughput.mistral-7b.128` | generation t/s | t/s | 40.429 | 40.2795 | 40.5417 | 3 | DONE |
| `gpu.throughput.mistral-7b.128` | generation allocation per token in recording | B/token | 750527 | 744336 | 1037000 | 3 | DONE |
| `gpu.throughput.mistral-7b.128` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.mistral-7b.128` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.mistral-7b.512` | prefill t/s | t/s | 564.55 | 545.629 | 571.738 | 3 | DONE |
| `gpu.throughput.mistral-7b.512` | generation t/s | t/s | 40.2048 | 40.2005 | 40.2869 | 3 | DONE |
| `gpu.throughput.mistral-7b.512` | generation allocation per token in recording | B/token | 736564 | 734599 | 1027010 | 3 | DONE |
| `gpu.throughput.mistral-7b.512` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.mistral-7b.512` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.mistral-7b.2048` | prefill t/s | t/s | 440.316 | 439.657 | 441 | 3 | DONE |
| `gpu.throughput.mistral-7b.2048` | generation t/s | t/s | 38.2663 | 38.1808 | 38.2848 | 3 | DONE |
| `gpu.throughput.mistral-7b.2048` | generation allocation per token in recording | B/token | 781746 | 722156 | 973009 | 3 | DONE |
| `gpu.throughput.mistral-7b.2048` | prefill GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `gpu.throughput.mistral-7b.2048` | generation GC pause in token span | ms | 0 | 0 | 0 | 3 | DONE |
| `parallel.tinyllama` | aggregate t/s, --parallel 1 | t/s | 10.8438 | 10.7753 | 10.8485 | 3 | DONE |
| `parallel.tinyllama` | mean request latency, --parallel 1 | ms | 47148.4 | 47074.2 | 47410.3 | 3 | DONE |
| `parallel.tinyllama` | aggregate t/s, --parallel 8 | t/s | 12.3976 | 12.1438 | 12.4018 | 3 | DONE |
| `parallel.tinyllama` | mean request latency, --parallel 8 | ms | 41297.7 | 41283.7 | 42161.2 | 3 | DONE |
| `gpu.parallel.tinyllama` | aggregate t/s, --parallel 1 | t/s | 104.179 | 101.081 | 104.674 | 3 | DONE |
| `gpu.parallel.tinyllama` | mean request latency, --parallel 1 | ms | 4770.28 | 4762.61 | 4947.55 | 3 | DONE |
| `gpu.parallel.tinyllama` | aggregate t/s, --parallel 8 | t/s | 84.4684 | 83.1384 | 84.8719 | 3 | DONE |
| `gpu.parallel.tinyllama` | mean request latency, --parallel 8 | ms | 6061.14 | 6032.28 | 6157.63 | 3 | DONE |
| `schedule.tinyllama` | static: aggregate t/s | t/s | 12.0937 | 12.0937 | 12.0937 | 1 | PASS |
| `schedule.tinyllama` | static: mean TTFT | ms | 8063.58 | 8063.58 | 8063.58 | 1 | PASS |
| `schedule.tinyllama` | static: mean TPOT | ms | 641.44 | 641.44 | 641.44 | 1 | PASS |
| `schedule.tinyllama` | static: first turn latency | ms | 1747.39 | 1747.39 | 1747.39 | 1 | PASS |
| `schedule.tinyllama` | static: mean later turn latency | ms | 1350.61 | 1350.61 | 1350.61 | 1 | PASS |
| `schedule.tinyllama` | continuous: aggregate t/s | t/s | 12.1448 | 12.1448 | 12.1448 | 1 | PASS |
| `schedule.tinyllama` | continuous: mean TTFT | ms | 7413.32 | 7413.32 | 7413.32 | 1 | PASS |
| `schedule.tinyllama` | continuous: mean TPOT | ms | 555.071 | 555.071 | 555.071 | 1 | PASS |
| `schedule.tinyllama` | continuous: first turn latency | ms | 2003.19 | 2003.19 | 2003.19 | 1 | PASS |
| `schedule.tinyllama` | continuous: mean later turn latency | ms | 1531.23 | 1531.23 | 1531.23 | 1 | PASS |
| `gpu.schedule.tinyllama` | static: aggregate t/s | t/s | 79.0073 | 79.0073 | 79.0073 | 1 | PASS |
| `gpu.schedule.tinyllama` | static: mean TTFT | ms | 255.349 | 255.349 | 255.349 | 1 | PASS |
| `gpu.schedule.tinyllama` | static: mean TPOT | ms | 67.9554 | 67.9554 | 67.9554 | 1 | PASS |
| `gpu.schedule.tinyllama` | static: first turn latency | ms | 137.931 | 137.931 | 137.931 | 1 | PASS |
| `gpu.schedule.tinyllama` | static: mean later turn latency | ms | 123.926 | 123.926 | 123.926 | 1 | PASS |
| `gpu.schedule.tinyllama` | continuous: aggregate t/s | t/s | 77.985 | 77.985 | 77.985 | 1 | PASS |
| `gpu.schedule.tinyllama` | continuous: mean TTFT | ms | 277.938 | 277.938 | 277.938 | 1 | PASS |
| `gpu.schedule.tinyllama` | continuous: mean TPOT | ms | 99.5289 | 99.5289 | 99.5289 | 1 | PASS |
| `gpu.schedule.tinyllama` | continuous: first turn latency | ms | 338.174 | 338.174 | 338.174 | 1 | PASS |
| `gpu.schedule.tinyllama` | continuous: mean later turn latency | ms | 331.367 | 331.367 | 331.367 | 1 | PASS |
| `mixed-prefill.tinyllama` | mixed: short-request TTFT | ms | 7507.83 | 7507.83 | 7507.83 | 1 | PASS |
| `mixed-prefill.tinyllama` | mixed: short-request TPOT | ms | 1233.64 | 1233.64 | 1233.64 | 1 | PASS |
| `mixed-prefill.tinyllama` | mixed: long-request TTFT | ms | 16313 | 16313 | 16313 | 1 | PASS |
| `mixed-prefill.tinyllama` | admit: short-request TTFT | ms | 13899.6 | 13899.6 | 13899.6 | 1 | PASS |
| `mixed-prefill.tinyllama` | admit: short-request TPOT | ms | 347.003 | 347.003 | 347.003 | 1 | PASS |
| `mixed-prefill.tinyllama` | admit: long-request TTFT | ms | 13899.9 | 13899.9 | 13899.9 | 1 | PASS |
| `gpu.mixed-prefill.tinyllama` | mixed: short-request TTFT | ms | 604.288 | 604.288 | 604.288 | 1 | PASS |
| `gpu.mixed-prefill.tinyllama` | mixed: short-request TPOT | ms | 146.907 | 146.907 | 146.907 | 1 | PASS |
| `gpu.mixed-prefill.tinyllama` | mixed: long-request TTFT | ms | 1694.02 | 1694.02 | 1694.02 | 1 | PASS |
| `gpu.mixed-prefill.tinyllama` | admit: short-request TTFT | ms | 749.837 | 749.837 | 749.837 | 1 | PASS |
| `gpu.mixed-prefill.tinyllama` | admit: short-request TPOT | ms | 138.9 | 138.9 | 138.9 | 1 | PASS |
| `gpu.mixed-prefill.tinyllama` | admit: long-request TTFT | ms | 744.107 | 744.107 | 744.107 | 1 | PASS |
| `prefill-batch.tinyllama` | prefill t/s, --prefill-batch 1 | t/s | 4.78147 | 4.68444 | 4.80543 | 3 | DONE |
| `prefill-batch.tinyllama` | prefill t/s, --prefill-batch 32 | t/s | 26.3555 | 25.486 | 26.4963 | 3 | DONE |
| `gpu.prefill-batch.tinyllama` | prefill t/s, --prefill-batch 1 | t/s | 58.2231 | 58.1079 | 58.9922 | 3 | DONE |
| `gpu.prefill-batch.tinyllama` | prefill t/s, --prefill-batch 32 | t/s | 1003.89 | 978.594 | 1019.23 | 3 | DONE |
| `lora.tinyllama` | training time per pass | ms | 48836.2 | 48836.2 | 48836.2 | 1 | PASS |
| `lora.tinyllama` | playback t/s, --lora-play | t/s | 0.898942 | 0.898942 | 0.898942 | 1 | PASS |
| `lora.tinyllama` | playback t/s, merged model | t/s | 4.35007 | 4.35007 | 4.35007 | 1 | PASS |
| `gpu.lora.tinyllama` | training time per pass | ms | 2795.58 | 2795.58 | 2795.58 | 1 | PASS |
| `gpu.lora.tinyllama` | playback t/s, --lora-play | t/s | 11.8916 | 11.8916 | 11.8916 | 1 | PASS |
| `gpu.lora.tinyllama` | playback t/s, merged model | t/s | 30.3172 | 30.3172 | 30.3172 | 1 | PASS |
| `vision.moondream2` | threads 4: latency | ms | 315412 | 315412 | 315412 | 1 | PASS |
| `vision.moondream2` | threads 4: decode t/s | t/s | 3.93958 | 3.93958 | 3.93958 | 1 | PASS |
| `vision.moondream2` | threads 12: latency | ms | 204027 | 204027 | 204027 | 1 | PASS |
| `vision.moondream2` | threads 12: decode t/s | t/s | 5.51959 | 5.51959 | 5.51959 | 1 | PASS |

## Correctness against the reference run

This is a reference run: later ingests at Juno `80ec949e` compare with it.

## Coverage on this host

| Item | Title | Probes run here |
|---|---|---|
| surface:1 | CPU inference | 12 DONE, 14 PASS |
| surface:2 | CUDA GPU inference | 12 DONE, 9 PASS |
| surface:3 | ROCm GPU inference | 12 DONE, 4 PASS |
| surface:4 | Static schedule | 2 DONE, 5 PASS |
| surface:5 | Continuous schedule | 11 PASS |
| surface:6 | Single-node local mode | 7 PASS |
| surface:7 | Pipeline-parallel cluster | 10 PASS |
| surface:8 | Tensor-parallel cluster | 10 PASS |
| surface:9 | LoRA training | 2 PASS |
| surface:10 | LoRA playback | 3 PASS |
| surface:11 | Vision | 2 PASS |
| surface:12 | OpenAI-compatible REST surface | 11 PASS |
| surface:13 | Native REST surface | 2 PASS |
| surface:14 | CLI direct usage | 2 PASS |
| surface:15 | JVM embedding facade | 24 DONE, 6 PASS |
| family:llama | Llama family | 6 PASS |
| family:mistral | Mistral (Llama-family handler) | 2 PASS |
| family:tinyllama | TinyLlama (Llama-family handler) | 2 PASS |
| family:qwen2 | Qwen2 family | 4 PASS |
| family:qwen2.5 | Qwen2.5 (Llama-family handler) | 2 PASS |
| family:phi2 | Phi-2 | 2 PASS |
| family:phi3 | Phi-3 | 5 PASS |
| family:qwen3 | Qwen3 | 2 PASS |
| family:qwen3moe | Qwen3 mixture of experts | 2 PASS |
| script:compare-llama-cpp.sh | Throughput, Juno lanes | 24 DONE |
| script:compare-lora.sh | LoRA training and playback | 2 PASS |
| script:compare-mixed-prefill.sh | Mixed prefill | 2 PASS |
| script:compare-parallel.sh | Static micro-batching under load | 2 DONE |
| script:compare-prefill-batch.sh | Prefill batch size | 2 DONE |
| script:compare-schedule.sh | Static against continuous | 2 PASS |
| script:compare-vision.sh | Vision latency and decode | 2 PASS |
| script:smoke-consistency.sh | Batched equals single requests | 2 PASS |
| script:smoke-context-policy.sh | Context policy | 2 PASS |
| script:smoke-cpu-hot-path.sh | CPU hot path | 4 PASS |
| script:smoke-cpu-kernels.sh | CPU kernels against the previous build | 4 PASS |
| script:smoke-gpu-residency.sh | GPU residency | 1 PASS |
| script:smoke-grammar.sh | Constrained decoding | 1 PASS |
| script:smoke-long-prompt-prefill.sh | Long-prompt prefill | 4 PASS |
| script:smoke-packed-kquant-matmul.sh | Packed K-quant matmul | 2 PASS |
| script:smoke-tools.sh | Tool calling | 1 PASS |
| feature:grammar | Constrained decoding: response_format, x_juno_grammar, the CLI grammar file | 1 PASS |
| feature:tools | Tool calling, and its refusal on templates without tool support | 1 PASS |
| feature:context-shift | Context shift opt-in, decode-time shift and the limit error | 2 PASS |
| feature:long-prompt | Long prompts, streamed text equal to blocking | 4 PASS |
| feature:static-batching | --parallel micro-batching: batched replies equal single ones | 2 DONE, 2 PASS |
| feature:session-reuse | x_juno_session_id prefix reuse | 2 PASS |
| feature:mixed-prefill | Prefill chunks inside continuous decode steps | 2 PASS |
| feature:prefill-batch | --prefill-batch | 2 DONE |
| feature:gpu-residency | Device-resident decode | 1 PASS |
| feature:packed-kquant | Packed K-quant matmul | 2 PASS |
| feature:cpu-integer-kernels | CPU integer kernels and their vector width | 4 PASS |
| feature:quality | Mean NLL per model | 4 PASS |
| feature:throughput | Prefill and generation throughput, allocation and GC | 24 DONE |
| feature:lora-merge | A trained adapter merged into a GGUF and served | 2 PASS |
| feature:unit-tests | Every module's unit tests and the GPU test groups | 2 PASS |
| feature:kit-environment | The host's description and the calibration that scales estimates | 2 DONE |

## Files

- `manifest.json`, `checkpoint.json`, `host.json`, `SUMMARY.txt`: as returned.
- `probes/`: each probe's report.
