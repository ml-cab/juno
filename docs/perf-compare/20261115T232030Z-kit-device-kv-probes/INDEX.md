# Hardware kit: the quantized device KV's probes, first run

**Purpose.** The kit probes the quantized device KV lands with: `gpu.kv-quality.<model>` (mean NLL against an `f16` device
KV per sweep model, and the handler's notice that the device copy holds the selected formats) and
`gpu.consistency.tinyllama`'s decode batch of eight at each KV type (every batched reply equal to its single-request
reply). This is their first run, on the reference host, to show they pass where the feature works and to measure their
cost and device memory for the catalog.

**Pinned:** no (a kit run; nothing here is a throughput reading). **Host:** Xeon E5-1650 v2, 64 GiB, GTX 1080 8 GiB (shared
with the desktop); calibration read NOISY at 1.06x the reference time (other processes on the CPU). **Build:** HEAD
`79fa87b0` plus the uncommitted quantized device KV work and these probes, built by the kit launcher
(`./juno-kit.sh --help`), then `./juno-kit.sh --skip-build --yes --probes gpu.consistency.tinyllama,gpu.kv-quality.*
--models-dir models`, 2026-11-15T23:20Z.

## 1. `gpu.kv-quality.<model>` (`probes/gpu.kv-quality.*.json`)

Each probe runs `QualityRunMain` three times with the handler on CUDA (`-Djuno.kit.quality.gpu=true`): `f16`, `q8_0`, and
`q8_0` keys with `q4_0` values, teacher-forced over the 2,048-token quality text in windows of 128 (the prefill path, so
attention reads the device KV at prefill width).

| Model | NLL `f16` | `q8_0`: NLL ratio, agreement | `q8_0`/`q4_0`: NLL ratio, agreement | Device KV announced (bytes per position, of FP16) | Time, device peak |
|---|---|---|---|---|---|
| TinyLlama 1.1B | 2.45657 | 1.00047, 98.63% | 1.00077, 96.48% | 544 / 416 of 1,024 | 0:28, 992 MiB |
| Qwen2.5-3B | 1.69616 | 1.00112, 99.41% | 1.00398, 99.02% | 544 / 416 of 1,024 | 0:49, 2,488 MiB |
| Phi-3.5-mini | 1.35936 | 1.00099, 99.80% | 1.00358, 99.22% | 6,528 / 4,992 of 12,288 | 2:37, 3,384 MiB |
| Mistral 7B | 1.52942 | 0.99914, 99.22% | 0.99735, 99.02% | 2,176 / 1,664 of 4,096 | 1:04, 4,796 MiB |

Checks: every NLL ratio at most 1.02x, and every quantized run's handler logged "GPU attention's KV copy stored as" the
selected formats: 16 of 16 pass. The agreement column is a reading, not a check: it compares the argmax along the fixed
text's first 512 positions, where the CPU `kv-quality` probe reads the same. It is not the device KV's agreement bar,
which `DeviceKvQuantQualityLiveTest` reads along each model's greedy continuation (99.41% to 100% at `q8_0`, 98.63% to
99.61% at `q8_0`/`q4_0`, [`20261113T194534Z-device-kv-quant`](../20261113T194534Z-device-kv-quant/INDEX.md)); on the human
text TinyLlama's argmax flips more often (98.63%, 96.48%).

The stored check is shown able to fail by the kit's self-test on the synthetic model: its head width (16) is not a
multiple of 32, so its handler keeps an FP16 device copy and warns, and the check fails with that warning while the NLL
check passes (`ProbeSelfTests`, run on this host's CUDA).

## 2. `gpu.consistency.tinyllama` (`probes/gpu.consistency.tinyllama.json`)

Static batching on the GPU as before, then a decode batch of eight (64 tokens, three prompts cycled) at `f16`, `q8_0` and
`q8_0`/`q4_0`: each of the 24 batched replies equals its single-request reply at its type. 44 of 44 checks pass in 0:33
(three servers where there was one), device peak 934 MiB.

## How to read it

The probes pass on every sweep model where the quantized device KV works, and the stored check fails where a handler
keeps an FP16 copy. The catalog takes their costs from this run (26, 46, 148 and 60 s at the reference speed; the
consistency probe 31 s) and their device peaks (992, 2,488, 3,384 and 4,796 MiB), so a tester's card that cannot hold a
model skips its probe.

Files: `SUMMARY.txt`, `host.json`, `estimate.json`, `probes/*.json` (each probe's report, calibration included).
