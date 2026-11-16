# Hardware kit: the quantized device KV's prefill probe, first run

**Purpose.** The kit probe the quantized device KV's prefill bar lands with: `gpu.kv-prefill.<model>`, the 2,048-token
prefill with the handler on CUDA at `q8_0` device KV against `f16`, in one 2,048-row window and in 128-row windows, as a
reading (the bar itself is the owner's pinned gate). This is its first run, on the reference host, after the host KV began
taking the device copy's stored rows
([`20261116T032708Z-device-kv-prefill-encoded-rows`](../20261116T032708Z-device-kv-prefill-encoded-rows/INDEX.md)), to show
it runs where the feature works and to measure its cost and device memory for the catalog.

**Pinned:** no (a kit run). **Host:** Xeon E5-1650 v2, 64 GiB, GTX 1080 8 GiB (shared with the desktop); calibration read
NOISY at 1.10x the reference time. **Build:** HEAD `a4a91182` plus the uncommitted quantized device KV prefill work and this
probe, built by the kit launcher (`./juno-kit.sh --help`), then `./juno-kit.sh --skip-build --yes --probes
gpu.kv-prefill.tinyllama-1.1b,gpu.kv-prefill.qwen2.5-3b,gpu.kv-prefill.phi-3.5-mini,gpu.kv-prefill.mistral-7b --models-dir
models`, 2026-11-16T15:00Z. (`--probes` takes exact ids; the previous kit entry's `gpu.kv-quality.*` was shorthand for its
four ids.)

## 1. `gpu.kv-prefill.<model>` (`probes/gpu.kv-prefill.*.json`)

Each attempt runs `KvPrefillRunMain` twice with the handler on CUDA, `f16` then `q8_0` keys and values: the model's
2,048-token quality text prefilled through `forwardBatch` in each width, one warm-up and three measured repetitions; the
ratio is the `q8_0` median over the `f16` median. Before reading, the probe requires both runs to have opened GPU attention
and the device prefill region, and the `q8_0` run's handler to have logged its device copy in the selected formats (every
run here: "GPU attention's KV copy stored as K q8_0, V q8_0"). Every probe read NOISY: the desktop used 5% to 12% of the
CPU, so the kit ran each three times and kept the last (`*.attempt1.json`, `*.attempt2.json` hold the others).

| Model | One window: `f16`, `q8_0`, ratio | 128-row windows: `f16`, `q8_0`, ratio | Ratios of the two earlier attempts | Time, device peak |
|---|---|---|---|---|
| TinyLlama 1.1B | 842.8, 869.1 ms, 1.031 (spread 17%) | 1,115, 1,075 ms, 0.965 | 0.961, 0.978; 0.915, 0.961 | 1:33 (3 attempts), 1,172 MiB |
| Qwen2.5-3B | 2,185, 2,114 ms, 0.968 | 2,819, 2,707 ms, 0.960 | 0.913, 1.004; 0.904, 0.983 | 2:52, 2,730 MiB |
| Phi-3.5-mini | 3,322, 2,960 ms, 0.891 | 4,065, 3,640 ms, 0.896 | 0.874, 0.888; 0.881, 0.894 | 4:03, 3,824 MiB |
| Mistral 7B | 5,031, 4,916 ms, 0.977 | 5,677, 5,528 ms, 0.974 | 0.973, 0.974; 0.937, 0.948 | 5:31, 5,266 MiB |

All 24 ratios (four models, two widths, three attempts) read 0.874x to 1.031x, every one under the proposed 1.10x; the
highest, TinyLlama's last one-window reading, is the one whose `f16` side spread 17% across its three repetitions. The same
session's unpinned A/B (median of five, one JVM per side) read 0.895x to 0.989x.

The probe is shown able to fail by the kit's self-test on the synthetic model: its head width (16) is not a multiple of 32,
so its handler keeps an FP16 device copy, and the probe fails with that warning instead of reading another path's time
(`ProbeSelfTests#gpuKvPrefillProbeFailsWhereTheDeviceCopyIsNotInTheSelectedFormats`, run on this host's CUDA).

## How to read it

The probe runs on every sweep model where the quantized device KV works, and on this shared desktop it reads the quantized
copy at or under the FP16 prefill time within noise. It is a reading, not the bar: the owner's pinned gate
(`dist/device-kv-prefill/run-gate.sh`, A B A B A B per model, five repetitions per JVM) scores decision 8 (a). The catalog
takes the probe's costs from this run, one attempt's time at the reference speed (28, 52, 73 and 100 s), and its device peaks
(1,172, 2,730, 3,824 and 5,266 MiB), so a tester's card that cannot hold a model skips its probe.

Files: `SUMMARY.txt`, `host.json`, `estimate.json`, `probes/*.json` (each probe's report and its earlier attempts,
calibration included).
