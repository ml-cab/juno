# Tier 04: Quantization coverage

Status: not started
Gap analysis refs: §1.1

## Objective

Add the missing quantization formats (Q4_1, Q5_0, Q5_1, and the IQ1-IQ4 importance-quantized
family), extend GPU MMQ kernel coverage to the formats that currently dequantize-and-fall-back
(Q2_K, Q3_K, Q8_0, Q4_0), close the CUDA/ROCm fused-kernel gap for the formats ROCm currently lacks
entirely, and add a `quantize` CLI command so Juno can produce a quantized GGUF from an F32/F16
checkpoint instead of only loading pre-quantized files.

## Why this tier, why now

Two real files already on disk (`Devstral-Small...IQ1_S.gguf`, `minimax-m2.5-tiny...iq4_nl-imat
.gguf`) cannot load at all today because of this gap — Tier 00 already had to document that as an
expected failure. This tier is what actually makes those files usable. It's sequenced after the KV
cache tier because Tier 03's Q4_0 KV codec work and this tier's Q4_0 *weight* dequant work touch
adjacent code and should share a design pass rather than each rediscovering the K-quant layout
conventions independently. It's before sampling/grammar (Tier 05) and speculative decoding (Tier
06) because those don't depend on quant coverage, but MoE architecture breadth (Tier 08) does
depend on IQ4_NL (`minimax-m2.5`'s quant) landing here first.

## Scope

### In scope

1. Implement Q4_1, Q5_0, Q5_1 dequantization (CPU) — these are simpler, non-K-quant legacy formats,
   good validation targets before the harder IQ family.
2. Implement the IQ1_S, IQ2_*, IQ3_*, IQ4_NL/IQ4_XS formats (CPU dequant first; GPU MMQ kernels for
   at minimum IQ4_NL, since that's what unblocks `minimax-m2.5-tiny` on disk today).
3. Extend `Q4KMmqKernel`-style fused GPU kernels to Q2_K, Q3_K, Q8_0, and Q4_0 (CUDA), so these
   formats stop losing the VRAM-compression benefit by falling back to dequantize-to-FP32/FP16.
4. Add ROCm fused K-quant kernels for at least the three formats CUDA already has (Q4_K/Q5_K/Q6_K),
   closing the `RocmMatVec`/`DeviceQ4KMatrix` gap noted in the analysis — implemented and
   unit-tested without hardware per the plan's ROCm-testing rule, flagged `NEEDS-AMD-HARDWARE` until
   validated.
5. A `./juno quantize` CLI command: takes an F32/F16 (or higher-precision quantized) GGUF and a
   target format, produces a new GGUF file. Scope the initial format set to whatever this tier has
   implemented dequant *and* a corresponding quantize (encode) path for — quantizing to a format
   Juno can't itself read back would be a bad first release.

### Out of scope

- Any IQ format beyond IQ1_S/IQ2_*/IQ3_*/IQ4_NL/IQ4_XS if new formats are released upstream after
  this plan is written — scope is fixed to the formats named in the gap analysis.
- Calibration-data-driven quantization (importance matrices, i-matrix generation) for the
  `./juno quantize` command's first cut — round-trip correctness on already-known formats first;
  imatrix support is a natural follow-up but not required for this tier's exit criteria.

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | every new format must have a correct CPU dequant path, tested against known-good reference values (the way `GgufReaderTest`'s existing Q6_K golden-value regression test works) |
| 2 | CUDA GPU inference | new MMQ kernels for Q2_K/Q3_K/Q8_0/Q4_0/IQ4_NL |
| 3 | ROCm GPU inference | new fused K-quant kernels, NEEDS-AMD-HARDWARE until validated |
| 4 | Static schedule | quant format choice must not interact badly with batched GEMM paths (`CudaFp16GemmOps`/`sgemmQ4KBatchedGemm`-style batch-size thresholds) — verify each new MMQ kernel has the same batch-size-aware dispatch as the existing Q4_K one |
| 5 | Continuous schedule | same, for `ContinuousBatchEngine`'s mixed prefill/decode batch shapes |
| 6 | Single-node local mode | primary dev surface, using the two real IQ-quantized files already on disk |
| 7 | Pipeline-parallel cluster | shard loading must correctly propagate the new quant formats' metadata to every node |
| 8 | Tensor-parallel cluster | same |
| 9 | LoRA training | training's frozen-weight dequant path must support the new formats too (a model quantized in a new format should be trainable, not just inferable) |
| 10 | LoRA playback | packed-Q4/新-format weights + LoRA delta composition — confirm `LoraMmqPolicy`'s playback-only MMQ gate extends cleanly or is explicitly scoped out per format |
| 11 | Vision | `VisionEncoder` reuses `MatVec`; confirm new formats work for `mmproj` GGUF weights too, not just the main LLM weights, if any vision model in the wild ships with these quant types |
| 12 | OpenAI REST surface | N/A directly — this is a loading/compute concern, not an API concern; verify indirectly via successful chat completions against the newly-loadable files |
| 13 | Native REST surface | same |
| 14 | CLI | `./juno gguf-info` must correctly describe the new formats; new `./juno quantize` command with its own `--help` |

## Implementation steps

1. Write golden-value regression tests for each new format's dequant math before implementing it
   (mirroring the existing Q6_K bug-fix test pattern) — use published reference dequant formulas,
   cross-checked against small hand-computed examples.
2. Implement Q4_1/Q5_0/Q5_1 CPU dequant; validate; then GPU MMQ kernels for the currently-fallback
   formats (Q2_K/Q3_K/Q8_0/Q4_0), in that order (simplest first).
3. Implement the IQ family CPU dequant, prioritizing IQ4_NL (unblocks `minimax-m2.5-tiny`) and
   IQ1_S (unblocks `Devstral-Small`), then the remaining IQ2_*/IQ3_*/IQ4_XS variants.
4. Implement ROCm fused K-quant kernels for Q4_K/Q5_K/Q6_K (parity with existing CUDA coverage).
5. Build `./juno quantize`, scoped to formats with both a working decode and a new encode path.
6. Confirm the two previously-unloadable real files (`Devstral-Small...IQ1_S`,
   `minimax-m2.5-tiny...iq4_nl-imat`) now load successfully via `./juno gguf-info` and
   `./juno local` (architecture-string handling for these two is still Tier 08's job — this tier
   only needs the *quantization* to stop being the blocker; if the architecture-string guard from
   Tier 00 rejects them for architecture reasons, that's expected and correct, not a Tier 04 defect).

## Tests to write/upgrade before implementation

- **`GgufReaderTest`**: one golden-value test per new format (Q4_1, Q5_0, Q5_1, IQ1_S, IQ2_*,
  IQ3_*, IQ4_NL, IQ4_XS), following the existing Q6_K regression-test pattern.
- **New MMQ kernel tests**: per format, correctness against the CPU dequant oracle, at multiple
  batch sizes crossing the existing serial/batched-GEMM thresholds.
- **New `./juno quantize` round-trip test**: quantize a small known model to each newly-supported
  format, reload it, confirm output is close to (not necessarily identical to) the F32/F16 source
  and structurally valid.
- **`ModelLiveRunnerIT`**: add a load-and-generate check for `Devstral-Small` and
  `minimax-m2.5-tiny` (post-quantization-fix; architecture-routing outcome documented per Tier 00's
  audit).
- **New bash smoke script**: `scripts/performance-tests/smoke-tier04-quant-coverage.sh` — runs
  `./juno gguf-info` and `./juno local` against every quant format now supported, plus a
  `./juno quantize` round-trip for each newly-encodable format.
- **Perf gate (required)**: new MMQ kernels are hot-path changes — `compare-lora.sh` plus a
  per-format microbenchmark; publish under `docs/perf-compare/`.

## Models needed

`Devstral-Small-2-24B-Instruct-2512-UD-IQ1_S.gguf` and
`minimax-m2.5-tiny-24e-iq4_nl-imat.gguf` (already present) cover IQ1_S and IQ4_NL. Per
[`INVENTORY.md`](INVENTORY.md), no file on disk exercises Q4_1/Q5_0/Q5_1 or the remaining IQ
variants (IQ2_*/IQ3_*/IQ4_XS) — golden-value unit tests can validate the math without a full model,
but if end-to-end `ModelLiveRunnerIT` coverage for these specific formats is wanted, flag to the
user for a small model download in that format at the point this tier starts.

## Exit criteria

- [ ] Q4_1, Q5_0, Q5_1 dequant implemented and golden-value tested.
- [ ] IQ1_S, IQ2_*, IQ3_*, IQ4_NL, IQ4_XS dequant implemented and golden-value tested.
- [ ] `Devstral-Small` and `minimax-m2.5-tiny` load successfully (quantization is no longer the
      blocker for either — architecture routing outcome documented separately per Tier 00).
- [ ] CUDA MMQ kernels extended to Q2_K, Q3_K, Q8_0, Q4_0, IQ4_NL at minimum.
- [ ] ROCm fused K-quant kernels implemented for Q4_K/Q5_K/Q6_K, unit-tested, marked
      NEEDS-AMD-HARDWARE.
- [ ] `./juno quantize` ships for at least the formats with both decode and encode support.
- [ ] Cross-surface checklist fully resolved.
- [ ] Perf gate published, no unexplained regression.
- [ ] Docs (`docs/howto.md` new `quantize` command docs, `docs/agent-arch.txt`) updated.
- [ ] `CHANGELOG.md` entry added.
