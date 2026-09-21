# Model and hardware inventory

Snapshot date: 2026-09-18. Re-check `models/` and `nvidia-smi`/`rocm-smi` before relying on this —
files get added/removed between sessions and this is not auto-maintained.

## Hardware

| Backend | Available | Notes |
|---|---|---|
| CPU | yes | Intel Xeon E5-1650 v2, 12 threads (per `docs/perf-compare/README.md` baseline host) |
| CUDA | yes | 1x NVIDIA GeForce GTX 1080 (8 GiB) |
| ROCm | **no** | no AMD GPU present in this environment |
| Metal / Vulkan / SYCL | no | not applicable to this host; only relevant if Tier 10 scope is later extended to new backends |

Any tier whose exit criteria include a ROCm row must mark it `FAIL-CLOSED` (code correctly refuses
to run/build the unsupported path) or `NEEDS-AMD-HARDWARE` (implemented, unit-tested without
hardware, needs real-device validation) rather than `PASS` — do not claim ROCm parity without
having run on real AMD hardware.

## Models present in `models/` (2026-09-18)

| File | GGUF `general.architecture` | Family / notes | Used by |
|---|---|---|---|
| `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | (llama-family) | TinyLlama, small/fast, primary CI-speed smoke model | most tiers, default smoke target |
| `tinyllama-1.1b-chat-v1.0.Q2_K.gguf` | (llama-family) | TinyLlama, Q2_K quant coverage | Tier 04 (quant coverage) |
| `tinyllama-1.1b-chat-v1.0.Q4_K_M.lora` | n/a (Juno `.lora`) | trained adapter for the Q4_K_M TinyLlama above | Tier 00, Tier 12 (LoRA) |
| `tinyllama-1.1b-chat-v1.0.Q4_K_M.lora.perf` | n/a | perf-comparison copy of the adapter | perf-compare scripts only |
| `mistral-7b-instruct-v0.1-q4_k_m.gguf` | (llama-family, Mistral) | dense Mistral-7B | most tiers, mid-size dense model |
| `mistral-7b-instruct-v0.2.Q2_K.llamafile` | (llama-family, Mistral) | llamafile container, Q2_K | Tier 04, llamafile-loading coverage |
| `qwen2.5-3b-instruct-q4_k_m.gguf` | (llama-family, Qwen2) | dense Qwen2.5 | most tiers |
| `Qwen3.5-0.8B.Q4_K_M.gguf` | **`qwen35`** | not `qwen3` — **unrecognized by today's architecture switch**, falls through to `LlamaTransformerHandler` | Tier 00 (fail-closed audit), Tier 08 (real handler or documented rejection) |
| `Phi-3.5-mini-instruct-Q4_K_M.gguf` | `phi3` | recognized, dedicated handler | most tiers |
| `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` | **`gemma4`** | **unrecognized**, falls through to `LlamaTransformerHandler` | Tier 00, Tier 08 |
| `Devstral-Small-2-24B-Instruct-2512-UD-IQ1_S.gguf` | **`mistral3`**, quant `IQ1_S` | **unrecognized architecture** + **unsupported quant** (IQ1_S is not implemented at all per gap analysis §1.1) — this file cannot currently load | Tier 00 (confirm it fails closed, not silently), Tier 04 (IQ-series support), Tier 08 |
| `minimax-m2.5-tiny-24e-iq4_nl-imat.gguf` | **`minimax-m2`**, MoE (`expert_count=24`, `expert_used_count=8`), quant `IQ4_NL` | **unrecognized architecture, real MoE model, unsupported quant** — highest-priority real-file example of the silent-degrade risk in gap analysis §1.9/§2.4 | Tier 00 (must confirm fail-closed today), Tier 04 (IQ4_NL), Tier 08 (MoE breadth) |
| `llama-1-30b.Q4_K_M.gguf` | (llama-family) | large dense model, useful for memory-pressure/GPU-layer-offload edge cases | Tier 01, Tier 03 (memory-pressure paths) |
| `Meta-Llama-3.2-1B-Instruct-Q8_0.llamafile` | (llama-family) | llamafile container, Q8_0 | Tier 04, llamafile coverage |
| `moondream2-q5_k.llamafile` | phi2 backbone + SigLIP vision encoder, embedded in llamafile | only vision-capable model on disk | Tier 11 (vision), and any tier claiming vision cross-surface PASS |

## Known architecture strings NOT recognized by Juno's dispatch today

`node/.../ForwardPassHandlerLoader.java`'s switch recognizes exactly: `phi2`, `phi3`, `qwen3`,
`qwen3moe`. Everything else — including three real files on disk (`qwen35`, `gemma4`, `mistral3`)
and one real MoE model (`minimax-m2`) — falls through to `LlamaTransformerHandler`. Tier 00 must
determine, for each of these four, whether that fallback (a) happens to be correct because the
tensor layout is close enough to a supported family, (b) is silently wrong, or (c) already fails
for an unrelated reason (e.g. `Devstral`/`minimax-m2`'s quant types aren't implemented at all, so
those two currently fail at load time regardless of architecture routing). Record the finding per
model in Tier 00's file.

## Gaps: models needed but not present

Flag these to the user when the relevant tier is reached — do not assume substitutes are
equivalent:

| Needed for | What's missing | Why a substitute won't do |
|---|---|---|
| Tier 08 (MoE breadth) | A true Mixtral (8x7B or a smaller Mixtral-shaped model) with `general.architecture=llama` and `ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps` tensors | `minimax-m2.5-tiny` is MoE but a different architecture string/tensor layout; it doesn't exercise the specific "architecture string says dense but tensors say MoE" case that is Mixtral's actual failure mode |
| Tier 08 (MoE breadth) | A working `qwen3moe` GGUF (the currently-supported MoE family) | not present on disk at all — Tier 08 cannot regression-test the *existing* MoE handler without one |
| Tier 08 | A plain `qwen3` (non-`qwen3.5`, non-MoE) GGUF | only `qwen35` is present; can't verify the currently-supported `qwen3` path still works without one, or diff `qwen3` vs `qwen35` handling |
| Tier 11 (vision) | A LLaVA-1.5 and/or LLaVA-1.6 GGUF + its `mmproj` file | only `moondream2` (phi2+SigLIP) is present; `docs/agent-arch.txt`/vision docs claim LLaVA-1.5/1.6 support but nothing on disk exercises it |
| Tier 04 (quant coverage) | A GGUF using Q4_1, Q5_0, or Q5_1 (once implemented, to test against) | none present; `Devstral`/`minimax-m2.5` cover IQ1_S/IQ4_NL once those are implemented, but not the non-K legacy formats |
| Tier 09 (tensor parallelism) | N/A — uses existing dense models, no new file needed | — |
| Tier 10 (ROCm ) | An AMD GPU, physically | see hardware table above |

Ask the user for each of these at the point the relevant tier actually starts, not before — needs
may change as earlier tiers land.
