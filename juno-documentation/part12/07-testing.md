(ch-12-7)=
# 12.7. Testing

## Running the vision module's tests

```bash
mvn test -pl vision
```

No model file, no GPU, and no network access are required: every test in the module runs
against small synthetic fixtures (constructed pixel tensors, small GGUF-shaped tensor arrays, or
the shared `StubForwardPassHandler` deterministic fake) rather than a real downloaded checkpoint.

The `vision` module is not currently included in the combined multi-module unit test command
listed in [Chapter 8.1](#ch-8-1) (`mvn test -pl tokenizer,lora,node,coordinator,sampler,kvcache,
health,registry,juno-player`); run it separately with the command above, or add `vision` to that
`-pl` list when running the full suite locally.

## What the test suite covers

| Test class | Covers |
|---|---|
| `VisionConfigTest` | GGUF metadata to encoder-shape resolution |
| `VisionConfigNormalizationTest` | CLS-token-based CLIP-vs-SigLIP normalisation default selection, and explicit-metadata override taking priority; see [Chapter 12.5](#ch-12-5) |
| `VisionModelPathsTest` | Two-file vs embedded-GGUF file resolution logic |
| `ImagePatchEmbedderTest` | Raw bytes to CHW pixel tensor, including the EXIF orientation parser and transform |
| `VisionEncoderTest` | CLIP/SigLIP forward pass: FFN orientation resolution, projector output-dimension resolution, post-encoder LayerNorm gating (`hasPostLn`), `quickGelu()` behavior |
| `VisionAwareForwardPassHandlerTest` | Single-token `forward()` path: image-token substitution, text-token `embedToken()` delegation |
| `VisionAwareForwardPassHandlerBatchTest` | Batched `forwardBatch()` path: pure-image window, text-only window, mixed window, non-first-node passthrough, no-embeddings passthrough, correct `windowSize` on the result |
| `LlavaHandlerFactoryEmbeddedVisionTest` | `LlamafileGgufIndex`-based embedded-GGUF detection (moondream2-style llamafiles) |

`Phi2RopeTest` (in the `node` module, not `vision`) covers the split-half RoPE fix described in
[Chapter 12.5](#ch-12-5), since `Phi2TransformerHandler` lives in `node` alongside the other
transformer handlers, not in the vision module itself.

## Real-model verification

The vision module's own unit tests deliberately avoid real GGUF files, so they cannot catch
metadata- or shape-specific surprises particular to one real mmproj export, exactly the class of
bug traced in [Chapter 12.5](#ch-12-5) (unreliable `general.architecture`, unreliable
`clip.vision.projection_dim`, inconsistent `ffn_up`/`ffn_down` naming). Two tools close that gap
when a real model file is available:

- `./juno gguf-info /path/to/model.gguf /path/to/mmproj.gguf`, dumps the file's actual metadata
  and tensor layout as plain text, for architecture review without guessing. See
  [Chapter 3.8](#ch-3-8) for the full diagnostics reference.
- A live `curl` request against a running `--local --mmproj-path ...` instance, ideally captured
  with `--jfr` so a hang or a slow response can be traced the same way described at the end of
  [Chapter 12.5](#ch-12-5).

`./juno test`, the general 8-check smoke suite described in [Chapter 3.7](#ch-3-7), does not
currently include a vision-specific check; verifying a real vision model end to end today means
the manual `curl` workflow above.

## See also

- [Chapter 8.1 -- Build and Test](#ch-8-1)
- [Chapter 3.7 -- Test Mode](#ch-3-7)
- [Chapter 3.8 -- Diagnostics and Tracing](#ch-3-8)
- [Chapter 12.5 -- Known Issues and Fixes](#ch-12-5)

---

[<- 12.6 Performance](#ch-12-6) &nbsp;|&nbsp; [Table of Contents](../index.md)