(ch-3-6)=
# 3.6. Merge Mode

`./juno merge` bakes a trained LoRA adapter into a standalone GGUF. It writes a new GGUF where
the LoRA-patched projection tensors (`wq`/`wv` on every layer) are stored as F32 for full
precision. All other tensors are copied verbatim in their original quantized encoding. The
resulting file loads with `./juno local` or `./juno` like any other model, with no sidecar
adapter needed.

```bash
# Default: reads <model>.lora, writes <model>-merged.gguf
./juno merge --model-path /path/to/TinyLlama.Q4_K_M.gguf

# Explicit paths
./juno merge --model-path /path/to/model.gguf \
             --lora-path  /adapters/my.lora   \
             --output     /path/to/merged.gguf

# Larger heap for big models (rule of thumb: 2x model file size)
./juno merge --model-path /path/to/Mistral-7B.gguf --heap 12g
```

**Windows (Command Prompt):**

```bat
juno.bat merge --model-path models\TinyLlama.Q4_K_M.gguf

juno.bat merge --model-path models\model.gguf ^
               --lora-path adapters\my.lora ^
               --output merged\merged.gguf

juno.bat merge --model-path models\Mistral-7B.gguf --heap 12g
```

The LoRA delta per element (roughly 6e-4) is smaller than Q4_K quantization noise (roughly
3e-3). Re-quantizing the merged weights back to Q4_K would erase the training entirely. F32
storage for the 44 patched tensors is the correct trade-off. For TinyLlama 1.1B Q4_K_M
(667 MB), the merged file is approximately 1 GB.

Before redistributing a merged GGUF, review the licensing implications: see
[LoRA and merge licensing](#ch-9-3).

## See also

- [Chapter 3.2 -- Flags](#ch-3-2)
- [Chapter 4.5 -- Merging Adapters](#ch-4-5)
- [Chapter 9.3 -- LoRA and Merge Licensing](#ch-9-3)

---

[<- 3.5 LoRA Mode](#ch-3-5) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [3.7 Test Mode ->](#ch-3-7)
