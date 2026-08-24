(ch-1-2)=
# 1.2. Quickstart: Local Player

This is the fastest path to a working chat session, including with Hugging Face-origin GGUF
weights.

Build from source:

```bash
git clone https://github.com/ml-cab/juno.git && cd juno
mvn clean package -DskipTests
```

Download a GGUF (replace the URL with your chosen model):

```bash
cd juno/models
wget https://huggingface.co/.../tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

Run the local interactive console:

**Linux / macOS:**

```bash
./juno local --model-path models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

**Windows:**

```bat
juno.bat local --model-path models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

`--model-path` is relative to the Juno project directory. To run the REST API alongside the
REPL, add `--api-port 8080`.

**Training:** `./juno lora --model-path ...` on Linux/macOS, `juno.bat lora --model-path ...` on
Windows. See [LoRA fine-tuning](#ch-4-1).

**Merging:** `./juno merge` (or `juno.bat merge` on Windows) bakes a trained `.lora` adapter
into a new GGUF, so inference needs no sidecar adapter. See [Merge mode](#ch-3-6).

## See also

- [Chapter 3.3 -- Local Mode](#ch-3-3)
- [Chapter 4.1 -- Concepts](#ch-4-1)
- [Chapter 1.4 -- Supported Models](#ch-1-4)

---

[<- 1.1 Requirements](#ch-1-1) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [1.3 Quickstart: JVM Embedding ->](#ch-1-3)
