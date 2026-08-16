(ch-3-1)=
# 3.1. Commands

Unified stand-alone launchers sit at the project root: `./juno` on Linux/macOS,
`juno.bat` on Windows (delegates to `scripts\run.bat`). Requires JDK 25+ and pre-built jars
(`mvn clean package -DskipTests`).

> All examples in this reference use `./juno`. Replace with `juno.bat` on Windows and use
> backslashes for paths (for example `--model-path models\model.gguf`). All flags, environment
> variables, and subcommands are identical across platforms.

| Command | Description |
|---------|-------------|
| `cluster` | 3-node cluster (default command): forked JVMs, real gRPC. Default `--pType pipeline`; use `--pType tensor` for AllReduce mode |
| `local` | In-process REPL: all transformer shards in one JVM, no forking, no gRPC |
| `lora` | LoRA fine-tuning REPL: single in-process JVM, adapter persisted to a `.lora` file |
| `merge` | Bake a trained `.lora` adapter into a new standalone GGUF; no sidecar needed at inference time |
| `test` | 8 automated real-model smoke checks (6 pipeline + 2 tensor); exits 0 if all pass, 1 if any fail |

## See also

- [Chapter 3.2 -- Flags](#ch-3-2)
- [Chapter 3.3 -- Local Mode](#ch-3-3)
- [Chapter 3.4 -- Cluster Mode](#ch-3-4)
- [Chapter 3.5 -- LoRA Mode](#ch-3-5)
- [Chapter 3.6 -- Merge Mode](#ch-3-6)
- [Chapter 3.7 -- Test Mode](#ch-3-7)

---

[<- 2.6 Module Map](#ch-2-6) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [3.2 Flags ->](#ch-3-2)
