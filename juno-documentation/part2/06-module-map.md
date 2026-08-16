(ch-2-6)=
# 2.6. Module Map

## Modules (overview)

| Module | Role |
|---|---|
| `juno-bom` | Maven BOM, aligned versions for all `cab.ml` artifacts |
| `api` | OpenAPI spec, protobuf/gRPC API |
| `registry` | Shard planning, model registry |
| `coordinator` | Scheduler, generation loop, REST |
| `node` | Transformer handlers, GGUF, GPU matmul (CUDA + ROCm via Panama FFI) |
| `lora` | Adapter tensors, optimizer |
| `tokenizer`, `sampler`, `kvcache`, `health`, `metrics` | Shared infrastructure |
| `juno-player` | CLI REPL and cluster harness |
| `juno-node`, `juno-master` | Shaded deploy jars |

## Module dependencies

```
juno-master (fat jar)
    +-- juno-player
    +-- coordinator
    +-- node
    |     +-- lora
    |     +-- kvcache
    |     +-- tokenizer
    |     +-- sampler
    |     +-- registry
    |     +-- api
    +-- health
    +-- metrics

juno-node (fat jar)
    +-- node
    +-- health
```

All modules share a common parent POM (`cab.ml:juno`) that manages dependency versions,
compiler settings, and plugin configuration.

## See also

- [Chapter 2.1 -- Overview](#ch-2-1)
- [Chapter 2.3 -- Handler Routing](#ch-2-3)

---

[<- 2.5 Key Design Decisions](#ch-2-5) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [3.1 Commands ->](#ch-3-1)
