(ch-4-4)=
# 4.4. Inference with a Trained Adapter

**How `--lora-play` routes through the stack:**

```
ConsoleMain (--lora-play PATH)
    |
    +-- local mode: LoraAdapterSet.load(path)
    |                    +-- ForwardPassHandlerLoader.load(model, ctx, backend, adapters)
    |                              +-- LoraTrainableHandler (inference-only, no optimizer)
    |
    +-- cluster mode: ClusterHarness.withLoraPlay(path)
                           +-- launchNode(): -Djuno.lora.play.path=PATH injected per JVM
                                    +-- EmbeddedNodeServer.loadShard()
                                             +-- LoraAdapterSet.load(Path.of(property))
                                             +-- ForwardPassHandlerLoader.load(..., adapters)
```

Trained adapters are applied in any mode without entering the training REPL.

**`local` mode:**
```bash
./juno local --model-path /path/to/model.gguf --lora-play /path/to/model.lora
```

`--lora-play` uses greedy decoding (`temperature=0`) by default so factual recall is
deterministic. Pass `--temperature F` explicitly for sampled output; at higher temperatures
a nearby base-model continuation may be selected instead of the memorised answer.

**`cluster` mode:**
```bash
./juno --model-path /path/to/model.gguf --lora-play /path/to/model.lora
```

In cluster mode, `ClusterHarness.withLoraPlay(path)` injects `-Djuno.lora.play.path=PATH`
into every forked node JVM. Each node loads the adapter before building its
`ForwardPassHandler`.

**AWS deployed cluster:**
```bash
./launcher.sh juno-deploy.sh setup \
  --lora-play /absolute/path/to/model.lora \
  --model-url https://...
```

See [AWS deployment](#ch-6-2) for the full deployment flow.

## See also

- [Chapter 4.5 -- Merging Adapters](#ch-4-5)
- [Chapter 3.3 -- Local Mode](#ch-3-3)
- [Chapter 6.2 -- AWS Deployment](#ch-6-2)

---

[<- 4.3 Training Guide](#ch-4-3) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [4.5 Merging Adapters ->](#ch-4-5)
