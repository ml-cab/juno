(ch-4-6)=
# 4.6. Programmatic API

## Simple facade (`LoraTrainer.open` with basic args)

Same single-shard layout as `./juno lora`. Train from code, then call `save()`:

```java
import java.nio.file.Path;

import cab.ml.juno.player.ChatModelType;
import cab.ml.juno.player.LoraTrainer;

Path model = Path.of("/path/to/model.gguf");
Path adapter = Path.of("/path/to/model.lora");

try (var trainer = LoraTrainer.open(model, adapter, /*rank*/ 8, /*alpha*/ 8f, /*lr*/ 1e-4)) {
    LoraTrainer.TrainUntilResult textResult = trainer.trainRawTextUntil(
            "Some prose to adapt style.", /*lossTarget*/ 1.8f, /*maxIters*/ 50, /*chunkTokens*/ 32);
    String modelKey = ChatModelType.fromPath(model.toString());
    LoraTrainer.TrainUntilResult qaResult = trainer.trainQaPairUntil(
            "What is my favorite color?", "Blue.", modelKey, /*lossTarget*/ 1.2f, /*maxIters*/ 50);
    trainer.save();
}
```

## Config-based facade (preferred)

Use `LoraTrainingConfig` when you need control over targets, accumulation, clipping, or
scheduling:

## Programmatic API

```java
import cab.ml.juno.lora.*;
import cab.ml.juno.node.*;
import cab.ml.juno.player.LoraTrainer;
import cab.ml.juno.player.LoraTrainingConfig;

// Config-based open (preferred: targets, accumulation, clipping, scheduling)
LoraTrainingConfig cfg = LoraTrainingConfig.builder()
    .rank(8).alpha(8f).learningRate(1e-4)
    .targets("qv")
    .gradientAccumulationSteps(4)
    .maxGradNorm(1.0f)
    .chunkTokens(128)
    .maxTrainTokens(0)      // 0 = unlimited
    .lrSchedule("cosine").warmupSteps(20).minLr(1e-5f)
    .loraMode("lora")       // or "dora", "qa-lora"
    .scaling("standard")    // or "rslora"
    .seed(42)
    .build();
try (LoraTrainer trainer = LoraTrainer.open(modelPath, adapterPath, cfg)) {
    trainer.trainQaPairUntil("What is my name?", "Dima", "tinyllama", 1.2f, 50);
    trainer.save();
}

// Multi-pair from a list
List<String[]> pairs = List.of(
    new String[]{"What is my name?", "Dima"},
    new String[]{"Where do I live?",  "Kyiv"}
);
try (LoraTrainer trainer = LoraTrainer.open(modelPath, adapterPath, cfg)) {
    trainer.trainQaPairsUntilResult(pairs, "tinyllama");
    trainer.save();
}

// Low-level: computeGradients + prepare + step
LoraAdapterSet adapters = LoraInitializer.create(llamaCfg, LoraProjection.qv(), 8, 8f, new Random(42));
LoraTrainableHandler handler = LoraTrainableHandler.load(modelPath, ctx, adapters);
adapters.zeroAllGrads();
LoraGradientResult r = handler.computeGradients(tokens);
LoraGradients.prepare(adapters, r.predictionCount(), 1.0f);
LoraAdamOptimizer.defaults(1e-4).step(adapters);
```

## See also

- [Chapter 4.1 -- Concepts](#ch-4-1)
- [Chapter 4.3 -- Training Guide](#ch-4-3)
- [Chapter 4.5 -- Merging Adapters](#ch-4-5)
- [Chapter 1.3 -- Quickstart: JVM Embedding](#ch-1-3)

---

[<- 4.5 Merging Adapters](#ch-4-5) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [4.7 Common Pitfalls ->](#ch-4-7)
