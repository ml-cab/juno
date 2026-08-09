(ch-06)=
# 6. JVM Integration: BOM, JunoPlayer, LoraTrainer, and the HTTP Client

This chapter covers embedding Juno directly in JVM code rather than talking to it over REST
(see [Chapter 5](#ch-05)) or driving it from the CLI (see [Chapters 3–4](#ch-03)).

## Maven BOM (`juno-bom`)

Import one POM so every `cab.ml` module shares the same version:

```xml
<dependencyManagement>
  <dependencies>
    <dependency>
      <groupId>cab.ml</groupId>
      <artifactId>juno-bom</artifactId>
      <version>0.1.0</version>
      <type>pom</type>
      <scope>import</scope>
    </dependency>
  </dependencies>
</dependencyManagement>

<dependencies>
  <dependency>
    <groupId>cab.ml</groupId>
    <artifactId>juno-player</artifactId>
    <!-- version comes from juno-bom -->
  </dependency>
</dependencies>
```

## Runnable jar versus library jar

After `mvn package`, `juno-player/target/` contains:

- `juno-player-0.1.0.jar` — normal thin classpath artifact for dependents (compose with
  BOM-managed modules).
- `juno-player-0.1.0-shaded.jar` — fat jar with `Main-Class: cab.ml.juno.player.ConsoleMain`.
  The `./juno` launcher selects this shaded jar when present.

## In-process facade (`JunoPlayer`)

Loads the GGUF, builds an in-process `LocalInferencePipeline`, `GenerationLoop`, and
`RequestScheduler` — the same wiring as `./juno local`:

```java
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.Flow;

import cab.ml.juno.player.JunoPlayer;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;

try (JunoPlayer player = JunoPlayer.builder(Path.of("/path/to/model.gguf"))
        .nodeCount(3)
        .useGpu(true)
        .samplingParams(SamplingParams.defaults().withMaxTokens(128).withTemperature(0.7f))
        .build()) {

    var messages = List.of(ChatMessage.user("Explain JDK virtual threads in one sentence."));
    var result = player.chat(messages);
    System.out.println(result.text());

    Flow.Publisher<String> pieces = player.streamPublisher(messages);
    pieces.subscribe(new Flow.Subscriber<>() {
        Flow.Subscription s;
        public void onSubscribe(Flow.Subscription s) {
            this.s = s;
            s.request(Long.MAX_VALUE);
        }
        public void onNext(String t) {
            System.out.print(t);
        }
        public void onError(Throwable e) {
            e.printStackTrace();
        }
        public void onComplete() {
            System.out.println();
        }
    });

    float[] vec = player.embed(messages); // length = model hidden dim (last RMS hidden before LM head)

    // Optional OpenAI-compatible REST server on port 8080:
    var api = player.startApiServer(8080);
    Runtime.getRuntime().addShutdownHook(Thread.ofVirtual().unstarted(api::stop));
}
```

## Programmatic LoRA (`LoraTrainer`)

Same single-shard layout as `./juno lora`; train from code then `save()`. The REPL semantics,
flags, and pitfalls behind this API are covered in [Chapter 8](#ch-08) and [Chapter 9](#ch-09).

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

## `Flow.Publisher` from `TokenConsumer` (`PublisherTokenConsumer`)

For custom scheduling — not using `JunoPlayer.streamPublisher` — wrap any `RequestScheduler`
submission:

```java
import java.util.List;
import java.util.concurrent.Flow;

import cab.ml.juno.coordinator.InferenceRequest;
import cab.ml.juno.coordinator.PublisherTokenConsumer;
import cab.ml.juno.coordinator.RequestPriority;
import cab.ml.juno.coordinator.RequestScheduler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;

void stream(RequestScheduler scheduler, String modelId, SamplingParams params) {
    InferenceRequest req = InferenceRequest.of(modelId,
            List.of(ChatMessage.user("Hello")), params, RequestPriority.NORMAL);
    PublisherTokenConsumer bridge = new PublisherTokenConsumer();
    Flow.Publisher<String> pub = bridge.publisher();
    scheduler.submit(req, bridge).whenComplete((r, e) -> bridge.finish());
    // subscribe to pub …
}
```

## Java HTTP client (`JunoHttpClient`)

Talk to a sidecar started with `./juno local … --api-port 8080` (or `JunoPlayer.startApiServer`,
see [Chapter 5](#ch-05) for the wire contract):

```java
import java.net.URI;
import java.util.List;
import java.util.concurrent.Flow;

import cab.ml.juno.player.JunoHttpClient;
import cab.ml.juno.tokenizer.ChatMessage;

var http = new JunoHttpClient(URI.create("http://localhost:8080"));

// Native blocking inference (/v1/inference)
String text = http.blockingInference("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        List.of(ChatMessage.user("Ping")), 64);

// Native SSE (/v1/inference/stream) — publisher emits decoded token pieces from JSON events
Flow.Publisher<String> nativeStream = http.streamingInference(null,
        List.of(ChatMessage.user("Stream ping")), 32);

// OpenAI-compatible blocking + SSE (/v1/chat/completions)
String openAiText = http.blockingOpenAiChat("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        List.of(ChatMessage.user("Ping")), 64, 0.7f);
Flow.Publisher<String> openAiSse = http.streamingOpenAiChat("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        List.of(ChatMessage.user("Stream")), 32, 0.7f);
```

For a quick-start using only `LocalChat.java` in a JUnit test — the minimal single-JVM path with
no REST server at all — see the example in the project README's JVM Integration section; the
shape mirrors `JunoPlayer.builder(...)` above with a narrower surface.

---

[← Chapter 5: The OpenAI-Compatible REST API](#ch-05) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [Chapter 7: AWS Deployment →](#ch-07)
