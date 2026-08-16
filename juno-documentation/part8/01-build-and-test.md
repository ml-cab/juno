(ch-8-1)=
# 8.1. Build and Test

Requires JDK 25+ and Maven 3.9+.

```bash
mvn clean package -DskipTests          # build: juno-player emits thin jar + *-shaded.jar runnable

mvn test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,juno-player
                                       # unit tests: no model file, no GPU needed

mvn verify -pl juno-master             # integration tests: forks 3 JVM nodes (stub mode)
                                       # includes ThreeNodeClusterIT and TensorParallelClusterIT

mvn verify -pl juno-master -Pintegration -Dmodels=/path/to/models
                                       # ModelLiveRunnerIT: requires real model files

./juno test --model-path /path/to/model.gguf   # real-model smoke test (8 checks, exits 0/1)
```

**Windows (Command Prompt):**
```bat
mvn clean package -DskipTests

mvn test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,juno-player

mvn verify -pl juno-master

mvn verify -pl juno-master -Pintegration -Dmodels=C:\models

juno.bat test --model-path models\model.gguf
```

## See also

- [Chapter 8.2 -- GPU Tests](#ch-8-2)
- [Chapter 3.7 -- Test Mode](#ch-3-7)
- [Chapter 4.8 -- Testing Checklist](#ch-4-8)

---

[<- 7.3 Performance Report](#ch-7-3) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [8.2 GPU Tests ->](#ch-8-2)
