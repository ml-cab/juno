Here is the complete reference, organized by purpose.

No-argument usage summary

```
./run.sh        # prints all commands, detected OS, and jar paths
```

cluster — 3-node distributed cluster + REPL (forked JVM nodes)
For GPU deployments and pipeline-parallel scenarios. Each node is a separate JVM process.

```
# Minimal
./run.sh cluster --model-path /path/to/model.gguf

# Via env var
MODEL_PATH=/path/to/model.gguf ./run.sh cluster

# Activation dtype
./run.sh cluster --model-path /path/to/model.gguf --float16      # default
./run.sh cluster --model-path /path/to/model.gguf --float32      # lossless debug
./run.sh cluster --model-path /path/to/model.gguf --int8         # max compression
./run.sh cluster --model-path /path/to/model.gguf --dtype FLOAT16

# Generation params
./run.sh cluster --model-path /path/to/model.gguf --max-tokens 512
./run.sh cluster --model-path /path/to/model.gguf --temperature 0.3

# Bigger heap (needed for 7B+ models)
./run.sh cluster --model-path /path/to/model.gguf --heap 8g

# Show gRPC and node logs
./run.sh cluster --model-path /path/to/model.gguf --verbose
./run.sh cluster --model-path /path/to/model.gguf -v

# Everything combined
./run.sh cluster --model-path /path/to/model.gguf --dtype FLOAT16 --max-tokens 512 --temperature 0.5 --heap 8g -v

# All via env vars
MODEL_PATH=/path/to/model.gguf DTYPE=FLOAT16 MAX_TOKENS=512 TEMPERATURE=0.5 HEAP=8g ./run.sh cluster

# Custom Java installation
JAVA_HOME=/opt/jdk-25 ./run.sh cluster --model-path /path/to/model.gguf

./run.sh cluster --help
```

console — single-JVM in-process REPL (fastest startup, everyday use)
All transformer shards run inside one JVM. No forking, no gRPC sockets.

```
# Minimal
./run.sh console --model-path /path/to/model.gguf

# Via env var
MODEL_PATH=/path/to/model.gguf ./run.sh console

# Dtype, generation, heap — same flags as cluster
./run.sh console --model-path /path/to/model.gguf --float32
./run.sh console --model-path /path/to/model.gguf --max-tokens 512 --temperature 0.1
./run.sh console --model-path /path/to/model.gguf --heap 8g

# Control how many in-process shards (default 3)
./run.sh console --model-path /path/to/model.gguf --nodes 1   # single shard
./run.sh console --model-path /path/to/model.gguf --nodes 6   # more shards

# Verbose
./run.sh console --model-path /path/to/model.gguf -v

# Everything combined
./run.sh console --model-path /path/to/model.gguf --dtype FLOAT16 --max-tokens 512 --temperature 0.3 --nodes 3 --heap 8g -v

# All via env vars
MODEL_PATH=/path/to/model.gguf DTYPE=FLOAT32 MAX_TOKENS=200 NODES=3 HEAP=4g ./run.sh console

./run.sh console --help
```

live — ModelLiveRunner (6 automated smoke checks, exits 0/1)

```
# Model as flag
./run.sh live --model-path /path/to/model.gguf

# Model as env var
MODEL_PATH=/path/to/model.gguf ./run.sh live

# Model as positional arg
./run.sh live /path/to/model.gguf

# Bigger heap
./run.sh live --model-path /path/to/model.gguf --heap 8g
./run.sh live /path/to/model.gguf --heap 8g
MODEL_PATH=/path/to/model.gguf HEAP=8g ./run.sh live

./run.sh live --help
```


Interactive cluster hyper.sh (ConsoleMain REPL)

```
# Minimal — stub mode, no model, cluster plumbing only
MODEL_PATH=/path/to/model.gguf ./hyper.sh cluster

# FLOAT16 is the default activation dtype — this is identical to the above
MODEL_PATH=/path/to/model.gguf ./hyper.sh cluster --float16
MODEL_PATH=/path/to/model.gguf ./hyper.sh cluster --fp16
MODEL_PATH=/path/to/model.gguf ./hyper.sh cluster --dtype FLOAT16

# FLOAT32 — lossless, for debugging / reference runs
MODEL_PATH=/path/to/model.gguf ./hyper.sh cluster --float32
MODEL_PATH=/path/to/model.gguf ./hyper.sh cluster --dtype FLOAT32

# INT8 — maximum compression, ~1% error
MODEL_PATH=/path/to/model.gguf ./hyper.sh cluster --int8
MODEL_PATH=/path/to/model.gguf ./hyper.sh cluster --dtype INT8

# Override generation params
MODEL_PATH=/path/to/model.gguf ./hyper.sh cluster --max-tokens 512
MODEL_PATH=/path/to/model.gguf ./hyper.sh cluster --temperature 0.3
MODEL_PATH=/path/to/model.gguf ./hyper.sh cluster --max-tokens 512 --temperature 0.3

# Override heap (needed for 7B+ models)
MODEL_PATH=/path/to/model.gguf ./hyper.sh cluster --heap 8g

# Skip recompile when source hasn't changed (~10s saved)
MODEL_PATH=/path/to/model.gguf ./hyper.sh cluster -B
MODEL_PATH=/path/to/model.gguf ./hyper.sh cluster --skip-build

# Show gRPC and Maven logs (verbose)
MODEL_PATH=/path/to/model.gguf ./hyper.sh cluster --verbose
MODEL_PATH=/path/to/model.gguf ./hyper.sh cluster -v

# Combine everything
MODEL_PATH=/path/to/model.gguf ./hyper.sh cluster --dtype FLOAT16 --max-tokens 512 --temperature 0.5 --heap 8g -B -v

# Same flags via environment variables
DTYPE=FLOAT16 MAX_TOKENS=512 TEMPERATURE=0.5 HEAP=8g MODEL_PATH=/path/to/model.gguf ./hyper.sh cluster

# Show flag reference
./hyper.sh cluster --help
```

Real-model smoke test (ModelLiveRunner — 6 checks, exit 0/1)

```
# Model path as env var
MODEL_PATH=/path/to/model.gguf ./hyper.sh live

# Model path as positional arg
./hyper.sh live /path/to/model.gguf

# Override heap
MODEL_PATH=/path/to/model.gguf ./hyper.sh live --heap 8g

# Skip recompile
MODEL_PATH=/path/to/model.gguf ./hyper.sh live -B
MODEL_PATH=/path/to/model.gguf ./hyper.sh live --skip-build

# Combine
./hyper.sh live /path/to/model.gguf --heap 8g -B

# Show flag reference
./hyper.sh live --help
```

Unit tests

```
# All modules, all unit tests, no integration tests (~10s)
./hyper.sh test

# One module at a time
./hyper.sh test-module api
./hyper.sh test-module registry
./hyper.sh test-module coordinator
./hyper.sh test-module node
./hyper.sh test-module kvcache
./hyper.sh test-module tokenizer
./hyper.sh test-module sampler
./hyper.sh test-module health
./hyper.sh test-module player

# Fault tolerance tests only (FaultTolerantPipelineTest, HealthReactorTest, RetryPolicyTest)
./hyper.sh test-fault
```

Integration tests (JUnit, no model needed)

```
# Full suite — forks 3 real JVM node processes (~30s)
# Runs InProcessClusterIT + ThreeNodeClusterIT
./hyper.sh integration

# Fast in-process only — zero network, ~250ms
./hyper.sh integration-fast
```

Build / clean / full verify

```
# Compile all modules, no tests
./hyper.sh build

# Remove all target/ directories
./hyper.sh clean

# Full: compile + all unit tests + all integration tests
./hyper.sh verify
```

Reference / demo commands

```
# Print fault-tolerance wiring walkthrough
./hyper.sh health-demo

# Print example curl commands for the REST API
./hyper.sh curl-demo

# Auto-rerun tests on file change (requires fswatch)
./hyper.sh watch                    # defaults to coordinator module
./hyper.sh watch node
./hyper.sh watch tokenizer
```

Global environment overrides (apply to any command)

```
# Use a non-default Maven installation
MVN=/opt/maven/bin/mvn ./hyper.sh test

# Change the coordinator REST port shown in curl-demo
PORT=9090 ./hyper.sh curl-demo
```

Typical dev session flow

```
# 1. First run — full build and verify everything compiles and all tests pass
./hyper.sh clean && ./hyper.sh verify

# 2. After changing node or tokenizer — fast unit test loop
./hyper.sh test-module node
./hyper.sh test-module tokenizer

# 3. Smoke the cluster stack without a model (stub mode, no MODEL_PATH)
./hyper.sh cluster -B                           # no MODEL_PATH → stubs

# 4. Interactive session with real model
MODEL_PATH=/path/to/TinyLlama.gguf ./hyper.sh cluster -B

# 5. Regression check after changes
MODEL_PATH=/path/to/TinyLlama.gguf ./hyper.sh live -B

# 6. Before committing — full suite
./hyper.sh verify
```