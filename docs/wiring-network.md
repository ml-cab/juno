**On CPU vs GPU node execution — the most flexible approach**

The codebase already has the right abstraction in place: `CudaAvailability.isAvailable()` is checked at startup inside `prepareGpuContext()`, and `ForwardPassHandlerLoader` routes to either `CudaMatVec` (cublasSgemv) or `CpuMatVec` (parallel IntStream) transparently. So the same `player.jar` JAR runs on both instance families without recompilation. That's the flexibility pivot.

The two current scripts hardcode instance types: `g4dn.xlarge` for GPU and `m7i-flex.large` for CPU. The most flexible production pattern is to make instance type a runtime parameter in `launcher.sh` / `juno-infra.sh`, driven by one env var (`JUNO_INSTANCE_TYPE`), and let the node bootstrap detect hardware at startup rather than baking it into the script. Then a single `setup` command works for T4, A10G, or a bare CPU node identically — you switch workloads by changing one variable.

Concretely in the bootstrap `USER_DATA`, the CUDA driver install block in `juno-infra.sh` should be gated: if `lspci | grep -i nvidia` returns nothing, skip driver install and set `JUNO_USE_GPU=false` in `/etc/environment`. The node JVM picks it up via the system property `JUNO_USE_GPU` already read in `ConsoleMain`.

For AMX on `m7i-flex.large` the kernel already enables tile registers via `arch_prctl`; the JVM and GGML-based matVec pick it up automatically — no extra config needed (your `juno-infra-ft.sh` bootstrap comment is correct on this).

**On coordinator placement — central instance vs client-side**

The coordinator must run inside the same VPC as the GPU/CPU nodes. Here is why it cannot be on the client machine:

Every decode step passes activation tensors across gRPC node-to-node hops. At FLOAT16 those are roughly 4 MB per hop for TinyLlama and  32 MB per hop for a 70B model. Over intra-AZ ENA (up to 12.5–25 Gbps) those round-trips are sub-millisecond. Over a residential internet connection (~50 Mbps upstream) the same 32 MB payload takes over five seconds per hop — which means each generated token would take more than ten seconds just on network IO. Even for TinyLlama at 4 MB, you would see 0.6 s per hop on a decent connection, adding latency per token. Beyond latency, AWS charges $0.09/GB for outbound traffic, and streaming activations would accumulate quickly.

There is also a session coherence issue: the coordinator holds the `KVCacheManager` (GPU tier + Caffeine CPU tier) and the `PrefixCache` trie. These are in-process data structures keyed by `sessionId`. If the coordinator runs on the client, the KV blocks written by Node-1's `NodeKVCacheAdapter.flush()` need to travel to the client on every token, then back to the node on the next token — the write-through path designed in session 15 only works efficiently when the coordinator and nodes share a low-latency private network.

The one exception where client-side makes sense is the `--local` mode, where `ConsoleMain` forks everything in-process. In that mode there are no network hops at all, so running it on the client machine is exactly right.

**Concrete recommendations for the scripts**

The coordinator should be a 4th instance inside the same VPC, in the same AZ as the compute nodes. A `t3.medium` (2 vCPU, 4 GB, $0.038/hr) is sufficient: the coordinator runs the `GenerationLoop`, `RequestScheduler`, Javalin REST server, and the `KVCacheManager` CPU tier, but does no matrix math at all. This adds only $0.03/hr to the cluster cost while keeping all activation traffic off the public internet.

If budget is tight, Node 1 can dual-host both the coordinator JVM and a node JVM. Node 1 already handles the embedding lookup (the lightest shard), so headroom is available. You would run two JVMs on the same `g4dn.xlarge`: one as `NodeMain` (with `-Xmx8g`) and one as the coordinator (with `-Xmx4g`). The `cluster-config.yaml` seed node entry would point to the loopback address for that local coordinator-to-node gRPC call.

The `juno-infra.sh` bootstrap script should be extended to support a `coordinator` node type: instead of installing CUDA drivers and launching a node JVM, it launches a coordinator JVM using the shade jar's `InferenceApiServer` entry point and registers its private IP in `cluster-config.yaml` for the Hazelcast seed list. The `launcher.sh` can take a `--with-coordinator` flag to spin up `NODE_COUNT+1` instances and configure the coordinator on the extra one.

For the CPU-only free-tier cluster (`juno-infra-ft.sh`), the same pattern applies — one `m7i-flex.large` acts as coordinator and the other two as compute nodes, or the coordinator is co-located on node 1 given the lower RAM pressure of CPU inference.
