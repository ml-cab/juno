# Distributed inference

Juno splits transformer work across JVM processes connected by gRPC. **Pipeline parallel** assigns contiguous layer ranges per node so activations flow serially and pooled VRAM fits larger models; **tensor parallel** keeps full depth on each node with head or FFN slices and combines partial logits at the coordinator via star AllReduce (constraint: head count divisible by node count).

Use `./juno` with cluster defaults or explicit `--pType pipeline|tensor`; remote deployments pair **juno-master** (coordinator) with **juno-node** workers. Full diagrams, REST vs native routes, and KV wiring live in [arch.md](../arch.md). Command-line flags and smoke tests are in [howto.md](../howto.md).
