# Host KV footprint: one 2,048-token request on Phi-3.5-mini, before and after real FP16 storage and exact growth

**Purpose.** The KV-cache tier's item 6 (real FP16 host KV storage, growth toward the request's length). Its bars, as
restated by owner decision 10 (a) of 2026-11-18: the host KV one 2,048-token request holds **<= 0.35x** the pre-tier
build's, and Phi-3.5-mini's live heap at that request **<= 0.75x** the pre-tier reading. The same readings showed the
original live-heap bar (0.60x) out of reach of item 6's levers, which is what the decision answered.

**Not pinned.** These are memory readings, not throughput: the live heap after full collections and the host KV
tensors' bytes do not move with clocks. Repeated runs agree to within 100 bytes.

**Setup.** Phi-3.5-mini-instruct-Q4_K_M.gguf, the handler on CUDA (GTX 1080) with GPU attention, the first 2,048 tokens
of `model-quality-nll.phi35.ids` prefilled in one window through `forwardBatch`, then 64 tokens decoded greedily one at
a time. The live heap is `MemoryMXBean` heap use after two `System.gc()` calls (full collections under G1), read after
load and with the request still held. JVM `-Xms12g -Xmx12g -XX:+UseG1GC` (4 MiB regions; the first run of each
tool used `-Xmx12g` alone, with identical readings), JDK 25.0.4.1.

| Build | Tool | Told the length | Host KV of the request | Live heap after load | Live heap with the request |
|---|---|---|---|---|---|
| pre-tier, `e16748ad` (kit jar `73b212a1412661b7`) | `heap-probe/HeapProbeMain.java` | no (the build has no hint) | 3,221,225,472 B: 64 float32 arrays of 4,096 positions x 3,072 | 3,446 MiB | **7,103 MiB** |
| `c607e926`, before item 6 | the same | no | as above | 3,446 MiB | 7,103 MiB |
| this tree, `c607e926` + item 6 (kit jar `c6a1f07b024be5ad`) | `heap-probe/HeapProbeMain.java` | no | 1,610,612,736 B (f16 halves, doubled to 4,096) | 3,446 MiB | 5,567 MiB |
| this tree | `KvFootprintRunMain` (kit) | yes, 2,112 positions | **830,472,192 B** = 2,112 x 393,216 | 3,446 MiB | **4,543.5 MiB** |

| Bar | Reading | Result |
|---|---|---|
| Host KV <= 0.35x the pre-tier build's | 830,472,192 / 3,221,225,472 = **0.258x** | met |
| Live heap <= 0.75x the pre-tier reading | 4,543.5 / 7,103 = **0.640x** | met |
| (restated away) live heap <= 0.60x | 0.640x | out of reach of item 6's levers, see below |

**How to read it.**
- *Where the pre-tier live set goes* (class histogram with the request held): `float[]` 4,086 MB, of which 3,221 MB is
  the KV (two sides x 32 layers, each array 48 MiB plus a header, 13 G1 regions), 788.8 MB the float32 host copies of
  the token embedding and the output projection (32,064 x 3,072 each, present after load), and about 76 MB window
  buffers; `byte[]` 2,259 MB, the quantized host weights. G1 counts each KV array in whole regions: 3,328 MiB in all.
- *Each lever alone*: real FP16 storage without the length (doubling kept) leaves 5,567 MiB, 0.784x, over the 0.75x
  bar; both levers give 0.640x. The 0.60x bar was derived from Tier 01C's figures (KV plus 2.2 GB of weights), which left
  out the 0.79 GB of float32 embedding and output copies and the region rounding; those are weights, not KV.
- *Unpinned prefill time, not a reading*: the request took 5.2 s on this tree against 7.1 to 7.4 s before; the host
  KV now takes the device's FP16 rows as they are (half the bytes over the bus, no float copy). A single unpinned run.

**Commands** (from the repository root; `<kit jar>` is the build's `juno-kit/target/juno-kit-0.1.2-shaded.jar`):

```bash
# the pre-tier and before-item-6 readings, and the "not told" reading on this tree
javac --add-modules jdk.incubator.vector -cp <kit jar> -d classes docs/perf-compare/20261118T220603Z-host-kv-footprint/heap-probe/HeapProbeMain.java
java -Xms12g -Xmx12g -XX:+UseG1GC --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -cp classes:<kit jar> cab.ml.juno.kit.HeapProbeMain models/Phi-3.5-mini-instruct-Q4_K_M.gguf gpu 2048 64 \
  node/src/test/resources/cab/ml/juno/node/model-quality-nll.phi35.ids 3
# this tree, told the request's length (what the server does)
java -Xms12g -Xmx12g -XX:+UseG1GC --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -cp <kit jar> cab.ml.juno.kit.KvFootprintRunMain models/Phi-3.5-mini-instruct-Q4_K_M.gguf 2048 64 true
```

The kit runs the second command as `gpu.kv-footprint.<model>` on each sweep model. `runs.txt` has every raw line.
