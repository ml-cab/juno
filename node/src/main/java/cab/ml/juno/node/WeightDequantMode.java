/*
 * Copyright 2026 Dmytro Soloviov (soulaway)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package cab.ml.juno.node;

/**
 * Controls when and where quantized projection weights are dequantized.
 *
 * <p>Selected via {@code --dequant eager|lazy} on all commands that load a
 * real model ({@code cluster}, {@code local}). Propagated through the stack:
 * run.sh/run.bat → ConsoleMain → ClusterHarness → NodeMain →
 * EmbeddedNodeServer → ForwardPassHandlerLoader → LlamaTransformerHandler.
 *
 * <h3>EAGER (default)</h3>
 * All projection weight matrices are dequantized to {@code float32} once at
 * shard load time and uploaded to the GPU as {@link DeviceFloatMatrix}
 * instances ({@link GpuWeightShard}). Every decode step calls
 * {@link CudaMatVec#sgemv(DeviceFloatMatrix, float[])} — no H2D weight copy
 * per token, GPU utilisation is high.
 * <p>
 * Cost: ~2–4x the compressed model size in VRAM (e.g. a 637 MB Q4_K_M model
 * occupies ~2.4 GB VRAM per shard after dequantization to float32).
 * Use when VRAM is sufficient.
 *
 * <h3>LAZY</h3>
 * Weights stay in their original quantized format (Q4_K, Q8_0, etc.) in
 * JVM heap memory. Each matmul dequantizes one block at a time on the CPU
 * using the static {@code matVec(QuantizedTensor, …)} overloads inside
 * {@link LlamaTransformerHandler}. No GPU is used for projection matmuls.
 * <p>
 * Cost: highest CPU usage and highest latency per token; lowest VRAM
 * footprint (only KV cache lives on the GPU). Use when VRAM is the
 * binding constraint and latency is secondary.
 *
 * <h3>Comparison</h3>
 * <pre>
 *  Mode   VRAM per shard          Latency           CPU load
 *  -----  ----------------------  ----------------  --------
 *  EAGER  ~2–4x compressed size   lowest (GPU)      low
 *  LAZY   ~0 (KV cache only)      highest (CPU)     high
 * </pre>
 */
public enum WeightDequantMode {

    /**
     * Dequantize all projection weights once at load time and upload to GPU.
     * Default. Requires sufficient VRAM (roughly 2–4x the compressed model
     * file size per shard).
     */
    EAGER,

    /**
     * Keep weights quantized in JVM heap; dequantize one block at a time on
     * the CPU during inference. Minimal VRAM usage; highest CPU load.
     */
    LAZY;

    /**
     * Parse the value of the {@code JUNO_DEQUANT} system property (or the
     * {@code --dequant} flag string).
     *
     * @param value "eager" or "lazy" (case-insensitive); null / blank / unknown
     *              falls back to {@link #EAGER}
     * @return the matching mode, defaulting to EAGER
     */
    public static WeightDequantMode parse(String value) {
        if (value == null || value.isBlank()) return EAGER;
        return switch (value.strip().toLowerCase()) {
            case "lazy" -> LAZY;
            default     -> EAGER;
        };
    }
}