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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Instant;
import java.util.Optional;
import java.util.logging.Logger;

import cab.ml.juno.kvcache.KVBlock;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.kvcache.KVKey;

/**
 * Bridges the transformer handler's in-process {@code float[][]} KV arrays and
 * the {@link KVCacheManager} (GPU + CPU tiers with LRU eviction).
 *
 * <p><b>Write-through</b>: after each token position is written to the
 * handler's local KV arrays, {@link #flush} serialises the updated K and V
 * data into a {@link KVBlock} and stores it in the manager. This keeps the
 * GPU-tier budget accounting accurate and allows the manager to evict under
 * real memory pressure.
 *
 * <p><b>Restore</b>: if a local {@code float[][]} entry was removed from
 * the handler's in-process map (e.g. under JVM heap pressure), calling
 * {@link #tryRestore} will rebuild it from whichever tier still holds the
 * block, transparently promoting it back to GPU if found only in CPU tier.
 *
 * <p><b>Evict</b>: {@link #evict} removes entries from both the local
 * in-process map and the manager's GPU/CPU tiers. It is the single eviction
 * call-site for completed requests.
 *
 * <h3>KVBlock serialisation format</h3>
 * {@code data} = float32 little-endian, concatenated:
 * <pre>
 *   bytes [0 .. seqLen*kvDim*4)         K values for positions 0..seqLen-1
 *   bytes [seqLen*kvDim*4 .. 2*above)   V values for positions 0..seqLen-1
 * </pre>
 *
 * <h3>Thread safety</h3>
 * Each requestId has independent KV state. {@link KVCacheManager} is internally
 * thread-safe; concurrent flushes for different requestIds are safe.
 */
public final class NodeKVCacheAdapter {

    private static final Logger log = Logger.getLogger(NodeKVCacheAdapter.class.getName());

    private final KVCacheManager manager;

    /**
     * @param manager the cluster-level cache manager for this node's layer range
     */
    public NodeKVCacheAdapter(KVCacheManager manager) {
        if (manager == null)
            throw new IllegalArgumentException("manager must not be null");
        this.manager = manager;
    }

    // ── Write-through ─────────────────────────────────────────────────────────

    /**
     * Serialise the current K and V arrays and store them in the
     * {@link KVCacheManager}. Called after each new token position is written to
     * the handler's local KV arrays.
     *
     * <p>This is a write-through operation: an existing block for the same key is
     * replaced with the updated (longer) block. GpuKVCache byte-budget eviction
     * fires here if VRAM is exhausted, so older requests will be demoted to the
     * CPU tier automatically.
     *
     * @param requestId          request or session identifier
     * @param absoluteLayerIndex absolute layer index ({@code startLayer + localLi})
     * @param kData              K array for this layer; first {@code seqLen * kvDim}
     *                           elements are valid
     * @param vData              V array for this layer; first {@code seqLen * kvDim}
     *                           elements are valid
     * @param seqLen             number of token positions written so far (1-based)
     * @param kvDim              key/value dimension per position
     */
    public void flush(String requestId, int absoluteLayerIndex,
                      float[] kData, float[] vData,
                      int seqLen, int kvDim) {
        int floatsPerSeq = seqLen * kvDim;
        int bytesPerSeq  = floatsPerSeq * Float.BYTES;
        byte[] data = new byte[bytesPerSeq * 2];

        ByteBuffer bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < floatsPerSeq; i++) bb.putFloat(kData[i]);
        for (int i = 0; i < floatsPerSeq; i++) bb.putFloat(vData[i]);

        KVKey   key = new KVKey(requestId, absoluteLayerIndex);
        Instant now = Instant.now();
        KVBlock blk = new KVBlock(key, data, seqLen, absoluteLayerIndex, now, now);
        manager.put(key, blk);
    }

    // ── Restore ───────────────────────────────────────────────────────────────

    /**
     * Attempt to restore K and V arrays from the {@link KVCacheManager} for
     * the given request and layer. Returns {@link Optional#empty()} when no block
     * is cached — i.e., this is the first forward pass for that request.
     *
     * <p>A GPU-tier hit returns immediately. A CPU-tier hit promotes the block back
     * to GPU (handled transparently by {@link KVCacheManager#get}).
     *
     * @param requestId          request or session identifier
     * @param absoluteLayerIndex absolute layer index
     * @param kvDim              key/value dimension per position
     * @return restored {@link KvPair} or empty
     */
    public Optional<KvPair> tryRestore(String requestId, int absoluteLayerIndex, int kvDim) {
        KVKey key = new KVKey(requestId, absoluteLayerIndex);
        return manager.get(key).map(blk -> {
            int floatsPerSeq = blk.sequenceLen() * kvDim;
            float[] k = new float[floatsPerSeq];
            float[] v = new float[floatsPerSeq];
            // Note: do NOT call asReadOnlyBuffer() here — it silently drops the
            // byte order on HeapByteBuffer in some JVM builds, producing garbage.
            ByteBuffer bb = ByteBuffer.wrap(blk.data()).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < floatsPerSeq; i++) k[i] = bb.getFloat();
            for (int i = 0; i < floatsPerSeq; i++) v[i] = bb.getFloat();
            log.fine("KV restored from manager: requestId=" + requestId + " layer=" + absoluteLayerIndex
                    + " seqLen=" + blk.sequenceLen());
            return new KvPair(k, v);
        });
    }

    // ── Eviction ──────────────────────────────────────────────────────────────

    /**
     * Evict all KV blocks for the given request from the manager's GPU and CPU
     * tiers. Call this when a request or session completes and its KV data is no
     * longer needed.
     *
     * @param requestId request or session identifier
     */
    public void evict(String requestId) {
        manager.evict(requestId);
        log.fine("KV evicted from manager: requestId=" + requestId);
    }

    // ── Accessors (for stats / testing) ──────────────────────────────────────

    /** Returns the underlying {@link KVCacheManager}. */
    public KVCacheManager manager() {
        return manager;
    }

    // ── Value type ────────────────────────────────────────────────────────────

    /**
     * Restored key and value arrays for one (requestId, layerIndex) pair.
     * Both arrays have length {@code seqLen * kvDim}.
     */
    public record KvPair(float[] k, float[] v) {}
}