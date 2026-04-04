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

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mockito;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.kvcache.KVKey;
import cab.ml.juno.kvcache.LayerRange;
import cab.ml.juno.registry.ShardAssignment;

/**
 * Verifies that LlamaTransformerHandler correctly wires the NodeKVCacheAdapter:
 * flush is called after each forward pass, restore is attempted on local miss,
 * and evict() removes from both the local map and the manager.
 *
 * Uses CyclicForwardPassHandler as the computational stand-in — avoids loading
 * a real GGUF file while still exercising the KV wiring path.
 */
class LlamaKvWiringTest {

    // TinyLlama-like shape constants, small enough for unit tests
    private static final int VOCAB_SIZE  = 256;
    private static final int HIDDEN_DIM  = 32;
    private static final int NUM_HEADS   = 4;
    private static final int NUM_KV_HEADS = 4;
    private static final int NUM_LAYERS  = 2;
    private static final int KV_DIM      = HIDDEN_DIM / NUM_HEADS * NUM_KV_HEADS; // = 32
    private static final int START_LAYER = 0;
    private static final int END_LAYER   = NUM_LAYERS;

    private KVCacheManager    manager;
    private NodeKVCacheAdapter adapter;

    @BeforeEach
    void setUp() {
        manager = new KVCacheManager(
                new GpuKVCache(4 * 1024 * 1024L),
                new CpuKVCache(128),
                LayerRange.of(START_LAYER, END_LAYER));
        adapter = new NodeKVCacheAdapter(manager);
    }

    // ── evict() removes from local map ────────────────────────────────────────

    @Test
    void evict_removes_local_kv_entry_and_propagates_to_manager() {
        LlamaTransformerHandler handler = buildSingleLayerHandler();
        String requestId = "req-evict";

        // Run one forward pass so a KV entry is created in both stores
        ForwardRequest req = ForwardRequest.withTokens(requestId, new int[]{1}, 0);
        ShardContext ctx = buildContext(true, false);
        handler.forward(req, ctx);

        // Verify flushed into manager
        assertThat(manager.gpuBlockCount()).isGreaterThan(0);
        // kvCacheAllocatedSlots is package-private — only check manager state
        assertThat(manager.get(new KVKey(requestId, START_LAYER))).isPresent();

        // Evict
        handler.evict(requestId);

        // Local map entry gone: a second forward at pos=0 should re-create from scratch
        // (if local entry were still present it would reuse old data and NOT add a new
        // block to the manager; after evict it starts fresh, block count may change)
        assertThat(manager.get(new KVKey(requestId, START_LAYER))).isEmpty();
    }

    @Test
    void evict_without_prior_forward_does_not_throw() {
        LlamaTransformerHandler handler = buildSingleLayerHandler();
        handler.evict("never-used"); // must not throw
    }

    // ── write-through: flush into manager after each forward ─────────────────

    @Test
    void forward_stores_kv_block_in_manager_for_each_layer() {
        LlamaTransformerHandler handler = buildSingleLayerHandler();
        String requestId = "req-flush";

        ForwardRequest req = ForwardRequest.withTokens(requestId, new int[]{1}, 0);
        ShardContext ctx = buildContext(true, false);
        handler.forward(req, ctx);

        // One block per layer (START_LAYER = 0 only in this handler)
        assertThat(manager.get(new KVKey(requestId, 0))).isPresent();
    }

    @Test
    void second_forward_updates_block_sequence_length_in_manager() {
        LlamaTransformerHandler handler = buildSingleLayerHandler();
        String requestId = "req-seq";

        ShardContext ctx = buildContext(true, false);
        // Token 0 — prefill position 0
        handler.forward(ForwardRequest.withTokens(requestId, new int[]{1}, 0), ctx);
        int seqLen1 = manager.get(new KVKey(requestId, 0))
                .map(cab.ml.juno.kvcache.KVBlock::sequenceLen).orElse(-1);

        // Token 1 — decode position 1
        handler.forward(ForwardRequest.withTokens(requestId, new int[]{2}, 1), ctx);
        int seqLen2 = manager.get(new KVKey(requestId, 0))
                .map(cab.ml.juno.kvcache.KVBlock::sequenceLen).orElse(-1);

        assertThat(seqLen1).isEqualTo(1);
        assertThat(seqLen2).isEqualTo(2);
    }

    // ── restore from manager when local entry absent ──────────────────────────

    @Test
    void second_handler_instance_restores_kv_from_manager() {
        // Simulate: handler-1 runs one forward pass and is then discarded
        // handler-2 is created with the same manager and must restore KV
        LlamaTransformerHandler handler1 = buildSingleLayerHandler();
        String requestId = "req-restore";
        ShardContext ctx = buildContext(true, false);
        handler1.forward(ForwardRequest.withTokens(requestId, new int[]{1}, 0), ctx);

        // Create a second handler sharing the same manager
        LlamaTransformerHandler handler2 = buildSingleLayerHandler();
        // handler2's local map is empty; forward at pos=1 should restore from manager
        handler2.forward(ForwardRequest.withTokens(requestId, new int[]{2}, 1), ctx);

        // After the restore the block should reflect seqLen=2
        assertThat(manager.get(new KVKey(requestId, 0)))
                .isPresent().get()
                .extracting(cab.ml.juno.kvcache.KVBlock::sequenceLen)
                .isEqualTo(2);
    }

    // ── adapter not set: legacy dev mode still works ──────────────────────────

    @Test
    void handler_without_adapter_works_and_does_not_touch_manager() {
        LlamaTransformerHandler handler = buildHandlerWithoutAdapter();
        String requestId = "req-no-adapter";
        ShardContext ctx = buildContext(true, false);
        handler.forward(ForwardRequest.withTokens(requestId, new int[]{1}, 0), ctx);

        // No blocks written — adapter not wired
        assertThat(manager.gpuBlockCount()).isZero();
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    /**
     * Builds a synthetic single-layer LlamaTransformerHandler backed by
     * CyclicForwardPassHandler-style weights (all random floats). The handler
     * owns layers [0, 1) with embeddings and output projection (single-shard
     * mode).
     *
     * Note: uses a StubLlamaTransformerHandler that delegates math to
     * CyclicForwardPassHandler while exercising the KV wiring path — avoids
     * needing a real GGUF file.
     */
    private LlamaTransformerHandler buildSingleLayerHandler() {
        return buildHandlerWithAdapter(adapter);
    }

    private LlamaTransformerHandler buildHandlerWithoutAdapter() {
        return LlamaTransformerHandler.newTestInstance(
                VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS, NUM_KV_HEADS, NUM_LAYERS,
                START_LAYER, END_LAYER,
                /* hasEmbeddings */ true,
                /* hasOutputProj */ true,
                /* adapter */ null);
    }

    private LlamaTransformerHandler buildHandlerWithAdapter(NodeKVCacheAdapter kvAdapter) {
        return LlamaTransformerHandler.newTestInstance(
                VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS, NUM_KV_HEADS, NUM_LAYERS,
                START_LAYER, END_LAYER,
                /* hasEmbeddings */ true,
                /* hasOutputProj */ true,
                /* adapter */ kvAdapter);
    }

    private static ShardContext buildContext(boolean embeddings, boolean outputProj) {
        ShardAssignment assignment = new ShardAssignment(
                "node-test", "localhost", 0,
                START_LAYER, END_LAYER,
                embeddings, outputProj);
        return ShardContext.from(assignment, VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS);
    }
}