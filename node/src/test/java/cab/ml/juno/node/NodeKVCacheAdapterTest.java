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
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.Optional;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;

class NodeKVCacheAdapterTest {

    private static final int KV_DIM    = 16;
    private static final int SEQ_LEN   = 4;
    private static final int ABS_LAYER = 2;
    private static final String REQUEST_ID = "req-001";

    private KVCacheManager  manager;
    private NodeKVCacheAdapter adapter;

    @BeforeEach
    void setUp() {
        // small budget so eviction logic can be exercised in other tests
        manager = new KVCacheManager(
                new GpuKVCache(1024 * 1024L),   // 1 MB VRAM budget
                new CpuKVCache(64));             // 64 blocks CPU
        adapter = new NodeKVCacheAdapter(manager);
    }

    // ── constructor ───────────────────────────────────────────────────────────

    @Test
    void constructor_rejects_null_manager() {
        assertThatThrownBy(() -> new NodeKVCacheAdapter(null))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void manager_accessor_returns_wrapped_instance() {
        assertThat(adapter.manager()).isSameAs(manager);
    }

    // ── flush ─────────────────────────────────────────────────────────────────

    @Test
    void flush_stores_block_retrievable_via_manager() {
        float[] k = kArray(SEQ_LEN, KV_DIM, 1.0f);
        float[] v = kArray(SEQ_LEN, KV_DIM, 2.0f);

        adapter.flush(REQUEST_ID, ABS_LAYER, k, v, SEQ_LEN, KV_DIM);

        assertThat(manager.gpuBlockCount()).isEqualTo(1);
    }

    @Test
    void flush_block_has_correct_sequence_length() {
        float[] k = kArray(SEQ_LEN, KV_DIM, 0.5f);
        float[] v = kArray(SEQ_LEN, KV_DIM, 0.5f);
        adapter.flush(REQUEST_ID, ABS_LAYER, k, v, SEQ_LEN, KV_DIM);

        cab.ml.juno.kvcache.KVKey key = new cab.ml.juno.kvcache.KVKey(REQUEST_ID, ABS_LAYER);
        assertThat(manager.get(key)).isPresent().get()
                .extracting(cab.ml.juno.kvcache.KVBlock::sequenceLen)
                .isEqualTo(SEQ_LEN);
    }

    @Test
    void flush_updates_block_on_second_call_with_longer_sequence() {
        float[] k1 = kArray(2, KV_DIM, 1.0f);
        float[] v1 = kArray(2, KV_DIM, 2.0f);
        adapter.flush(REQUEST_ID, ABS_LAYER, k1, v1, 2, KV_DIM);

        float[] k2 = kArray(5, KV_DIM, 3.0f);
        float[] v2 = kArray(5, KV_DIM, 4.0f);
        adapter.flush(REQUEST_ID, ABS_LAYER, k2, v2, 5, KV_DIM);

        cab.ml.juno.kvcache.KVKey key = new cab.ml.juno.kvcache.KVKey(REQUEST_ID, ABS_LAYER);
        assertThat(manager.get(key)).isPresent().get()
                .extracting(cab.ml.juno.kvcache.KVBlock::sequenceLen)
                .isEqualTo(5);
        // Only one block per key
        assertThat(manager.gpuBlockCount()).isEqualTo(1);
    }

    @Test
    void flush_different_layers_produce_distinct_blocks() {
        float[] k = kArray(SEQ_LEN, KV_DIM, 1.0f);
        float[] v = kArray(SEQ_LEN, KV_DIM, 2.0f);
        adapter.flush(REQUEST_ID, 0, k, v, SEQ_LEN, KV_DIM);
        adapter.flush(REQUEST_ID, 1, k, v, SEQ_LEN, KV_DIM);
        adapter.flush(REQUEST_ID, 2, k, v, SEQ_LEN, KV_DIM);

        assertThat(manager.gpuBlockCount()).isEqualTo(3);
    }

    // ── tryRestore ────────────────────────────────────────────────────────────

    @Test
    void tryRestore_returns_empty_when_no_block_exists() {
        Optional<NodeKVCacheAdapter.KvPair> result =
                adapter.tryRestore(REQUEST_ID, ABS_LAYER, KV_DIM);
        assertThat(result).isEmpty();
    }

    @Test
    void tryRestore_returns_kv_pair_after_flush() {
        float[] k = kArray(SEQ_LEN, KV_DIM, 1.0f);
        float[] v = kArray(SEQ_LEN, KV_DIM, 2.0f);
        adapter.flush(REQUEST_ID, ABS_LAYER, k, v, SEQ_LEN, KV_DIM);

        Optional<NodeKVCacheAdapter.KvPair> result =
                adapter.tryRestore(REQUEST_ID, ABS_LAYER, KV_DIM);

        assertThat(result).isPresent();
        assertThat(result.get().k()).hasSize(SEQ_LEN * KV_DIM);
        assertThat(result.get().v()).hasSize(SEQ_LEN * KV_DIM);
    }


    @Test
    void tryRestore_reconstructs_k_values_correctly() {
        // Use distinct recognisable values
        float[] k = new float[SEQ_LEN * KV_DIM];
        float[] v = new float[SEQ_LEN * KV_DIM];
        for (int i = 0; i < k.length; i++) {
            k[i] = i + 0.5f;
            v[i] = -(i + 0.5f);
        }
        adapter.flush(REQUEST_ID, ABS_LAYER, k, v, SEQ_LEN, KV_DIM);

        NodeKVCacheAdapter.KvPair pair =
                adapter.tryRestore(REQUEST_ID, ABS_LAYER, KV_DIM).orElseThrow();

        for (int i = 0; i < k.length; i++) {
            assertThat(pair.k()[i]).isCloseTo(k[i], within(1e-6f));
            assertThat(pair.v()[i]).isCloseTo(v[i], within(1e-6f));
        }
    }

    @Test
    void tryRestore_round_trips_zero_values() {
        float[] k = new float[1 * KV_DIM]; // all zeros
        float[] v = new float[1 * KV_DIM];
        adapter.flush(REQUEST_ID, ABS_LAYER, k, v, 1, KV_DIM);

        NodeKVCacheAdapter.KvPair pair =
                adapter.tryRestore(REQUEST_ID, ABS_LAYER, KV_DIM).orElseThrow();

        for (float f : pair.k()) assertThat(f).isZero();
        for (float f : pair.v()) assertThat(f).isZero();
    }

    // ── evict ─────────────────────────────────────────────────────────────────

    @Test
    void evict_removes_all_layers_for_request_from_manager() {
        float[] k = kArray(SEQ_LEN, KV_DIM, 1.0f);
        float[] v = kArray(SEQ_LEN, KV_DIM, 2.0f);
        adapter.flush(REQUEST_ID, 0, k, v, SEQ_LEN, KV_DIM);
        adapter.flush(REQUEST_ID, 1, k, v, SEQ_LEN, KV_DIM);
        assertThat(manager.gpuBlockCount()).isEqualTo(2);

        adapter.evict(REQUEST_ID);

        assertThat(manager.gpuBlockCount()).isZero();
    }

    @Test
    void evict_does_not_remove_other_requests() {
        float[] k = kArray(SEQ_LEN, KV_DIM, 1.0f);
        float[] v = kArray(SEQ_LEN, KV_DIM, 2.0f);
        adapter.flush("req-A", ABS_LAYER, k, v, SEQ_LEN, KV_DIM);
        adapter.flush("req-B", ABS_LAYER, k, v, SEQ_LEN, KV_DIM);

        adapter.evict("req-A");

        assertThat(manager.gpuBlockCount()).isEqualTo(1);
        assertThat(adapter.tryRestore("req-B", ABS_LAYER, KV_DIM)).isPresent();
        assertThat(adapter.tryRestore("req-A", ABS_LAYER, KV_DIM)).isEmpty();
    }

    @Test
    void evict_is_idempotent() {
        float[] k = kArray(SEQ_LEN, KV_DIM, 1.0f);
        float[] v = kArray(SEQ_LEN, KV_DIM, 2.0f);
        adapter.flush(REQUEST_ID, ABS_LAYER, k, v, SEQ_LEN, KV_DIM);

        adapter.evict(REQUEST_ID);
        adapter.evict(REQUEST_ID); // should not throw

        assertThat(manager.gpuBlockCount()).isZero();
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    /** Allocate a float[] with seqLen * kvDim elements all set to {@code value}. */
    private static float[] kArray(int seqLen, int kvDim, float value) {
        float[] arr = new float[seqLen * kvDim];
        java.util.Arrays.fill(arr, value);
        return arr;
    }
}