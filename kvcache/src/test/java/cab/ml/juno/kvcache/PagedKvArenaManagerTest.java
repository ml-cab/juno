package cab.ml.juno.kvcache;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.time.Instant;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("KVCacheManager paged arena plumbing")
class PagedKvArenaManagerTest {

	private static final long MB = 1024 * 1024;

	@Test
	@DisplayName("manager exposes arena; flush blob roundtrips via put/get")
	void manager_arena_and_blob_roundtrip() {
		int kvDim = 32;
		int seqLen = 5;
		PagedKvArena arena = new PagedKvArena(4, kvDim, KvElementType.Q8_0, KvElementType.F32);
		KVCacheManager mgr = new KVCacheManager(
				new GpuKVCache(100 * MB), new CpuKVCache(1000), LayerRange.all(), arena);

		assertThat(mgr.pagedArena()).isPresent();
		assertThat(mgr.pagedArena().get()).isSameAs(arena);

		PagedKvTensor k = arena.newK();
		PagedKvTensor v = arena.newV();
		float[] tok = new float[kvDim];
		for (int p = 0; p < seqLen; p++) {
			for (int i = 0; i < kvDim; i++)
				tok[i] = p + i * 0.01f;
			k.writeToken(p, tok);
			v.writeToken(p, tok);
		}

		byte[] data = PagedKvCodec.encode(k, v, seqLen);
		KVKey key = new KVKey("req-paged", 0);
		Instant now = Instant.now();
		mgr.put(key, new KVBlock(key, data, seqLen, 0, now, now, k.type(), v.type()));

		KVBlock got = mgr.get(key).orElseThrow();
		assertThat(got.kType()).isEqualTo(KvElementType.Q8_0);
		assertThat(got.vType()).isEqualTo(KvElementType.F32);
		assertThat(got.sequenceLen()).isEqualTo(seqLen);

		PagedKvTensor k2 = arena.newK();
		PagedKvTensor v2 = arena.newV();
		PagedKvCodec.decodeInto(got, kvDim, k2, v2);
		float[] kOut = k2.toFloatArray(seqLen);
		float[] vOut = v2.toFloatArray(seqLen);
		for (int p = 0; p < seqLen; p++) {
			for (int i = 0; i < kvDim; i++) {
				assertThat(kOut[p * kvDim + i]).isCloseTo(p + i * 0.01f, within(0.05f));
				assertThat(vOut[p * kvDim + i]).isCloseTo(p + i * 0.01f, within(1e-5f));
			}
		}
	}

	/**
	 * A flushed block names each side's type in its header; a {@code q4_0} side is stored as its packed rows and
	 * restores to the rows the live tensor read, bit for bit.
	 */
	@Test
	@DisplayName("a q4_0 side flushes packed and restores the rows the tensor read")
	void q4_blob_roundtrip() {
		int kvDim = 64;
		int seqLen = 6;
		PagedKvArena arena = new PagedKvArena(4, kvDim, KvElementType.F32, KvElementType.Q4_0);
		PagedKvTensor k = arena.newK();
		PagedKvTensor v = arena.newV();
		float[] tok = new float[kvDim];
		for (int p = 0; p < seqLen; p++) {
			for (int i = 0; i < kvDim; i++)
				tok[i] = (float) Math.sin(p * 7 + i * 0.3) * (1 + p);
			k.writeToken(p, tok);
			v.writeToken(p, tok);
		}
		byte[] data = PagedKvCodec.encode(k, v, seqLen);
		assertThat(data.length).isEqualTo(4 + seqLen * kvDim * Float.BYTES + seqLen * Q4_0KvCodec.encodedBytes(kvDim));
		KVKey key = new KVKey("req-q4", 0);
		Instant now = Instant.now();
		KVBlock blk = new KVBlock(key, data, seqLen, 0, now, now, k.type(), v.type());
		assertThat(blk.vType()).isEqualTo(KvElementType.Q4_0);

		PagedKvTensor k2 = arena.newK();
		PagedKvTensor v2 = arena.newV();
		PagedKvCodec.decodeInto(blk, kvDim, k2, v2);
		assertThat(k2.toFloatArray(seqLen)).containsExactly(k.toFloatArray(seqLen));
		assertThat(v2.toFloatArray(seqLen)).containsExactly(v.toFloatArray(seqLen));
	}

	@Test
	@DisplayName("evict clears tiers; released tensors free pool pages")
	void evict_and_release() {
		PagedKvArena arena = new PagedKvArena(4, 8, KvElementType.F32, KvElementType.F32);
		KVCacheManager mgr = new KVCacheManager(
				new GpuKVCache(100 * MB), new CpuKVCache(1000), LayerRange.all(), arena);
		PagedKvTensor k = arena.newK();
		float[] tok = new float[8];
		for (int p = 0; p < 5; p++)
			k.writeToken(p, tok);
		assertThat(arena.kPool().liveBlocks()).isEqualTo(2);

		KVKey key = new KVKey("req-x", 1);
		Instant now = Instant.now();
		mgr.put(key, new KVBlock(key, new byte[64], 1, 1, now, now));
		mgr.evict("req-x");
		assertThat(mgr.get(key)).isEmpty();

		k.release();
		assertThat(arena.kPool().liveBlocks()).isZero();
	}
}
