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
package cab.ml.juno.kvcache;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.time.Instant;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Host KV storage of {@code f16} and {@code f32}: an {@code f16} tensor, dense or paged, stores each value as an IEEE
 * half (two bytes) and reads back that rounding everywhere (rows, attention view, serialized blocks), so the static and
 * continuous schedules read the same values; an {@code f32} tensor keeps the floats it is given, as {@code f16} did
 * before it stored halves.
 */
@DisplayName("Host KV storage - f16 as IEEE halves, f32 as floats")
class HostKvHalfStorageTest {

	private static final int KV_DIM = 96;
	private static final int ROWS = 40;

	@Test
	@DisplayName("dense f16: two bytes a value, reads back the half rounding")
	void denseF16() {
		DenseKvTensor t = new DenseKvTensor(KvElementType.F16, KV_DIM);
		float[][] rows = rows(ROWS, 1);
		for (int p = 0; p < ROWS; p++)
			t.writeToken(p, rows[p]);
		assertThat(t.allocatedBytes()).isEqualTo(t.capacityTokens() * KV_DIM * 2);
		assertThat(t.heapBytes()).isEqualTo(t.allocatedBytes());
		assertRounded(t.toFloatArray(ROWS), rows);
		float[] one = new float[KV_DIM];
		t.readToken(7, one);
		assertRounded(one, rows[7]);
	}

	@Test
	@DisplayName("dense f32: four bytes a value, reads back exactly what was written")
	void denseF32() {
		DenseKvTensor t = new DenseKvTensor(KvElementType.F32, KV_DIM);
		float[][] rows = rows(ROWS, 2);
		for (int p = 0; p < ROWS; p++)
			t.writeToken(p, rows[p]);
		assertThat(t.allocatedBytes()).isEqualTo(t.capacityTokens() * KV_DIM * 4);
		float[] all = t.toFloatArray(ROWS);
		for (int p = 0; p < ROWS; p++)
			for (int i = 0; i < KV_DIM; i++)
				assertThat(all[p * KV_DIM + i]).isEqualTo(rows[p][i]);
		assertThat(t.attentionRows(ROWS).floatPage(0)).isNotNull();
	}

	@Test
	@DisplayName("attention reads an f16 tensor as byte pages decoded a slice at a time, dense and paged alike")
	void attentionView() {
		float[][] rows = rows(ROWS, 3);
		DenseKvTensor dense = new DenseKvTensor(KvElementType.F16, KV_DIM);
		PagedKvArena arena = new PagedKvArena(16, KV_DIM, KvElementType.F16, KvElementType.F16);
		PagedKvTensor paged = arena.newK();
		for (int p = 0; p < ROWS; p++) {
			dense.writeToken(p, rows[p]);
			paged.writeToken(p, rows[p]);
		}
		for (SessionKvTensor t : new SessionKvTensor[] { dense, paged }) {
			KvRows view = t.attentionRows(ROWS);
			assertThat(view.type()).isEqualTo(KvElementType.F16);
			assertThat(view.bytesPerRow()).isEqualTo(2 * KV_DIM);
			float[] slice = new float[32];
			for (int p = 0; p < ROWS; p++) {
				int page = p / view.rowsPerPage();
				int slot = p - page * view.rowsPerPage();
				view.decodeRange(view.bytePage(page), slot * view.bytesPerRow(), 40, 32, slice, 0);
				for (int i = 0; i < 32; i++)
					assertThat(slice[i]).isEqualTo(Float.float16ToFloat(Float.floatToFloat16(rows[p][40 + i])));
			}
		}
		assertThat(paged.toFloatArray(ROWS)).isEqualTo(dense.toFloatArray(ROWS));
	}

	@Test
	@DisplayName("paged f16 pool: pages of two bytes a value")
	void pagedPool() {
		KvBlockPool pool = new KvBlockPool(16, KV_DIM, KvElementType.F16);
		assertThat(pool.bytesPerToken()).isEqualTo(2 * KV_DIM);
		int b = pool.allocate();
		float[] row = rows(1, 4)[0];
		pool.writeToken(b, 3, row);
		float[] got = new float[KV_DIM];
		pool.readToken(b, 3, got);
		assertRounded(got, row);
		assertThat(pool.storedBytes()).isEqualTo(16L * KV_DIM * 2);
		KvBlockPool f32 = new KvBlockPool(16, KV_DIM, KvElementType.F32);
		assertThat(f32.bytesPerToken()).isEqualTo(4 * KV_DIM);
	}

	@Test
	@DisplayName("f16 takes stored rows as they are (a device FP16 copy's bytes); f32 refuses them")
	void encodedRows() {
		float[][] rows = rows(ROWS, 5);
		byte[] halves = new byte[ROWS * 2 * KV_DIM];
		for (int p = 0; p < ROWS; p++)
			F16KvCodec.encode(rows[p], 0, KV_DIM, halves, p * 2 * KV_DIM);
		try (Arena a = Arena.ofConfined()) {
			MemorySegment seg = a.allocate(halves.length);
			MemorySegment.copy(halves, 0, seg, JAVA_BYTE, 0, halves.length);
			DenseKvTensor dense = new DenseKvTensor(KvElementType.F16, KV_DIM);
			dense.writeEncodedRows(0, ROWS, seg, 0);
			PagedKvTensor paged = new PagedKvArena(16, KV_DIM, KvElementType.F16, KvElementType.F16).newK();
			paged.writeEncodedRows(0, ROWS, seg, 0);
			DenseKvTensor viaFloats = new DenseKvTensor(KvElementType.F16, KV_DIM);
			for (int p = 0; p < ROWS; p++)
				viaFloats.writeToken(p, rows[p]);
			assertThat(dense.toFloatArray(ROWS)).isEqualTo(viaFloats.toFloatArray(ROWS));
			assertThat(paged.toFloatArray(ROWS)).isEqualTo(viaFloats.toFloatArray(ROWS));
			DenseKvTensor f32 = new DenseKvTensor(KvElementType.F32, KV_DIM);
			assertThatThrownBy(() -> f32.writeEncodedRows(0, 1, seg, 0)).isInstanceOf(IllegalArgumentException.class)
					.hasMessageContaining("f32");
		}
	}

	@Test
	@DisplayName("serialized: f16 is a typed block of halves; f32 pairs keep the headerless float32 layout")
	void serialized() {
		float[][] rows = rows(ROWS, 6);
		float[] k = flat(rows);
		float[] v = flat(rows(ROWS, 7));
		byte[] half = PagedKvCodec.encodeFloats(k, v, ROWS, KV_DIM, KvElementType.F16, KvElementType.F16);
		assertThat(half.length).isEqualTo(4 + 2 * ROWS * 2 * KV_DIM);
		assertThat(half[3]).isEqualTo((byte) 0x00);
		float[][] back = PagedKvCodec.decodeFloats(block(half, KvElementType.F16), KV_DIM);
		for (int i = 0; i < k.length; i++)
			assertThat(back[0][i]).isEqualTo(Float.float16ToFloat(Float.floatToFloat16(k[i])));

		byte[] full = PagedKvCodec.encodeFloats(k, v, ROWS, KV_DIM, KvElementType.F32, KvElementType.F32);
		assertThat(full.length).isEqualTo(2 * ROWS * KV_DIM * Float.BYTES);
		float[][] exact = PagedKvCodec.decodeFloats(block(full, KvElementType.F32), KV_DIM);
		assertThat(exact[0]).isEqualTo(k);
		assertThat(exact[1]).isEqualTo(v);

		byte[] mixed = PagedKvCodec.encodeFloats(k, v, ROWS, KV_DIM, KvElementType.F32, KvElementType.F16);
		assertThat(mixed[3]).isEqualTo((byte) 0x30);
	}

	@Test
	@DisplayName("a KV block without types is the float32 layout")
	void legacyBlockIsF32() {
		KVBlock b = new KVBlock(new KVKey("r", 0), new byte[8], 1, 0, Instant.now(), Instant.now());
		assertThat(b.kType()).isEqualTo(KvElementType.F32);
		assertThat(b.vType()).isEqualTo(KvElementType.F32);
	}

	private static KVBlock block(byte[] data, KvElementType type) {
		return new KVBlock(new KVKey("r", 0), data, ROWS, 0, Instant.now(), Instant.now(), type, type);
	}

	private static void assertRounded(float[] got, float[][] rows) {
		for (int p = 0; p < rows.length; p++)
			for (int i = 0; i < KV_DIM; i++)
				assertThat(Float.floatToRawIntBits(got[p * KV_DIM + i])).as("row %d value %d", p, i)
						.isEqualTo(Float.floatToRawIntBits(Float.float16ToFloat(Float.floatToFloat16(rows[p][i]))));
	}

	private static void assertRounded(float[] got, float[] row) {
		assertRounded(got, new float[][] { row });
	}

	private static float[] flat(float[][] rows) {
		float[] out = new float[rows.length * KV_DIM];
		for (int p = 0; p < rows.length; p++)
			System.arraycopy(rows[p], 0, out, p * KV_DIM, KV_DIM);
		return out;
	}

	static float[][] rows(int n, long seed) {
		float[][] out = new float[n][];
		for (int p = 0; p < n; p++)
			out[p] = F16KvCodecTest.values(KV_DIM, seed * 1000 + p);
		return out;
	}
}
