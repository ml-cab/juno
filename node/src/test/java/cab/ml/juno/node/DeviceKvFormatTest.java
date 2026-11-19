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
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.CacheTypeOptions;
import cab.ml.juno.kvcache.KvElementType;
import cab.ml.juno.kvcache.KvRowCodec;

@DisplayName("DeviceKvFormat - how a device KV row is stored")
class DeviceKvFormatTest {

	/** kvDim of the four sweep models: TinyLlama, Qwen2.5-3B, Phi-3.5-mini, Mistral 7B. */
	private static final int[] SWEEP_KV_DIMS = { 256, 256, 3072, 1024 };

	@Test
	@DisplayName("row bytes: FP16 2 per value, q8_0 34 and q4_0 18 per 32 values (the host codecs' blocks)")
	void rowBytes() {
		assertThat(DeviceKvFormat.F16.rowBytes(256)).isEqualTo(512);
		assertThat(DeviceKvFormat.Q8_0.rowBytes(256)).isEqualTo(8 * 34);
		assertThat(DeviceKvFormat.Q4_0.rowBytes(256)).isEqualTo(8 * 18);
		for (int kvDim : SWEEP_KV_DIMS) {
			assertThat(DeviceKvFormat.Q8_0.rowBytes(kvDim)).isEqualTo(KvRowCodec.rowBytes(KvElementType.Q8_0, kvDim));
			assertThat(DeviceKvFormat.Q4_0.rowBytes(kvDim)).isEqualTo(KvRowCodec.rowBytes(KvElementType.Q4_0, kvDim));
		}
	}

	@Test
	@DisplayName("bytes per position against the FP16 copy: q8_0 <= 0.55x, q4_0 <= 0.30x on every sweep model")
	void bytesPerPositionMeetTheBars() {
		for (int kvDim : SWEEP_KV_DIMS) {
			double f16 = DeviceKvFormats.F16.bytesPerPosition(kvDim);
			DeviceKvFormats q8 = new DeviceKvFormats(DeviceKvFormat.Q8_0, DeviceKvFormat.Q8_0);
			DeviceKvFormats q4 = new DeviceKvFormats(DeviceKvFormat.Q4_0, DeviceKvFormat.Q4_0);
			DeviceKvFormats q8q4 = new DeviceKvFormats(DeviceKvFormat.Q8_0, DeviceKvFormat.Q4_0);
			assertThat(q8.bytesPerPosition(kvDim) / f16).as("q8_0 at kvDim " + kvDim).isLessThanOrEqualTo(0.55)
					.isEqualTo(34.0 / 64);
			assertThat(q4.bytesPerPosition(kvDim) / f16).as("q4_0 at kvDim " + kvDim).isLessThanOrEqualTo(0.30)
					.isEqualTo(18.0 / 64);
			assertThat(q8q4.bytesPerPosition(kvDim) / f16).as("q8_0 K, q4_0 V at kvDim " + kvDim)
					.isEqualTo(52.0 / 128);
		}
	}

	@Test
	@DisplayName("the device holds a quantized type only for a head width that is a multiple of 32, else FP16")
	void holding() {
		assertThat(DeviceKvFormat.holding(KvElementType.F16, 80)).isEqualTo(DeviceKvFormat.F16);
		assertThat(DeviceKvFormat.holding(KvElementType.Q8_0, 128)).isEqualTo(DeviceKvFormat.Q8_0);
		assertThat(DeviceKvFormat.holding(KvElementType.Q4_0, 96)).isEqualTo(DeviceKvFormat.Q4_0);
		assertThat(DeviceKvFormat.holding(KvElementType.Q8_0, 64)).isEqualTo(DeviceKvFormat.Q8_0);
		// Phi-2's 80-wide heads: 32 of them make a 2,560-wide row of whole blocks, but a block would straddle two heads.
		assertThat(DeviceKvFormat.holding(KvElementType.Q4_0, 80)).isEqualTo(DeviceKvFormat.F16);
		assertThat(DeviceKvFormat.holding(KvElementType.Q8_0, 48)).isEqualTo(DeviceKvFormat.F16);
		assertThat(DeviceKvFormat.Q8_0.hostType()).isEqualTo(KvElementType.Q8_0);
		assertThat(DeviceKvFormat.F16.quantized()).isFalse();
		assertThat(DeviceKvFormat.Q4_0.quantized()).isTrue();
	}

	@Test
	@DisplayName("selection: the flags' types when the kernel reads them, FP16 when it does not, probed only for a quantized type")
	void select() {
		CacheTypeOptions q8q4 = CacheTypeOptions.of(KvElementType.Q8_0, KvElementType.Q4_0);
		CacheTypeOptions f16 = CacheTypeOptions.defaults();
		assertThat(DeviceKvFormats.select(q8q4, 128, () -> true))
				.isEqualTo(new DeviceKvFormats(DeviceKvFormat.Q8_0, DeviceKvFormat.Q4_0));
		assertThat(DeviceKvFormats.select(q8q4, 128, () -> false)).isEqualTo(DeviceKvFormats.F16);
		assertThat(DeviceKvFormats.select(q8q4, 80, () -> true)).isEqualTo(DeviceKvFormats.F16);
		assertThat(DeviceKvFormats.select(f16, 128, () -> {
			throw new AssertionError("an f16 launch must not load the quantized kernel");
		})).isEqualTo(DeviceKvFormats.F16);
		assertThat(DeviceKvFormats.select(CacheTypeOptions.of(KvElementType.F16, KvElementType.Q8_0), 64, () -> true))
				.isEqualTo(new DeviceKvFormats(DeviceKvFormat.F16, DeviceKvFormat.Q8_0));
	}

	@Test
	@DisplayName("holds: the device rows carry the host types' values exactly when both sides match")
	void holds() {
		DeviceKvFormats q8q4 = new DeviceKvFormats(DeviceKvFormat.Q8_0, DeviceKvFormat.Q4_0);
		assertThat(q8q4.holds(CacheTypeOptions.of(KvElementType.Q8_0, KvElementType.Q4_0))).isTrue();
		assertThat(q8q4.holds(CacheTypeOptions.of(KvElementType.Q8_0, KvElementType.Q8_0))).isFalse();
		assertThat(DeviceKvFormats.F16.holds(CacheTypeOptions.defaults())).isTrue();
		assertThat(DeviceKvFormats.F16.holds(CacheTypeOptions.of(KvElementType.Q8_0, KvElementType.Q8_0))).isFalse();
		assertThat(q8q4.quantized()).isTrue();
		assertThat(DeviceKvFormats.F16.quantized()).isFalse();
		assertThat(q8q4.summary()).isEqualTo("K q8_0, V q4_0");
	}
	@Test
	@DisplayName("f32: never stored on the device; FP16 holds it for sharing; selected without the quantized kernel")
	void f32() {
		assertThat(DeviceKvFormat.canHold(KvElementType.F32, 128)).isFalse();
		assertThat(DeviceKvFormat.canHold(KvElementType.F16, 80)).isTrue();
		assertThat(DeviceKvFormat.holding(KvElementType.F32, 128)).isEqualTo(DeviceKvFormat.F16);
		CacheTypeOptions f32 = CacheTypeOptions.of(KvElementType.F32, KvElementType.F32);
		assertThat(DeviceKvFormats.select(f32, 128, () -> {
			throw new AssertionError("an f32 launch loads no quantized kernel");
		})).isEqualTo(DeviceKvFormats.F16);
		assertThat(DeviceKvFormats.F16.holds(f32)).isTrue();
		assertThat(new DeviceKvFormats(DeviceKvFormat.F16, DeviceKvFormat.Q8_0)
				.holds(CacheTypeOptions.of(KvElementType.F32, KvElementType.Q8_0))).isTrue();
		assertThat(new DeviceKvFormats(DeviceKvFormat.Q8_0, DeviceKvFormat.Q8_0).holds(f32)).isFalse();
	}

	@Test
	@DisplayName("an FP16 device row is the f16 host row, byte for byte")
	void f16RowIsTheHostRow() {
		float[] row = new float[64];
		for (int i = 0; i < row.length; i++)
			row[i] = (float) Math.sin(i * 0.7) * (i - 20);
		byte[] host = new byte[KvRowCodec.rowBytes(KvElementType.F16, row.length)];
		KvRowCodec.encode(KvElementType.F16, row, 0, row.length, host, 0);
		try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
			java.lang.foreign.MemorySegment seg = a.allocate(DeviceKvFormat.F16.rowBytes(row.length));
			DeviceKvFormat.F16.pack(row, 0, row.length, seg, 0);
			assertThat(seg.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE)).isEqualTo(host);
		}
	}


	@Test
	@DisplayName("host packing: FP16 rounds as Float.floatToFloat16; q8_0 and q4_0 are the host codecs' bytes")
	void packIsTheHostEncoding() {
		int kvDim = 96;
		float[] row = randomRow(new Random(7), kvDim);
		row[5] = 3.5f;
		row[40] = -3.5f; // q4_0: two elements of one block at the same magnitude, the first sets the scale
		row[33] = -3.5f;
		try (Arena arena = Arena.ofConfined()) {
			MemorySegment f16 = arena.allocate(DeviceKvFormat.F16.rowBytes(kvDim) + 8);
			DeviceKvFormat.F16.pack(row, 0, kvDim, f16, 8);
			for (int i = 0; i < kvDim; i++)
				assertThat(f16.get(ValueLayout.JAVA_SHORT_UNALIGNED, 8 + 2L * i)).isEqualTo(Float.floatToFloat16(row[i]));
			for (DeviceKvFormat f : new DeviceKvFormat[] { DeviceKvFormat.Q8_0, DeviceKvFormat.Q4_0 }) {
				byte[] host = new byte[KvRowCodec.rowBytes(f.hostType(), kvDim)];
				KvRowCodec.encode(f.hostType(), row, 0, kvDim, host, 0);
				MemorySegment dev = arena.allocate(f.rowBytes(kvDim) + 4);
				f.pack(row, 0, kvDim, dev, 4);
				assertThat(dev.asSlice(4, host.length).toArray(ValueLayout.JAVA_BYTE)).as(f.name()).containsExactly(host);
			}
		}
	}

	@Test
	@DisplayName("host unpacking reads the values the host codecs decode, bit for bit")
	void unpackIsTheHostDecoding() {
		int kvDim = 128;
		float[] row = randomRow(new Random(11), kvDim);
		try (Arena arena = Arena.ofConfined()) {
			for (DeviceKvFormat f : DeviceKvFormat.values()) {
				MemorySegment dev = arena.allocate(f.rowBytes(kvDim));
				f.pack(row, 0, kvDim, dev, 0);
				float[] got = new float[kvDim + 3];
				f.unpack(dev, 0, kvDim, got, 3);
				float[] want = new float[kvDim];
				if (f.quantized()) {
					byte[] host = new byte[KvRowCodec.rowBytes(f.hostType(), kvDim)];
					KvRowCodec.encode(f.hostType(), row, 0, kvDim, host, 0);
					KvRowCodec.decode(f.hostType(), host, 0, want, 0, kvDim);
				} else {
					for (int i = 0; i < kvDim; i++)
						want[i] = Float.float16ToFloat(Float.floatToFloat16(row[i]));
				}
				for (int i = 0; i < kvDim; i++)
					assertThat(Float.floatToRawIntBits(got[3 + i])).as(f + " [" + i + "]")
							.isEqualTo(Float.floatToRawIntBits(want[i]));
			}
		}
	}

	@Test
	@DisplayName("a quantized format refuses a kvDim it cannot store")
	void refusesARaggedRow() {
		assertThrows(IllegalArgumentException.class, () -> DeviceKvFormat.Q8_0.rowBytes(80));
		assertThrows(IllegalArgumentException.class, () -> DeviceKvFormat.F16.rowBytes(0));
	}

	@Test
	@DisplayName("sessions that fit: whole sessions of layers x tokens x bytes per position, K and V")
	void sessionsThatFit() {
		int layers = 32;
		int kvDim = 1024; // Mistral 7B
		long f16Session = 32L * 2048 * 2 * 1024 * 2;
		long q8Session = 32L * 2048 * 2 * 32 * 34;
		DeviceKvFormats q8 = new DeviceKvFormats(DeviceKvFormat.Q8_0, DeviceKvFormat.Q8_0);
		assertThat(DeviceKvFormats.F16.sessionBytes(layers, kvDim, 2048)).isEqualTo(f16Session);
		assertThat(q8.sessionBytes(layers, kvDim, 2048)).isEqualTo(q8Session);
		assertThat((double) f16Session / q8Session).isEqualTo(64.0 / 34);
		long free = 3L << 30;
		assertThat(DeviceKvFormats.F16.sessionsThatFit(free, layers, kvDim, 2048)).isEqualTo(free / f16Session);
		assertThat(q8.sessionsThatFit(free, layers, kvDim, 2048)).isEqualTo(free / q8Session);
		assertThat(DeviceKvFormats.F16.sessionsThatFit(f16Session - 1, layers, kvDim, 2048)).isZero();
		assertThat(DeviceKvFormats.F16.sessionsThatFit(-5, layers, kvDim, 2048)).isZero();
	}

	private static float[] randomRow(Random rng, int n) {
		float[] row = new float[n];
		for (int i = 0; i < n; i++)
			row[i] = (float) rng.nextGaussian();
		return row;
	}
}
