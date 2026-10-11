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

import java.util.Random;

/** Synthetic packed weights for the integer-kernel tests and microbenchmarks. Measurement only. */
final class CpuIntegerKernelsTestSupport {

	private CpuIntegerKernelsTestSupport() {
	}

	/** Random packed blocks with finite FP16 scales. */
	static GgufReader.QuantizedTensor synthetic(int type, int rows, int cols, Random rnd) {
		int blockElems = type == 8 ? 32 : 256;
		int blockBytes = switch (type) {
		case 8 -> 34;
		case 12 -> 144;
		case 13 -> 176;
		case 14 -> 210;
		default -> throw new IllegalArgumentException("type " + type);
		};
		int blocks = rows * (cols / blockElems);
		byte[] raw = new byte[blocks * blockBytes];
		rnd.nextBytes(raw);
		for (int i = 0; i < blocks; i++) {
			int bo = i * blockBytes;
			switch (type) {
			case 8 -> putF16(raw, bo, 0.01f + 0.02f * rnd.nextFloat());
			case 12, 13 -> {
				putF16(raw, bo, 0.005f + 0.02f * rnd.nextFloat());
				putF16(raw, bo + 2, 0.002f + 0.01f * rnd.nextFloat());
			}
			case 14 -> putF16(raw, bo + 208, 0.001f + 0.004f * rnd.nextFloat());
			default -> throw new IllegalArgumentException();
			}
		}
		return new GgufReader.QuantizedTensor("w", type, (long) rows * cols, raw);
	}

	private static void putF16(byte[] raw, int off, float v) {
		short h = Float.floatToFloat16(v);
		raw[off] = (byte) h;
		raw[off + 1] = (byte) (h >> 8);
	}
}
