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

import java.time.Instant;

/**
 * A single KV cache block — attention keys + values for one transformer layer
 * of one request sequence.
 *
 * <p>Payload layout depends on {@link #kType()} / {@link #vType()}:
 * <ul>
 * <li>{@link KvElementType#F16}: float32 LE — K then V, each {@code seqLen * kvDim}
 * floats (legacy adapter format).</li>
 * <li>Otherwise: opaque bytes produced by {@code NodeKVCacheAdapter} for the
 * declared element types (typically per-token Q8_0 packs).</li>
 * </ul>
 */
public record KVBlock(KVKey key, byte[] data, // serialized keys + values
		int sequenceLen, // number of tokens cached
		int layerIndex, Instant createdAt, Instant lastAccessedAt,
		KvElementType kType, KvElementType vType) {

	public KVBlock {
		if (key == null)
			throw new IllegalArgumentException("key must not be null");
		if (data == null || data.length == 0)
			throw new IllegalArgumentException("data must not be empty");
		if (sequenceLen < 1)
			throw new IllegalArgumentException("sequenceLen must be >= 1");
		if (kType == null)
			kType = KvElementType.F16;
		if (vType == null)
			vType = KvElementType.F16;
	}

	/** Legacy ctor: float32 K/V payload. */
	public KVBlock(KVKey key, byte[] data, int sequenceLen, int layerIndex,
			Instant createdAt, Instant lastAccessedAt) {
		this(key, data, sequenceLen, layerIndex, createdAt, lastAccessedAt,
				KvElementType.F16, KvElementType.F16);
	}

	/** Size of this block in bytes. */
	public int sizeBytes() {
		return data.length;
	}

	/** Return a copy with updated lastAccessedAt. */
	public KVBlock accessed() {
		return new KVBlock(key, data, sequenceLen, layerIndex, createdAt, Instant.now(), kType, vType);
	}
}
