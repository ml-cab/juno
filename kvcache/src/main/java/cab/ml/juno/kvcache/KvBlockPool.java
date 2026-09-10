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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

/**
 * Shared pool of fixed-size KV pages (token slots × {@code kvDim}).
 *
 * <p>Used by the continuous-schedule paged KV path. Payloads are float32 for
 * {@link KvElementType#F16} (current float path) or packed q8_0 bytes for
 * {@link KvElementType#Q8_0}.
 */
public final class KvBlockPool {

	private final int pageSize;
	private final int kvDim;
	private final KvElementType type;
	private final int bytesPerToken;
	private final int bytesPerPage;

	private final Object lock = new Object();
	private final List<byte[]> pages = new ArrayList<>();
	private final BitSet live = new BitSet();
	private final ArrayDeque<Integer> free = new ArrayDeque<>();

	public KvBlockPool(int pageSize, int kvDim, KvElementType type) {
		if (pageSize < 1)
			throw new IllegalArgumentException("pageSize must be >= 1");
		if (kvDim < 1)
			throw new IllegalArgumentException("kvDim must be >= 1");
		if (type == null)
			throw new IllegalArgumentException("type must not be null");
		this.pageSize = pageSize;
		this.kvDim = kvDim;
		this.type = type;
		this.bytesPerToken = switch (type) {
		case F16 -> kvDim * Float.BYTES;
		case Q8_0 -> Q8_0KvCodec.encodedBytes(kvDim);
		};
		this.bytesPerPage = pageSize * bytesPerToken;
	}

	public int pageSize() {
		return pageSize;
	}

	public int kvDim() {
		return kvDim;
	}

	public KvElementType type() {
		return type;
	}

	public int bytesPerToken() {
		return bytesPerToken;
	}

	/** Allocate a free page; recycles when available. */
	public int allocate() {
		synchronized (lock) {
			Integer recycled = free.pollFirst();
			if (recycled != null) {
				live.set(recycled);
				return recycled;
			}
			int id = pages.size();
			pages.add(new byte[bytesPerPage]);
			live.set(id);
			return id;
		}
	}

	public void free(int blockId) {
		synchronized (lock) {
			checkLive(blockId);
			live.clear(blockId);
			free.addLast(blockId);
		}
	}

	public int liveBlocks() {
		synchronized (lock) {
			return live.cardinality();
		}
	}

	public long allocatedBytes() {
		synchronized (lock) {
			return (long) live.cardinality() * bytesPerPage;
		}
	}

	/**
	 * Live page payload (caller must not retain across {@link #free}). Used by
	 * F16 bulk gather.
	 */
	byte[] pageBytes(int blockId) {
		synchronized (lock) {
			checkLive(blockId);
			return pages.get(blockId);
		}
	}

	public void writeToken(int blockId, int slot, float[] src) {
		writeToken(blockId, slot, src, 0);
	}

	public void writeToken(int blockId, int slot, float[] src, int srcOff) {
		if (slot < 0 || slot >= pageSize)
			throw new IllegalArgumentException("slot out of range: " + slot);
		if (srcOff < 0 || srcOff + kvDim > src.length)
			throw new IllegalArgumentException("src too small for kvDim=" + kvDim);
		byte[] page;
		synchronized (lock) {
			checkLive(blockId);
			page = pages.get(blockId);
		}
		int off = slot * bytesPerToken;
		switch (type) {
		case F16 -> {
			ByteBuffer.wrap(page, off, bytesPerToken).order(ByteOrder.LITTLE_ENDIAN)
					.asFloatBuffer().put(src, srcOff, kvDim);
		}
		case Q8_0 -> Q8_0KvCodec.encode(src, srcOff, kvDim, page, off);
		}
	}

	public void readToken(int blockId, int slot, float[] dst) {
		readToken(blockId, slot, dst, 0);
	}

	public void readToken(int blockId, int slot, float[] dst, int dstOff) {
		if (slot < 0 || slot >= pageSize)
			throw new IllegalArgumentException("slot out of range: " + slot);
		if (dstOff < 0 || dstOff + kvDim > dst.length)
			throw new IllegalArgumentException("dst too small for kvDim=" + kvDim);
		byte[] page;
		synchronized (lock) {
			checkLive(blockId);
			page = pages.get(blockId);
		}
		int off = slot * bytesPerToken;
		switch (type) {
		case F16 -> {
			ByteBuffer.wrap(page, off, bytesPerToken).order(ByteOrder.LITTLE_ENDIAN)
					.asFloatBuffer().get(dst, dstOff, kvDim);
		}
		case Q8_0 -> Q8_0KvCodec.decode(page, off, dst, dstOff, kvDim);
		}
	}

	private void checkLive(int blockId) {
		if (blockId < 0 || blockId >= pages.size() || !live.get(blockId))
			throw new IllegalStateException("block not live: " + blockId);
	}
}
