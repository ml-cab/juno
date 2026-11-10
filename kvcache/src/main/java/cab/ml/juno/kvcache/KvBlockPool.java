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
import java.util.Arrays;
import java.util.List;

/**
 * Shared pool of fixed-size KV pages (token slots × {@code kvDim}).
 *
 * <p>Used by the continuous-schedule paged KV path. Payloads are float32 for
 * {@link KvElementType#F16} (current float path) or packed q8_0 bytes for
 * {@link KvElementType#Q8_0}.
 *
 * <p>Each page carries a reference count. {@link #allocate} hands out a page
 * held once; {@link #retain} adds a holder (a second {@link KvPageTable} sharing
 * a prefix); {@link #free} drops one, and the page returns to the free list when
 * the last holder drops it. A page with more than one holder is read-only:
 * {@link #writeToken} and {@link #copyToken} refuse to write into it, and a
 * holder that needs to write first takes a page of its own with
 * {@link #exclusive}.
 */
public final class KvBlockPool {

	private final int pageSize;
	private final int kvDim;
	private final KvElementType type;
	private final int bytesPerToken;
	private final int bytesPerPage;

	private final Object lock = new Object();
	private final List<byte[]> pages = new ArrayList<>();
	/** Holders per page; 0 means free. Indexed by block id, grown with {@link #pages}. */
	private int[] refs = new int[16];
	private final ArrayDeque<Integer> free = new ArrayDeque<>();
	private int live;

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

	/** Allocate a free page, held once; recycles when available. */
	public int allocate() {
		synchronized (lock) {
			return allocateLocked();
		}
	}

	private int allocateLocked() {
		Integer recycled = free.pollFirst();
		int id;
		if (recycled != null) {
			id = recycled;
		} else {
			id = pages.size();
			pages.add(new byte[bytesPerPage]);
			if (id == refs.length)
				refs = Arrays.copyOf(refs, 2 * refs.length);
		}
		refs[id] = 1;
		live++;
		return id;
	}

	/** Drop one holder of {@code blockId}; the page returns to the pool when the last holder drops it. */
	public void free(int blockId) {
		synchronized (lock) {
			checkLive(blockId);
			if (--refs[blockId] == 0) {
				live--;
				free.addLast(blockId);
			}
		}
	}

	/** Add a holder to the live page {@code blockId}. */
	void retain(int blockId) {
		synchronized (lock) {
			checkLive(blockId);
			refs[blockId]++;
		}
	}

	/** Holders of {@code blockId}; 0 when it is free. */
	int refCount(int blockId) {
		synchronized (lock) {
			return blockId >= 0 && blockId < pages.size() ? refs[blockId] : 0;
		}
	}

	/**
	 * A page the caller may write in place of its hold on {@code blockId}: the same
	 * page when the caller is its only holder, otherwise a new page with the same
	 * bytes, and the caller's hold on {@code blockId} dropped. Atomic, so holders of
	 * one page calling this at once each end with a page of their own.
	 */
	int exclusive(int blockId) {
		synchronized (lock) {
			checkLive(blockId);
			if (refs[blockId] == 1)
				return blockId;
			int copy = allocateLocked();
			System.arraycopy(pages.get(blockId), 0, pages.get(copy), 0, bytesPerPage);
			refs[blockId]--;
			return copy;
		}
	}

	/** Distinct pages in use, each counted once however many tables hold it. */
	public int liveBlocks() {
		synchronized (lock) {
			return live;
		}
	}

	public long allocatedBytes() {
		synchronized (lock) {
			return (long) live * bytesPerPage;
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
			checkWritable(blockId);
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

	/** Copy one stored row between slots, byte for byte (no re-encoding). */
	public void copyToken(int srcBlock, int srcSlot, int dstBlock, int dstSlot) {
		if (srcSlot < 0 || srcSlot >= pageSize || dstSlot < 0 || dstSlot >= pageSize)
			throw new IllegalArgumentException("slot out of range: " + srcSlot + " -> " + dstSlot);
		byte[] src;
		byte[] dst;
		synchronized (lock) {
			checkLive(srcBlock);
			checkWritable(dstBlock);
			src = pages.get(srcBlock);
			dst = pages.get(dstBlock);
		}
		System.arraycopy(src, srcSlot * bytesPerToken, dst, dstSlot * bytesPerToken, bytesPerToken);
	}

	private void checkLive(int blockId) {
		if (blockId < 0 || blockId >= pages.size() || refs[blockId] == 0)
			throw new IllegalStateException("block not live: " + blockId);
	}

	private void checkWritable(int blockId) {
		checkLive(blockId);
		if (refs[blockId] > 1)
			throw new IllegalStateException("block " + blockId + " is shared by " + refs[blockId]
					+ " holders and cannot be written; take a page of its own first");
	}
}
