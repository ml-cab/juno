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

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Shared pool of fixed-size KV pages (token slots × {@code kvDim}).
 *
 * <p>Used by the continuous-schedule paged KV path. A page is a float array of
 * {@code kvDim} floats per row for {@link KvElementType#F16} (the current float
 * path) or a byte array of packed rows for {@link KvElementType#Q8_0} and {@link KvElementType#Q4_0}
 * ({@link KvRowCodec}).
 * Attention reads the pages in place ({@link KvRows}), through
 * {@link #floatPage} and {@link #bytePage}.
 *
 * <p>Each page carries a reference count. {@link #allocate} hands out a page
 * held once; {@link #retain} adds a holder (a second {@link KvPageTable} sharing
 * a prefix); {@link #free} drops one, and the page returns to the free list when
 * the last holder drops it. A page with more than one holder is read-only:
 * {@link #writeToken} and {@link #copyToken} refuse to write into it, and a
 * holder that needs to write first takes a page of its own with
 * {@link #exclusive}.
 *
 * <p>A page every holder has dropped is idle: it keeps its storage, and the
 * next {@link #allocate} takes it before making a new one. Idle pages are what
 * a burst of requests leaves behind. {@link #reclaimIdle} releases the storage
 * of those that have stayed unused for a whole period between two calls, so the
 * heap gets them back once the load has dropped while a steady load keeps
 * reusing them; their ids stay, and get new zeroed storage when next allocated.
 * No live page is moved or touched: attention reads each page where it is
 * through its table, so a page in use never needs to change place.
 * {@link KvPageReclaimer} calls it in the background.
 */
public final class KvBlockPool {

	private final int pageSize;
	private final int kvDim;
	private final KvElementType type;
	private final int bytesPerToken;
	private final int bytesPerPage;

	private final Object lock = new Object();
	/** F16 pages, else null. */
	private final List<float[]> floatPages;
	/** Q8_0 pages, else null. */
	private final List<byte[]> bytePages;
	/** Holders per page; 0 means free. Indexed by block id, grown with {@link #pages}. */
	private int[] refs = new int[16];
	/** Idle pages that still hold their storage, reused first. */
	private final IntRing free = new IntRing();
	/** Free ids whose storage {@link #reclaimIdle} released; given new storage when next allocated. */
	private final IntRing released = new IntRing();
	private int live;
	/** The fewest idle pages since the last {@link #reclaimIdle}: that many stayed unused the whole period. */
	private int fewestIdle;
	private long reclaimed;

	/** Most idle pages released per hold of the lock, so a large reclaim never stalls a writer for long. */
	static final int RECLAIM_CHUNK = 256;

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
		this.bytesPerToken = KvRowCodec.rowBytes(type, kvDim);
		this.bytesPerPage = pageSize * bytesPerToken;
		this.floatPages = type == KvElementType.F16 ? new ArrayList<>() : null;
		this.bytePages = type == KvElementType.F16 ? null : new ArrayList<>();
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
		int id;
		if (!free.isEmpty()) {
			id = free.pollFirst();
			if (free.size() < fewestIdle)
				fewestIdle = free.size();
		} else if (!released.isEmpty()) {
			id = released.pollFirst();
			if (floatPages != null)
				floatPages.set(id, new float[pageSize * kvDim]);
			else
				bytePages.set(id, new byte[bytesPerPage]);
		} else {
			id = pageCount();
			if (floatPages != null)
				floatPages.add(new float[pageSize * kvDim]);
			else
				bytePages.add(new byte[bytesPerPage]);
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
			return blockId >= 0 && blockId < pageCount() ? refs[blockId] : 0;
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
			if (floatPages != null)
				System.arraycopy(floatPages.get(blockId), 0, floatPages.get(copy), 0, pageSize * kvDim);
			else
				System.arraycopy(bytePages.get(blockId), 0, bytePages.get(copy), 0, bytesPerPage);
			refs[blockId]--;
			return copy;
		}
	}

	/**
	 * Releases the storage of the idle pages that stayed unused since the previous call (the first call counts from
	 * construction), and starts the next period. A page taken and returned during the period was needed, so it
	 * keeps its storage; live and shared pages are never touched. Returns the pages released.
	 */
	public int reclaimIdle() {
		int target;
		synchronized (lock) {
			target = fewestIdle;
		}
		int done = 0;
		while (true) {
			synchronized (lock) {
				int n = Math.min(Math.min(RECLAIM_CHUNK, target - done), free.size());
				for (int i = 0; i < n; i++) {
					int id = free.pollLast();
					if (floatPages != null)
						floatPages.set(id, null);
					else
						bytePages.set(id, null);
					released.addLast(id);
				}
				done += n;
				reclaimed += n;
				if (n == 0 || done == target) {
					fewestIdle = free.size();
					return done;
				}
			}
			// The lock is not fair: give a writer waiting on it the next hold.
			Thread.yield();
		}
	}

	/** Idle pages that still hold their storage. */
	public int idlePages() {
		synchronized (lock) {
			return free.size();
		}
	}

	/** Pages that hold storage: the live ones and the idle ones. */
	public int storedPages() {
		synchronized (lock) {
			return live + free.size();
		}
	}

	/** Heap bytes the pages' storage takes: {@link #storedPages} pages. */
	public long storedBytes() {
		synchronized (lock) {
			return (long) (live + free.size()) * bytesPerPage;
		}
	}

	/** Idle pages whose storage {@link #reclaimIdle} has released, since construction. */
	public long reclaimedPages() {
		synchronized (lock) {
			return reclaimed;
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
	 * The live {@code F16} page {@code blockId} itself: {@code kvDim} floats per row. The caller reads it while it
	 * holds the page (never across its {@link #free}) and writes it only through {@link #writeToken}.
	 */
	float[] floatPage(int blockId) {
		synchronized (lock) {
			checkLive(blockId);
			return floatPages.get(blockId);
		}
	}

	/** As {@link #floatPage}, for a {@code Q8_0} pool's page: {@link #bytesPerToken} bytes per row. */
	byte[] bytePage(int blockId) {
		synchronized (lock) {
			checkLive(blockId);
			return bytePages.get(blockId);
		}
	}

	/**
	 * A copy of page {@code blockId}'s stored bytes: float32 little-endian rows for {@code F16}, packed rows for
	 * {@code Q8_0}. For tests that compare page contents.
	 */
	byte[] pageBytes(int blockId) {
		synchronized (lock) {
			checkLive(blockId);
			if (bytePages != null)
				return bytePages.get(blockId).clone();
			byte[] out = new byte[bytesPerPage];
			ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().put(floatPages.get(blockId));
			return out;
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
		float[] f32 = null;
		byte[] q8 = null;
		synchronized (lock) {
			checkWritable(blockId);
			if (floatPages != null)
				f32 = floatPages.get(blockId);
			else
				q8 = bytePages.get(blockId);
		}
		if (f32 != null)
			System.arraycopy(src, srcOff, f32, slot * kvDim, kvDim);
		else
			KvRowCodec.encode(type, src, srcOff, kvDim, q8, slot * bytesPerToken);
	}

	/**
	 * Copies {@code count} stored rows ({@link #bytesPerToken} bytes each, already in this pool's blocks) from
	 * {@code src} at byte offset {@code srcOffset} into slots {@code [slot, slot + count)} of page {@code blockId}.
	 *
	 * @throws IllegalArgumentException for an {@code F16} pool, whose rows are floats, not blocks
	 */
	public void writeEncodedRows(int blockId, int slot, int count, MemorySegment src, long srcOffset) {
		if (bytePages == null)
			throw new IllegalArgumentException("f16 rows are stored as floats, not encoded");
		if (slot < 0 || count < 1 || slot + count > pageSize)
			throw new IllegalArgumentException("slots [" + slot + ", " + (slot + count) + ") outside a page of "
					+ pageSize);
		byte[] page;
		synchronized (lock) {
			checkWritable(blockId);
			page = bytePages.get(blockId);
		}
		MemorySegment.copy(src, ValueLayout.JAVA_BYTE, srcOffset, page, slot * bytesPerToken, count * bytesPerToken);
	}

	public void readToken(int blockId, int slot, float[] dst) {
		readToken(blockId, slot, dst, 0);
	}

	public void readToken(int blockId, int slot, float[] dst, int dstOff) {
		if (slot < 0 || slot >= pageSize)
			throw new IllegalArgumentException("slot out of range: " + slot);
		if (dstOff < 0 || dstOff + kvDim > dst.length)
			throw new IllegalArgumentException("dst too small for kvDim=" + kvDim);
		if (floatPages != null)
			System.arraycopy(floatPage(blockId), slot * kvDim, dst, dstOff, kvDim);
		else
			KvRowCodec.decode(type, bytePage(blockId), slot * bytesPerToken, dst, dstOff, kvDim);
	}

	/** Copy one stored row between slots, byte for byte (no re-encoding). */
	public void copyToken(int srcBlock, int srcSlot, int dstBlock, int dstSlot) {
		if (srcSlot < 0 || srcSlot >= pageSize || dstSlot < 0 || dstSlot >= pageSize)
			throw new IllegalArgumentException("slot out of range: " + srcSlot + " -> " + dstSlot);
		Object src;
		Object dst;
		synchronized (lock) {
			checkLive(srcBlock);
			checkWritable(dstBlock);
			src = floatPages != null ? floatPages.get(srcBlock) : bytePages.get(srcBlock);
			dst = floatPages != null ? floatPages.get(dstBlock) : bytePages.get(dstBlock);
		}
		int rowElems = floatPages != null ? kvDim : bytesPerToken;
		System.arraycopy(src, srcSlot * rowElems, dst, dstSlot * rowElems, rowElems);
	}

	private int pageCount() {
		return floatPages != null ? floatPages.size() : bytePages.size();
	}

	private void checkLive(int blockId) {
		if (blockId < 0 || blockId >= pageCount() || refs[blockId] == 0)
			throw new IllegalStateException("block not live: " + blockId);
	}

	private void checkWritable(int blockId) {
		checkLive(blockId);
		if (refs[blockId] > 1)
			throw new IllegalStateException("block " + blockId + " is shared by " + refs[blockId]
					+ " holders and cannot be written; take a page of its own first");
	}
}
