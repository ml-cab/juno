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

import java.util.Arrays;

/**
 * One layer's K or V rows of one request as attention reads them: the stored rows themselves, never a copy
 * ({@link SessionKvTensor#attentionRows}).
 *
 * <p>Rows sit in pages of {@link #rowsPerPage()} rows: row {@code t} is row {@code t % rowsPerPage()} of page
 * {@code t / rowsPerPage()}. A paged tensor's pages are its block table's pages in the pool, in table order; a dense
 * tensor is one page holding every row ({@link #ONE_PAGE}). An {@link KvElementType#F16} page is a float array of
 * {@code kvDim} floats per row ({@link #floatPage}). A {@link KvElementType#Q8_0} page is a byte array of
 * {@link #bytesPerRow()} bytes per row in {@link Q8_0KvCodec}'s blocks ({@link #bytePage}); the reader decodes the
 * slice it needs as it goes ({@link Q8_0KvCodec#decodeRange}).
 *
 * <p>The tensor that hands out a view keeps it current as its storage changes (a page added or dropped, a shared
 * page copied before a write, a dense array grown), so a view is valid until the tensor is next written. Attention
 * reads it between a forward pass's KV write and the next one.
 */
public final class KvRows {

	/** {@link #rowsPerPage()} of a one-page view: every row of the tensor is in page 0. */
	public static final int ONE_PAGE = Integer.MAX_VALUE;

	private final KvElementType type;
	private final int kvDim;
	private final int bytesPerRow;
	private final int rowsPerPage;
	private float[][] floatPages;
	private byte[][] bytePages;

	KvRows(KvElementType type, int kvDim, int rowsPerPage) {
		if (type == null)
			throw new IllegalArgumentException("type must not be null");
		if (kvDim < 1 || rowsPerPage < 1)
			throw new IllegalArgumentException("kvDim and rowsPerPage must be >= 1");
		this.type = type;
		this.kvDim = kvDim;
		this.rowsPerPage = rowsPerPage;
		this.bytesPerRow = switch (type) {
		case F16 -> kvDim * Float.BYTES;
		case Q8_0 -> Q8_0KvCodec.encodedBytes(kvDim);
		};
		if (type == KvElementType.F16)
			floatPages = new float[1][];
		else
			bytePages = new byte[1][];
	}

	/** {@code F16} rows already held in one array, {@code kvDim} floats per row: a one-page view of that array. */
	public static KvRows of(float[] rows, int kvDim) {
		KvRows view = new KvRows(KvElementType.F16, kvDim, ONE_PAGE);
		view.floatPages[0] = rows;
		return view;
	}

	public KvElementType type() {
		return type;
	}

	public int kvDim() {
		return kvDim;
	}

	/** Rows per page; {@link #ONE_PAGE} when every row is in page 0. */
	public int rowsPerPage() {
		return rowsPerPage;
	}

	/** Stored bytes per row: {@code kvDim} floats for {@code F16}, the row's Q8_0 blocks for {@code Q8_0}. */
	public int bytesPerRow() {
		return bytesPerRow;
	}

	/** Page {@code page} of an {@code F16} view: {@code kvDim} floats per row, read in place. */
	public float[] floatPage(int page) {
		return floatPages[page];
	}

	/** Page {@code page} of a {@code Q8_0} view: {@link #bytesPerRow()} bytes per row, read in place. */
	public byte[] bytePage(int page) {
		return bytePages[page];
	}

	void setPage(int page, float[] rows) {
		if (page >= floatPages.length)
			floatPages = Arrays.copyOf(floatPages, Math.max(page + 1, 2 * floatPages.length));
		floatPages[page] = rows;
	}

	void setPage(int page, byte[] rows) {
		if (page >= bytePages.length)
			bytePages = Arrays.copyOf(bytePages, Math.max(page + 1, 2 * bytePages.length));
		bytePages[page] = rows;
	}

	/** Forgets pages {@code page} and after: the tensor no longer holds them. */
	void clearFrom(int page) {
		if (floatPages != null)
			Arrays.fill(floatPages, Math.min(page, floatPages.length), floatPages.length, null);
		else
			Arrays.fill(bytePages, Math.min(page, bytePages.length), bytePages.length, null);
	}
}
