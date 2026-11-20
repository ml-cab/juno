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

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UTFDataFormatException;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.ThreadLocalRandom;
import java.util.zip.CRC32;
import java.util.zip.CheckedInputStream;
import java.util.zip.CheckedOutputStream;

import cab.ml.juno.kvcache.KvSessionFileException.Reason;

/**
 * One stage's KV of one session on disk: what an explicit save writes and a restore, possibly in another process,
 * reads back.
 *
 * <p>Layout, big-endian except the rows: the magic {@code JUNOKVS1}; the format version ({@link #FORMAT_VERSION}); the
 * stamp's model, adapter and stage (modified UTF-8); the layer count, the KV width and the K and V element types by
 * name; the token count {@code n} and the {@code n} tokens; then for each layer its K rows and its V rows at positions
 * {@code [0, n)} exactly as the tensor stores them ({@link KvRows}: IEEE halves for {@code f16}, the blocks for
 * {@code q8_0} / {@code q4_0}, little-endian floats for {@code f32}); then a CRC-32 of everything before it.
 *
 * <p>Rows are stored by position, never by page, so a file written from the static schedule's dense KV restores into
 * the continuous schedule's paged KV and the other way round. A write goes to a temporary file in the same directory,
 * forced to disk and renamed over the target, so a reader never sees half a file. A read checks the whole header
 * against the reader's stamp and layers before it writes a row, and the checksum after the last row; on any failure
 * the caller releases the layers it passed.
 */
public final class KvSessionFile {

	/** Bumped whenever the layout changes; a file of another version is refused, never guessed at. */
	public static final int FORMAT_VERSION = 1;

	private static final long MAGIC = 0x4A554E4F4B565331L; // "JUNOKVS1"
	private static final int CHUNK_BYTES = 1 << 20;
	private static final int STREAM_BUFFER = 1 << 16;

	private KvSessionFile() {
	}

	/**
	 * Writes positions {@code [0, stamp.positions())} of {@code k} and {@code v} (one entry per layer) to {@code file},
	 * replacing it.
	 *
	 * @return the file's size in bytes
	 * @throws IllegalArgumentException if the layers are inconsistent or hold fewer rows than the stamp's tokens;
	 *                                  nothing is written then
	 */
	public static long write(Path file, KvSessionStamp stamp, SessionKvTensor[] k, SessionKvTensor[] v)
			throws IOException {
		checkLayers(k, v);
		int n = stamp.positions();
		KvRows[] kRows = new KvRows[k.length];
		KvRows[] vRows = new KvRows[v.length];
		for (int l = 0; l < k.length; l++) {
			kRows[l] = rowsOf(k[l], n, "K", l);
			vRows[l] = rowsOf(v[l], n, "V", l);
		}
		Path tmp = file.resolveSibling(
				file.getFileName() + ".tmp-" + Long.toHexString(ThreadLocalRandom.current().nextLong()));
		try {
			try (FileChannel ch = FileChannel.open(tmp, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
				OutputStream raw = new BufferedOutputStream(Channels.newOutputStream(ch), STREAM_BUFFER);
				CheckedOutputStream checked = new CheckedOutputStream(raw, new CRC32());
				DataOutputStream out = new DataOutputStream(checked);
				out.writeLong(MAGIC);
				out.writeInt(FORMAT_VERSION);
				out.writeUTF(stamp.model());
				out.writeUTF(stamp.adapter());
				out.writeUTF(stamp.stage());
				out.writeInt(k.length);
				out.writeInt(k[0].kvDim());
				out.writeUTF(k[0].type().cliName());
				out.writeUTF(v[0].type().cliName());
				out.writeInt(n);
				for (int t : stamp.tokens())
					out.writeInt(t);
				byte[] scratch = new byte[CHUNK_BYTES];
				for (int l = 0; l < k.length; l++) {
					writeRows(out, kRows[l], n, scratch);
					writeRows(out, vRows[l], n, scratch);
				}
				long crc = checked.getChecksum().getValue();
				new DataOutputStream(raw).writeLong(crc);
				raw.flush();
				ch.force(true);
			}
			try {
				Files.move(tmp, file, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
			} catch (AtomicMoveNotSupportedException e) {
				Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING);
			}
		} catch (IOException | RuntimeException | Error e) {
			try {
				Files.deleteIfExists(tmp);
			} catch (IOException suppressed) {
				e.addSuppressed(suppressed);
			}
			throw e;
		}
		return Files.size(file);
	}

	/**
	 * Reads {@code file} into {@code k} and {@code v}, new empty layers of the reader's shape, after checking that it
	 * holds exactly this KV: {@code expected}'s model, adapter, stage and tokens, the layers' count, width and element
	 * types, and this format version.
	 *
	 * @return the file's size in bytes
	 * @throws KvSessionFileException {@link Reason#NOT_FOUND} without the file, {@link Reason#MISMATCH} naming the
	 *                                first header field that differs (no row written yet), {@link Reason#CORRUPT} for
	 *                                a file that is not a session file, ends early, or fails its checksum
	 */
	public static long read(Path file, KvSessionStamp expected, SessionKvTensor[] k, SessionKvTensor[] v)
			throws IOException {
		checkLayers(k, v);
		InputStream opened;
		try {
			opened = Files.newInputStream(file);
		} catch (NoSuchFileException e) {
			throw new KvSessionFileException(Reason.NOT_FOUND, "no saved session file " + file.getFileName());
		}
		try (InputStream raw = new BufferedInputStream(opened, STREAM_BUFFER)) {
			CheckedInputStream checked = new CheckedInputStream(raw, new CRC32());
			DataInputStream in = new DataInputStream(checked);
			if (in.readLong() != MAGIC)
				throw corrupt(file, "not a Juno session file");
			int version = in.readInt();
			if (version != FORMAT_VERSION)
				throw mismatch(file, "format version " + version + "; this build reads version " + FORMAT_VERSION);
			expect(file, "model", in.readUTF(), expected.model());
			expect(file, "adapter", in.readUTF(), expected.adapter());
			expect(file, "stage", in.readUTF(), expected.stage());
			int layers = in.readInt();
			if (layers != k.length)
				throw mismatch(file, "layer count: file " + layers + ", this stage " + k.length);
			int kvDim = in.readInt();
			if (kvDim != k[0].kvDim())
				throw mismatch(file, "KV width: file " + kvDim + ", this stage " + k[0].kvDim());
			expect(file, "K type", in.readUTF(), k[0].type().cliName());
			expect(file, "V type", in.readUTF(), v[0].type().cliName());
			int n = in.readInt();
			if (n < 1)
				throw corrupt(file, "token count " + n);
			if (n != expected.positions())
				throw mismatch(file, "token count: file " + n + ", the session's manifest " + expected.positions());
			int[] want = expected.tokens();
			for (int i = 0; i < n; i++) {
				int t = in.readInt();
				if (t != want[i])
					throw mismatch(file, "tokens differ at position " + i);
			}
			byte[] scratch = new byte[CHUNK_BYTES];
			for (int l = 0; l < k.length; l++) {
				readRows(in, k[l], n, scratch);
				readRows(in, v[l], n, scratch);
			}
			long crc = checked.getChecksum().getValue();
			long stored = new DataInputStream(raw).readLong();
			if (crc != stored)
				throw corrupt(file, "checksum differs: the file changed after it was written");
			if (raw.read() != -1)
				throw corrupt(file, "bytes after the checksum");
		} catch (EOFException e) {
			throw corrupt(file, "the file ends early (truncated)");
		} catch (UTFDataFormatException e) {
			throw corrupt(file, "unreadable header text");
		}
		return Files.size(file);
	}

	private static void checkLayers(SessionKvTensor[] k, SessionKvTensor[] v) {
		if (k == null || v == null || k.length < 1 || k.length != v.length)
			throw new IllegalArgumentException("K and V need the same number of layers, at least one");
		for (int l = 0; l < k.length; l++) {
			if (k[l] == null || v[l] == null)
				throw new IllegalArgumentException("layer " + l + " is missing");
			if (k[l].kvDim() != k[0].kvDim() || v[l].kvDim() != k[0].kvDim())
				throw new IllegalArgumentException("layer " + l + " has another KV width");
			if (k[l].type() != k[0].type() || v[l].type() != v[0].type())
				throw new IllegalArgumentException("layer " + l + " has another element type");
		}
	}

	private static KvRows rowsOf(SessionKvTensor t, int n, String which, int layer) {
		if (t.capacityTokens() < n)
			throw new IllegalArgumentException(which + " layer " + layer + " holds " + t.capacityTokens()
					+ " positions, fewer than the " + n + " to save");
		return t.attentionRows(n); // a paged tensor refuses more rows than it stores
	}

	private static void writeRows(DataOutputStream out, KvRows rows, int n, byte[] scratch) throws IOException {
		int rpp = rows.rowsPerPage();
		boolean floats = rows.type().storesFloats();
		int dim = rows.kvDim();
		int bpr = rows.bytesPerRow();
		ByteBuffer le = floats ? ByteBuffer.wrap(scratch).order(ByteOrder.LITTLE_ENDIAN) : null;
		FloatBuffer leFloats = floats ? le.asFloatBuffer() : null;
		int p = 0;
		while (p < n) {
			int page = rpp == KvRows.ONE_PAGE ? 0 : p / rpp;
			int slot = rpp == KvRows.ONE_PAGE ? p : p - page * rpp;
			int count = rpp == KvRows.ONE_PAGE ? n - p : Math.min(rpp - slot, n - p);
			if (!floats) {
				out.write(rows.bytePage(page), slot * bpr, count * bpr);
			} else {
				float[] src = rows.floatPage(page);
				int from = slot * dim;
				int left = count * dim;
				int perChunk = scratch.length / Float.BYTES;
				while (left > 0) {
					int m = Math.min(perChunk, left);
					leFloats.clear();
					leFloats.put(src, from, m);
					out.write(scratch, 0, m * Float.BYTES);
					from += m;
					left -= m;
				}
			}
			p += count;
		}
	}

	private static void readRows(DataInputStream in, SessionKvTensor t, int n, byte[] scratch) throws IOException {
		int dim = t.kvDim();
		int bpr = KvRowCodec.rowBytes(t.type(), dim);
		if (bpr > scratch.length)
			throw new IllegalArgumentException("row of " + bpr + " bytes exceeds the read buffer");
		int perChunk = scratch.length / bpr;
		if (!t.type().storesFloats()) {
			MemorySegment seg = MemorySegment.ofArray(scratch);
			for (int p = 0; p < n;) {
				int c = Math.min(perChunk, n - p);
				in.readFully(scratch, 0, c * bpr);
				t.writeEncodedRows(p, c, seg, 0);
				p += c;
			}
			return;
		}
		FloatBuffer le = ByteBuffer.wrap(scratch).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
		float[] row = new float[dim];
		for (int p = 0; p < n;) {
			int c = Math.min(perChunk, n - p);
			in.readFully(scratch, 0, c * bpr);
			for (int r = 0; r < c; r++) {
				le.position(r * dim);
				le.get(row, 0, dim);
				t.writeToken(p + r, row);
			}
			p += c;
		}
	}

	private static void expect(Path file, String field, String found, String wanted) {
		if (!found.equals(wanted))
			throw mismatch(file, field + ": file \"" + found + "\", this server \"" + wanted + "\"");
	}

	private static KvSessionFileException mismatch(Path file, String what) {
		return new KvSessionFileException(Reason.MISMATCH, file.getFileName() + " holds other KV: " + what);
	}

	private static KvSessionFileException corrupt(Path file, String what) {
		return new KvSessionFileException(Reason.CORRUPT, file.getFileName() + " is damaged: " + what);
	}
}
