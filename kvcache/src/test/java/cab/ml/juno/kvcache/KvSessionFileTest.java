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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Stream;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import cab.ml.juno.kvcache.KvSessionFileException.Reason;

/**
 * One stage's session KV on disk ({@link KvSessionFile}): the rows come back exactly as stored, under either schedule's
 * layers, and a file that does not describe the layers it is read into, or is damaged, is refused before or without
 * being taken as KV.
 */
class KvSessionFileTest {

	private static final int KV_DIM = 64;
	private static final int LAYERS = 3;
	private static final int TOKENS = 37; // spans three 16-row pages, the last one partly filled
	private static final int PAGE = 16;

	@TempDir
	Path dir;

	static Stream<Arguments> typePairs() {
		return Stream.of(Arguments.of(KvElementType.F16, KvElementType.F16),
				Arguments.of(KvElementType.F32, KvElementType.F32),
				Arguments.of(KvElementType.Q8_0, KvElementType.Q8_0),
				Arguments.of(KvElementType.Q8_0, KvElementType.Q4_0),
				Arguments.of(KvElementType.Q4_0, KvElementType.F16));
	}

	@ParameterizedTest(name = "{0}/{1}")
	@MethodSource("typePairs")
	@DisplayName("dense to dense, paged to paged, dense to paged and paged to dense: every row byte for byte")
	void roundTripsEveryTypeAndLayout(KvElementType typeK, KvElementType typeV) throws IOException {
		for (Layout from : Layout.values()) {
			for (Layout to : Layout.values()) {
				SessionKvTensor[][] saved = filled(from, typeK, typeV, 11);
				Path file = dir.resolve("s-" + from + "-" + to + "-" + typeK + typeV + ".kv");
				long written = KvSessionFile.write(file, stamp(), saved[0], saved[1]);
				assertThat(written).isEqualTo(Files.size(file));

				SessionKvTensor[][] restored = empty(to, typeK, typeV);
				long read = KvSessionFile.read(file, stamp(), restored[0], restored[1]);
				assertThat(read).isEqualTo(written);
				for (int l = 0; l < LAYERS; l++) {
					assertThat(stored(restored[0][l], TOKENS)).as("K layer %d %s->%s", l, from, to)
							.isEqualTo(stored(saved[0][l], TOKENS));
					assertThat(stored(restored[1][l], TOKENS)).as("V layer %d %s->%s", l, from, to)
							.isEqualTo(stored(saved[1][l], TOKENS));
				}
			}
		}
	}

	@Test
	@DisplayName("the file's size is the header, the tokens, the stored rows and an 8-byte checksum")
	void fileHoldsTheStoredRowsOnly() throws IOException {
		SessionKvTensor[][] saved = filled(Layout.DENSE, KvElementType.Q8_0, KvElementType.Q4_0, 3);
		Path file = dir.resolve("size.kv");
		long written = KvSessionFile.write(file, stamp(), saved[0], saved[1]);
		long rows = (long) LAYERS * TOKENS
				* (KvRowCodec.rowBytes(KvElementType.Q8_0, KV_DIM) + KvRowCodec.rowBytes(KvElementType.Q4_0, KV_DIM));
		assertThat(written).isGreaterThan(rows + 4L * TOKENS).isLessThan(rows + 4L * TOKENS + 512);
	}

	@Test
	@DisplayName("a write replaces the file through a temporary one and leaves nothing else in the directory")
	void writeIsRenamedIntoPlace() throws IOException {
		Path file = dir.resolve("atomic.kv");
		SessionKvTensor[][] first = filled(Layout.DENSE, KvElementType.F16, KvElementType.F16, 1);
		KvSessionFile.write(file, stamp(), first[0], first[1]);
		SessionKvTensor[][] second = filled(Layout.DENSE, KvElementType.F16, KvElementType.F16, 2);
		KvSessionFile.write(file, stamp(), second[0], second[1]);
		try (Stream<Path> files = Files.list(dir)) {
			assertThat(files.map(p -> p.getFileName().toString())).containsExactly("atomic.kv");
		}
		SessionKvTensor[][] restored = empty(Layout.DENSE, KvElementType.F16, KvElementType.F16);
		KvSessionFile.read(file, stamp(), restored[0], restored[1]);
		assertThat(stored(restored[0][2], TOKENS)).isEqualTo(stored(second[0][2], TOKENS));
	}

	@Test
	@DisplayName("a missing file: NOT_FOUND")
	void missingFile() {
		SessionKvTensor[][] restored = empty(Layout.DENSE, KvElementType.F16, KvElementType.F16);
		assertThatThrownBy(() -> KvSessionFile.read(dir.resolve("none.kv"), stamp(), restored[0], restored[1]))
				.isInstanceOfSatisfying(KvSessionFileException.class, e -> assertThat(e.reason()).isEqualTo(Reason.NOT_FOUND));
	}

	static Stream<Arguments> mismatches() {
		KvSessionStamp s = stamp();
		int[] otherTokens = s.tokens().clone();
		otherTokens[TOKENS - 1]++;
		return Stream.of(
				Arguments.of("model", (Function<Reader, Object>) r -> r.read(new KvSessionStamp("other.gguf", s.adapter(), s.stage(), s.tokens()))),
				Arguments.of("adapter", (Function<Reader, Object>) r -> r.read(new KvSessionStamp(s.model(), "a.lora", s.stage(), s.tokens()))),
				Arguments.of("stage", (Function<Reader, Object>) r -> r.read(new KvSessionStamp(s.model(), s.adapter(), "layers-0-2", s.tokens()))),
				Arguments.of("tokens", (Function<Reader, Object>) r -> r.read(new KvSessionStamp(s.model(), s.adapter(), s.stage(), otherTokens))),
				Arguments.of("token count", (Function<Reader, Object>) r -> r.read(new KvSessionStamp(s.model(), s.adapter(), s.stage(), Arrays.copyOf(s.tokens(), TOKENS - 1)))),
				Arguments.of("layer count", (Function<Reader, Object>) r -> r.readInto(empty(Layout.DENSE, KvElementType.F16, KvElementType.F16, LAYERS - 1, KV_DIM))),
				Arguments.of("KV width", (Function<Reader, Object>) r -> r.readInto(empty(Layout.DENSE, KvElementType.F16, KvElementType.F16, LAYERS, KV_DIM + 32))),
				Arguments.of("K type", (Function<Reader, Object>) r -> r.readInto(empty(Layout.DENSE, KvElementType.Q8_0, KvElementType.F16, LAYERS, KV_DIM))),
				Arguments.of("V type", (Function<Reader, Object>) r -> r.readInto(empty(Layout.DENSE, KvElementType.F16, KvElementType.F32, LAYERS, KV_DIM))));
	}

	@ParameterizedTest(name = "{0}")
	@MethodSource("mismatches")
	@DisplayName("each header field that differs from the reader's: MISMATCH, naming the field, nothing written")
	void mismatchRefused(String field, Function<Reader, Object> attempt) throws IOException {
		SessionKvTensor[][] saved = filled(Layout.DENSE, KvElementType.F16, KvElementType.F16, 5);
		Path file = dir.resolve("m.kv");
		KvSessionFile.write(file, stamp(), saved[0], saved[1]);
		Reader reader = new Reader(file);
		assertThatThrownBy(() -> attempt.apply(reader))
				.isInstanceOfSatisfying(KvSessionFileException.class, e -> {
					assertThat(e.reason()).isEqualTo(Reason.MISMATCH);
					assertThat(e.getMessage()).containsIgnoringCase(field);
				});
		for (SessionKvTensor t : reader.lastK)
			assertThat(t.heapBytes()).as("no row written before the header was checked").isLessThanOrEqualTo(emptyHeapBytes(t));
	}

	@Test
	@DisplayName("another format version: MISMATCH naming both versions")
	void versionRefused() throws IOException {
		Path file = writeDefault("v.kv");
		byte[] bytes = Files.readAllBytes(file);
		bytes[11] = 99; // the version int follows the 8-byte magic
		Files.write(file, bytes);
		SessionKvTensor[][] restored = empty(Layout.DENSE, KvElementType.F16, KvElementType.F16);
		assertThatThrownBy(() -> KvSessionFile.read(file, stamp(), restored[0], restored[1]))
				.isInstanceOfSatisfying(KvSessionFileException.class, e -> {
					assertThat(e.reason()).isEqualTo(Reason.MISMATCH);
					assertThat(e.getMessage()).contains("99").contains(String.valueOf(KvSessionFile.FORMAT_VERSION));
				});
	}

	@Test
	@DisplayName("not a session file: CORRUPT")
	void wrongMagic() throws IOException {
		Path file = dir.resolve("junk.kv");
		Files.write(file, "not a session file at all, just text".getBytes());
		SessionKvTensor[][] restored = empty(Layout.DENSE, KvElementType.F16, KvElementType.F16);
		assertThatThrownBy(() -> KvSessionFile.read(file, stamp(), restored[0], restored[1]))
				.isInstanceOfSatisfying(KvSessionFileException.class, e -> assertThat(e.reason()).isEqualTo(Reason.CORRUPT));
	}

	@Test
	@DisplayName("a truncated file (rows or checksum cut off): CORRUPT")
	void truncatedRefused() throws IOException {
		Path file = writeDefault("t.kv");
		byte[] bytes = Files.readAllBytes(file);
		for (int cut : new int[] { 4, 200, bytes.length / 2, bytes.length - 3 }) {
			Files.write(file, Arrays.copyOf(bytes, bytes.length - cut));
			SessionKvTensor[][] restored = empty(Layout.PAGED, KvElementType.F16, KvElementType.F16);
			assertThatThrownBy(() -> KvSessionFile.read(file, stamp(), restored[0], restored[1])).as("cut %d", cut)
					.isInstanceOfSatisfying(KvSessionFileException.class, e -> assertThat(e.reason()).isEqualTo(Reason.CORRUPT));
		}
	}

	@Test
	@DisplayName("one changed byte in the rows, or a byte appended: CORRUPT")
	void damagedRefused() throws IOException {
		Path file = writeDefault("d.kv");
		byte[] bytes = Files.readAllBytes(file);
		byte[] flipped = bytes.clone();
		flipped[bytes.length - 100] ^= 0x10;
		Files.write(file, flipped);
		SessionKvTensor[][] a = empty(Layout.DENSE, KvElementType.F16, KvElementType.F16);
		assertThatThrownBy(() -> KvSessionFile.read(file, stamp(), a[0], a[1]))
				.isInstanceOfSatisfying(KvSessionFileException.class, e -> assertThat(e.reason()).isEqualTo(Reason.CORRUPT));

		ByteArrayOutputStream longer = new ByteArrayOutputStream();
		longer.write(bytes);
		longer.write(0);
		Files.write(file, longer.toByteArray());
		SessionKvTensor[][] b = empty(Layout.DENSE, KvElementType.F16, KvElementType.F16);
		assertThatThrownBy(() -> KvSessionFile.read(file, stamp(), b[0], b[1]))
				.isInstanceOfSatisfying(KvSessionFileException.class, e -> assertThat(e.reason()).isEqualTo(Reason.CORRUPT));
	}

	@Test
	@DisplayName("layers holding fewer rows than the stamp's tokens are not saved")
	void shortLayersRefused() {
		SessionKvTensor[][] saved = filled(Layout.PAGED, KvElementType.F16, KvElementType.F16, 5);
		int[] more = Arrays.copyOf(stamp().tokens(), TOKENS + 4);
		KvSessionStamp longer = new KvSessionStamp("m.gguf", "", "layers-0-3", more);
		assertThatThrownBy(() -> KvSessionFile.write(dir.resolve("short.kv"), longer, saved[0], saved[1]))
				.isInstanceOf(IllegalArgumentException.class);
		assertThat(dir.toFile().list()).isEmpty();
	}

	// ── helpers ──────────────────────────────────────────────────────────────

	private enum Layout {
		DENSE, PAGED
	}

	/** Reads the default file into layers of a given shape, recording them for the "nothing written" check. */
	private final class Reader {
		final Path file;
		SessionKvTensor[] lastK = new SessionKvTensor[0];

		Reader(Path file) {
			this.file = file;
		}

		Object read(KvSessionStamp expected) {
			SessionKvTensor[][] layers = empty(Layout.DENSE, KvElementType.F16, KvElementType.F16);
			lastK = layers[0];
			return call(() -> KvSessionFile.read(file, expected, layers[0], layers[1]));
		}

		Object readInto(SessionKvTensor[][] layers) {
			lastK = layers[0];
			return call(() -> KvSessionFile.read(file, stamp(), layers[0], layers[1]));
		}

		private Object call(IoCall c) {
			try {
				return c.run();
			} catch (IOException e) {
				throw new AssertionError(e);
			}
		}
	}

	private interface IoCall {
		Object run() throws IOException;
	}

	private Path writeDefault(String name) throws IOException {
		SessionKvTensor[][] saved = filled(Layout.DENSE, KvElementType.F16, KvElementType.F16, 7);
		Path file = dir.resolve(name);
		KvSessionFile.write(file, stamp(), saved[0], saved[1]);
		return file;
	}

	private static KvSessionStamp stamp() {
		int[] tokens = new int[TOKENS];
		for (int i = 0; i < TOKENS; i++)
			tokens[i] = 1000 + 7 * i;
		return new KvSessionStamp("m.gguf", "", "layers-0-3", tokens);
	}

	private static SessionKvTensor[][] empty(Layout layout, KvElementType typeK, KvElementType typeV) {
		return empty(layout, typeK, typeV, LAYERS, KV_DIM);
	}

	private static SessionKvTensor[][] empty(Layout layout, KvElementType typeK, KvElementType typeV, int layers,
			int kvDim) {
		SessionKvTensor[] k = new SessionKvTensor[layers];
		SessionKvTensor[] v = new SessionKvTensor[layers];
		PagedKvArena arena = layout == Layout.PAGED ? new PagedKvArena(PAGE, kvDim, typeK, typeV) : null;
		for (int l = 0; l < layers; l++) {
			k[l] = arena != null ? arena.newK() : new DenseKvTensor(typeK, kvDim);
			v[l] = arena != null ? arena.newV() : new DenseKvTensor(typeV, kvDim);
		}
		return new SessionKvTensor[][] { k, v };
	}

	private static long emptyHeapBytes(SessionKvTensor t) {
		SessionKvTensor fresh = t instanceof DenseKvTensor ? new DenseKvTensor(t.type(), t.kvDim()) : null;
		return fresh != null ? fresh.heapBytes() : 0;
	}

	private static SessionKvTensor[][] filled(Layout layout, KvElementType typeK, KvElementType typeV, long seed) {
		SessionKvTensor[][] layers = empty(layout, typeK, typeV);
		Random rnd = new Random(seed);
		float[] row = new float[KV_DIM];
		for (int l = 0; l < LAYERS; l++)
			for (int p = 0; p < TOKENS; p++) {
				for (int i = 0; i < KV_DIM; i++)
					row[i] = (float) rnd.nextGaussian();
				layers[0][l].writeToken(p, row);
				for (int i = 0; i < KV_DIM; i++)
					row[i] = (float) rnd.nextGaussian();
				layers[1][l].writeToken(p, row);
			}
		return layers;
	}

	/** Positions {@code [0, n)} of {@code t} as stored: encoded bytes, or the floats' bits for {@code f32}. */
	private static byte[] stored(SessionKvTensor t, int n) {
		KvRows rows = t.attentionRows(n);
		ByteArrayOutputStream out = new ByteArrayOutputStream();
		for (int p = 0; p < n; p++) {
			int page = rows.rowsPerPage() == KvRows.ONE_PAGE ? 0 : p / rows.rowsPerPage();
			int slot = rows.rowsPerPage() == KvRows.ONE_PAGE ? p : p % rows.rowsPerPage();
			if (rows.type().storesFloats()) {
				float[] f = rows.floatPage(page);
				for (int i = 0; i < rows.kvDim(); i++) {
					int bits = Float.floatToRawIntBits(f[slot * rows.kvDim() + i]);
					out.write(bits);
					out.write(bits >>> 8);
					out.write(bits >>> 16);
					out.write(bits >>> 24);
				}
			} else {
				out.write(rows.bytePage(page), slot * rows.bytesPerRow(), rows.bytesPerRow());
			}
		}
		return out.toByteArray();
	}
}
