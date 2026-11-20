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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import cab.ml.juno.kvcache.CacheTypeOptions;
import cab.ml.juno.kvcache.KvPageSizeOptions;
import cab.ml.juno.kvcache.KvSessionFileException;
import cab.ml.juno.kvcache.KvSessionFileException.Reason;
import cab.ml.juno.kvcache.KvSessionStamp;
import cab.ml.juno.kvcache.ServeScheduleOptions;

/**
 * A session saved by one handler ({@link ForwardPassHandler#saveKv}) and restored into a new handler built from the same
 * weights ({@link ForwardPassHandler#restoreKv}), as a later process would, reads at its next position, bit for bit,
 * the logits the unbroken session reads: on every handler family, under every pair of save and restore schedules
 * (rows by position, dense or paged on either side), on three host KV types. A file that does not match the handler is
 * refused and leaves the request without KV; a request without KV is not saved. The prompt ends mid-page.
 */
@DisplayName("Session KV snapshot - a restored session continues as the unbroken one")
class KvSessionSnapshotHandlerTest {

	private static final int PAGE_SIZE = 4;
	private static final int PROMPT = 11;
	private static final int FIRST_WINDOW = 7;

	private final String prevSchedule = System.getProperty(ServeScheduleOptions.ENV);
	private final String prevPage = System.getProperty(KvPageSizeOptions.ENV);
	private final String prevTypeK = System.getProperty(CacheTypeOptions.ENV_K);
	private final String prevTypeV = System.getProperty(CacheTypeOptions.ENV_V);

	@TempDir
	Path tmp;

	@AfterEach
	void restore() {
		restoreProp(ServeScheduleOptions.ENV, prevSchedule);
		restoreProp(KvPageSizeOptions.ENV, prevPage);
		restoreProp(CacheTypeOptions.ENV_K, prevTypeK);
		restoreProp(CacheTypeOptions.ENV_V, prevTypeV);
	}

	static Stream<Arguments> cases() {
		List<Arguments> out = new ArrayList<>();
		PagedKvHandlerOracleTest.families().forEach(f -> {
			for (String saveSchedule : new String[] { "static", "continuous" })
				for (String restoreSchedule : new String[] { "static", "continuous" })
					for (String type : new String[] { "f16", "q8_0", "f32" })
						out.add(Arguments.of(f, saveSchedule, restoreSchedule, type));
		});
		return out.stream();
	}

	@ParameterizedTest(name = "{0}, saved {1}, restored {2}, {3}")
	@MethodSource("cases")
	void restoredSessionReadsTheUnbrokenLogits(PagedKvHandlerOracleTest.Family family, String saveSchedule,
			String restoreSchedule, String type) throws IOException {
		String what = family + ", saved " + saveSchedule + ", restored " + restoreSchedule + ", " + type;
		System.setProperty(KvPageSizeOptions.ENV, Integer.toString(PAGE_SIZE));
		System.setProperty(CacheTypeOptions.ENV_K, type);
		System.setProperty(CacheTypeOptions.ENV_V, type);
		ShardContext ctx = family.ctx();
		int vocab = ctx.vocabSize();
		int[] p = prompt(PROMPT, 0, vocab);
		KvSessionStamp stamp = new KvSessionStamp("model.gguf", "", "layers-0-" + ctx.endLayer(), p);
		Path file = tmp.resolve("s.kv");

		System.setProperty(ServeScheduleOptions.ENV, saveSchedule);
		ForwardPassHandler saver = family.build().apply(dir("a"));
		prefill(saver, ctx, "s", p);
		long written = saver.saveKv("s", stamp, file);
		float[] unbroken = decode(saver, ctx, "s", PROMPT, vocab);
		saver.evict("s");
		expectRefusal(() -> saver.saveKv("nobody", stamp, tmp.resolve("n.kv")), Reason.REFUSED, what + ", no KV");

		System.setProperty(ServeScheduleOptions.ENV, restoreSchedule);
		ForwardPassHandler restorer = family.build().apply(dir("b"));
		prefill(restorer, ctx, "computed", p);
		assertBitsEqual(decode(restorer, ctx, "computed", PROMPT, vocab), unbroken,
				what + ": a second handler from the same weights must compute the same logits, or this test cannot compare");
		int[] other = prompt(PROMPT, 30, vocab);
		prefill(restorer, ctx, "other", other);
		if (!(maxAbsDiff(decode(restorer, ctx, "other", PROMPT, vocab), unbroken) > 1e-4))
			throw new AssertionError(what + ": the KV does not reach the logits, so this test cannot fail");

		long read = restorer.restoreKv("t", stamp, file);
		if (read != written)
			throw new AssertionError(what + ": read " + read + " bytes, wrote " + written);
		assertBitsEqual(decode(restorer, ctx, "t", PROMPT, vocab), unbroken, what + ", decode after the restore");

		KvSessionStamp otherTokens = new KvSessionStamp("model.gguf", "", stamp.stage(), other);
		expectRefusal(() -> restorer.restoreKv("w", otherTokens, file), Reason.MISMATCH, what + ", other tokens");
		if (restorer.hostKvBytes("w") != 0)
			throw new AssertionError(what + ": a refused restore left KV behind");
		expectRefusal(() -> restorer.restoreKv("w", stamp, tmp.resolve("missing.kv")), Reason.NOT_FOUND,
				what + ", missing file");
		for (String id : new String[] { "t", "computed", "other" })
			restorer.evict(id);
	}

	private Path dir(String name) throws IOException {
		return Files.createDirectories(tmp.resolve(name));
	}

	private interface Call {
		Object run() throws IOException;
	}

	private static void expectRefusal(Call call, Reason reason, String what) throws IOException {
		try {
			call.run();
		} catch (KvSessionFileException e) {
			if (e.reason() != reason)
				throw new AssertionError(what + ": refused as " + e.reason() + ", expected " + reason, e);
			return;
		}
		throw new AssertionError(what + ": not refused");
	}

	private static void prefill(ForwardPassHandler h, ShardContext ctx, String id, int[] tokens) {
		h.forwardBatch(BatchForwardRequest.withTokens(id, Arrays.copyOfRange(tokens, 0, FIRST_WINDOW), 0), ctx);
		h.forwardBatch(BatchForwardRequest.withTokens(id, Arrays.copyOfRange(tokens, FIRST_WINDOW, tokens.length),
				FIRST_WINDOW), ctx);
	}

	private static float[] decode(ForwardPassHandler h, ShardContext ctx, String id, int pos, int vocab) {
		return h.forward(ForwardRequest.withTokens(id, new int[] { token(pos + 3, vocab) }, pos), ctx).logits();
	}

	private static double maxAbsDiff(float[] a, float[] b) {
		double m = 0;
		for (int i = 0; i < a.length; i++)
			m = Math.max(m, Math.abs(a[i] - b[i]));
		return m;
	}

	private static int token(int i, int vocab) {
		return (i * 7 + 3) % vocab;
	}

	private static int[] prompt(int n, int shift, int vocab) {
		int[] t = new int[n];
		for (int i = 0; i < n; i++)
			t[i] = token(i + shift, vocab);
		return t;
	}

	private static void assertBitsEqual(float[] actual, float[] expected, String what) {
		if (actual == null || expected == null)
			throw new AssertionError(what + ": no logits");
		if (actual.length != expected.length)
			throw new AssertionError(what + ": " + actual.length + " logits, expected " + expected.length);
		for (int i = 0; i < actual.length; i++)
			if (Float.floatToRawIntBits(actual[i]) != Float.floatToRawIntBits(expected[i]))
				throw new AssertionError(what + ": logit " + i + " is " + actual[i] + ", expected " + expected[i]);
	}

	private static void restoreProp(String key, String prev) {
		if (prev == null)
			System.clearProperty(key);
		else
			System.setProperty(key, prev);
	}
}
