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
import cab.ml.juno.kvcache.ServeScheduleOptions;

/**
 * A request started from another request's prefix ({@link ForwardPassHandler#shareKvPrefix}) reads, bit for bit,
 * the logits it reads when it computes that prefix itself, on every handler family, both schedules and both host KV
 * types; the source keeps its own rows, and the new request keeps the prefix after the source is evicted. The LoRA
 * handlers refuse. The shared length ends mid-page, so the new request's first write goes into a page both hold.
 */
@DisplayName("Shared KV prefix - a request started from another's prefix reads the logits it computes itself")
class KvPrefixShareHandlerTest {

	private static final int PAGE_SIZE = 4;
	private static final int PROMPT = 11;
	private static final int SHARED = 7;

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
			for (String schedule : new String[] { "static", "continuous" })
				for (String type : new String[] { "f16", "q8_0" })
					out.add(Arguments.of(f, schedule, type));
		});
		return out.stream();
	}

	@ParameterizedTest(name = "{0}, {1}, {2}")
	@MethodSource("cases")
	void sharedPrefixReadsTheLogitsOfAComputedOne(PagedKvHandlerOracleTest.Family family, String schedule,
			String type) throws IOException {
		System.setProperty(ServeScheduleOptions.ENV, schedule);
		System.setProperty(KvPageSizeOptions.ENV, Integer.toString(PAGE_SIZE));
		System.setProperty(CacheTypeOptions.ENV_K, type);
		System.setProperty(CacheTypeOptions.ENV_V, type);
		Path dir = Files.createDirectories(tmp.resolve(family.name().replaceAll("[^a-z0-9]", "_")));
		ForwardPassHandler h = family.build().apply(dir);
		ShardContext ctx = family.ctx();
		int vocab = ctx.vocabSize();
		String what = family + ", " + schedule + ", " + type;

		int[] p = prompt(PROMPT, 0, vocab); // the source's prompt
		int[] x = prompt(PROMPT, 20, vocab); // the new request's: p's first SHARED tokens, then its own
		System.arraycopy(p, 0, x, 0, SHARED);
		prefill(h, ctx, "src", p);

		if (family.name().contains("LoRA")) {
			if (h.supportsKvPrefixSharing() || h.shareKvPrefix("src", "x", SHARED))
				throw new AssertionError(what + ": a LoRA handler shared a prefix");
			return;
		}
		if (!h.supportsKvPrefixSharing())
			throw new AssertionError(what + ": sharing not supported");
		if (h.shareKvPrefix("src", "y", PROMPT + 1) || h.shareKvPrefix("nobody", "y", 3)
				|| h.shareKvPrefix("src", "src", 3))
			throw new AssertionError(what + ": shared from a source too short, absent, or itself");

		float[] computed = prefill(h, ctx, "ref", x);
		int[] other = x.clone();
		for (int i = 0; i < SHARED; i++)
			other[i] = token(i + 40, vocab);
		float[] otherPrefix = prefill(h, ctx, "other", other);
		if (!(maxAbsDiff(computed, otherPrefix) > 1e-4))
			throw new AssertionError(what + ": the prefix does not reach the logits, so this test cannot fail");

		if (!h.shareKvPrefix("src", "x", SHARED))
			throw new AssertionError(what + ": share refused");
		float[] shared = h.forwardBatch(BatchForwardRequest.withTokens("x", Arrays.copyOfRange(x, SHARED, PROMPT),
				SHARED), ctx).lastLogits();
		assertBitsEqual(shared, computed, what + ", prefill after the shared prefix");

		// The source is unchanged by the new request's write into the page they share.
		prefill(h, ctx, "src2", p);
		assertBitsEqual(decode(h, ctx, "src", PROMPT, vocab), decode(h, ctx, "src2", PROMPT, vocab),
				what + ", the source's next token");

		// The new request keeps the prefix once the source is gone.
		h.evict("src");
		assertBitsEqual(decode(h, ctx, "x", PROMPT, vocab), decode(h, ctx, "ref", PROMPT, vocab),
				what + ", decode after the source is evicted");
		for (String id : new String[] { "x", "ref", "other", "src2" })
			h.evict(id);
	}

	/** Prefills in the same two windows every request here uses, so rows are computed alike. */
	private static float[] prefill(ForwardPassHandler h, ShardContext ctx, String id, int[] tokens) {
		h.forwardBatch(BatchForwardRequest.withTokens(id, Arrays.copyOfRange(tokens, 0, SHARED), 0), ctx);
		return h.forwardBatch(BatchForwardRequest.withTokens(id, Arrays.copyOfRange(tokens, SHARED, tokens.length),
				SHARED), ctx).lastLogits();
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
