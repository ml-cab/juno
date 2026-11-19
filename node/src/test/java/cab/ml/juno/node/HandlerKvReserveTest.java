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

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import cab.ml.juno.kvcache.CacheTypeOptions;
import cab.ml.juno.kvcache.ServeScheduleOptions;

/**
 * {@link ForwardPassHandler#reserveKv} on every handler family that keeps host KV (the LoRA handlers included): a
 * request told the most positions it can write holds a dense host KV of exactly that many positions
 * ({@link ForwardPassHandler#hostKvBytes}), where an untold one doubles past it, whether the reservation comes before
 * its first forward pass or after its layers exist; an evicted request's reservation is gone; {@code f32} rows take
 * twice the bytes of {@code f16} rows; and the paged KV of the continuous schedule, which grows a page at a time, is
 * the same with or without a reservation.
 */
@DisplayName("ForwardPassHandler.reserveKv - dense host KV grows toward the request's length")
class HandlerKvReserveTest {

	/** A prompt past the dense tensor's first 64 positions, so an untold request doubles to 128. */
	private static final int PROMPT = 70;
	private static final int DECODED = 3;
	private static final int POSITIONS = PROMPT + DECODED;

	private final String prevSchedule = System.getProperty(ServeScheduleOptions.ENV);
	private final String prevTypeK = System.getProperty(CacheTypeOptions.ENV_K);
	private final String prevTypeV = System.getProperty(CacheTypeOptions.ENV_V);

	@TempDir
	Path tmp;

	@AfterEach
	void restore() {
		restoreProp(ServeScheduleOptions.ENV, prevSchedule);
		restoreProp(CacheTypeOptions.ENV_K, prevTypeK);
		restoreProp(CacheTypeOptions.ENV_V, prevTypeV);
	}

	static java.util.stream.Stream<PagedKvHandlerOracleTest.Family> families() {
		return PagedKvHandlerOracleTest.families();
	}

	@ParameterizedTest(name = "{0}")
	@MethodSource("families")
	@DisplayName("static schedule: exactly the reserved positions, before or after the layers; doubling without")
	void denseGrowsToTheReservation(PagedKvHandlerOracleTest.Family family) {
		ForwardPassHandler h = handler(family, "static", "f16");
		ShardContext ctx = family.ctx();

		run(h, ctx, "untold", 0, PROMPT);
		long untold = h.hostKvBytes("untold");
		assertThat(untold).as("an untold request's dense KV").isPositive();

		h.reserveKv("before", POSITIONS);
		run(h, ctx, "before", 0, PROMPT);
		long before = h.hostKvBytes("before");
		assertThat(before * 128).as("reserved before the first pass: %d positions, not 128", POSITIONS)
				.isEqualTo(untold * POSITIONS);

		h.forwardBatch(BatchForwardRequest.withTokens("after", tokens(0, 10), 0), ctx);
		h.reserveKv("after", POSITIONS);
		run(h, ctx, "after", 10, PROMPT - 10);
		assertThat(h.hostKvBytes("after")).as("reserved once the layers exist").isEqualTo(before);

		h.reserveKv("gone", POSITIONS);
		h.evict("gone");
		run(h, ctx, "gone", 0, PROMPT);
		assertThat(h.hostKvBytes("gone")).as("an evicted request's reservation is dropped").isEqualTo(untold);
		assertThat(h.hostKvBytes("never")).isZero();
	}

	@ParameterizedTest(name = "{0}")
	@MethodSource("families")
	@DisplayName("f32 rows take twice the bytes of f16 rows")
	void f32IsTwiceF16(PagedKvHandlerOracleTest.Family family) {
		ForwardPassHandler half = handler(family, "static", "f16");
		half.reserveKv("r", POSITIONS);
		run(half, family.ctx(), "r", 0, PROMPT);
		ForwardPassHandler full = handler(family, "static", "f32");
		full.reserveKv("r", POSITIONS);
		run(full, family.ctx(), "r", 0, PROMPT);
		assertThat(full.hostKvBytes("r")).isEqualTo(2 * half.hostKvBytes("r"));
	}

	@ParameterizedTest(name = "{0}")
	@MethodSource("families")
	@DisplayName("continuous schedule: paged KV is the same with or without a reservation")
	void pagedIgnoresTheReservation(PagedKvHandlerOracleTest.Family family) {
		ForwardPassHandler h = handler(family, "continuous", "f16");
		run(h, family.ctx(), "untold", 0, PROMPT);
		h.reserveKv("told", POSITIONS);
		run(h, family.ctx(), "told", 0, PROMPT);
		assertThat(h.hostKvBytes("told")).isEqualTo(h.hostKvBytes("untold")).isPositive();
	}

	/** Prefills {@code n} tokens from {@code start} in one window, then decodes {@link #DECODED} after the prompt. */
	private static void run(ForwardPassHandler h, ShardContext ctx, String id, int start, int n) {
		h.forwardBatch(BatchForwardRequest.withTokens(id, tokens(start, n), start), ctx);
		for (int i = 0; i < DECODED; i++)
			h.forward(ForwardRequest.withTokens(id, new int[] { (7 * i + 3) % 50 }, start + n + i), ctx);
	}

	private ForwardPassHandler handler(PagedKvHandlerOracleTest.Family family, String schedule, String type) {
		System.setProperty(ServeScheduleOptions.ENV, schedule);
		System.setProperty(CacheTypeOptions.ENV_K, type);
		System.setProperty(CacheTypeOptions.ENV_V, type);
		try {
			return family.build().apply(Files.createDirectories(tmp.resolve(family.name().replaceAll("[^a-z0-9]", "_")
					+ "-" + schedule + "-" + type + "-" + System.nanoTime())));
		} catch (IOException e) {
			throw new UncheckedIOException(e);
		}
	}

	private static int[] tokens(int start, int n) {
		int[] t = new int[n];
		for (int i = 0; i < n; i++)
			t[i] = (start + i) * 13 % 50 + 1;
		return t;
	}

	private static void restoreProp(String key, String prev) {
		if (prev == null)
			System.clearProperty(key);
		else
			System.setProperty(key, prev);
	}
}
