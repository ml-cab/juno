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

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Written-prefix watermark for {@link DeviceKvCache}, which is what keeps a
 * mirror that is out of step with the host KV tensors from being read as
 * history by the GPU attention kernel.
 *
 * <p>Device memory is never zeroed, so an unwritten position holds whatever the
 * allocator last left there — on-device it is indistinguishable from a real K/V
 * row, and attention over it produces logits that are noise while every other
 * signal (token counts, timings, no exception) still looks healthy. The
 * watermark makes that state observable host-side so the handler can fall back
 * to the CPU tensors instead.
 *
 * <p>Run: {@code mvn test -Dgroups=gpu -pl node -Dtest=DeviceKvMirrorWatermarkTest}.
 */
@Tag("gpu")
@DisplayName("DeviceKvCache — written-prefix watermark gates attention reads")
class DeviceKvMirrorWatermarkTest {

	private static final int KV_DIM = 16;

	private static GpuContext ctx;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping — no CUDA device");
		ctx = GpuContext.init(0);
	}

	@AfterAll
	static void destroy() {
		if (ctx != null)
			ctx.close();
	}

	@Test
	@DisplayName("a freshly allocated mirror is readable through nothing")
	void fresh_mirror_is_not_readable_as_history() {
		DeviceKvCache kv = new DeviceKvCache(ctx, KV_DIM);
		try {
			// The regression this guards: a mirror given up mid-request used to be
			// unmapped, so the next token allocated a replacement and attention read
			// the whole conversation out of it -- all of it uninitialized.
			assertThat(kv.validTokens()).isZero();
			assertThat(kv.readableThrough(1)).isFalse();
			assertThat(kv.readableThrough(64)).isFalse();
		} finally {
			kv.close();
		}
	}

	@Test
	@DisplayName("sequential appends advance the watermark exactly one position at a time")
	void sequential_appends_advance_watermark() {
		DeviceKvCache kv = new DeviceKvCache(ctx, KV_DIM);
		try {
			for (int pos = 0; pos < 10; pos++) {
				kv.appendToken(pos, row(pos), row(pos));
				assertThat(kv.validTokens()).isEqualTo(pos + 1);
				assertThat(kv.readableThrough(pos + 1)).isTrue();
				assertThat(kv.readableThrough(pos + 2)).isFalse();
			}
		} finally {
			kv.close();
		}
	}

	@Test
	@DisplayName("a write past the end leaves a hole and does not advance the watermark")
	void hole_does_not_advance_watermark() {
		DeviceKvCache kv = new DeviceKvCache(ctx, KV_DIM);
		try {
			kv.appendToken(0, row(0), row(0));
			kv.appendToken(1, row(1), row(1));
			// Position 2 skipped: this is the shape of a request whose prefix was
			// restored into the host tensors only, or one that missed appends while
			// another request's mirror was being retired.
			kv.appendToken(3, row(3), row(3));

			assertThat(kv.validTokens()).isEqualTo(2);
			assertThat(kv.readableThrough(2)).isTrue();
			assertThat(kv.readableThrough(4)).isFalse();

			// Filling the gap in order makes the prefix contiguous again.
			kv.appendToken(2, row(2), row(2));
			assertThat(kv.validTokens()).isEqualTo(3);
		} finally {
			kv.close();
		}
	}

	@Test
	@DisplayName("growth preserves the watermark along with the rows it covers")
	void growth_preserves_watermark() {
		DeviceKvCache kv = new DeviceKvCache(ctx, KV_DIM);
		try {
			int past = DeviceKvCache.INITIAL_SEQ_CAPACITY + 5;
			for (int pos = 0; pos < past; pos++)
				kv.appendToken(pos, row(pos), row(pos));

			assertThat(kv.capacityTokens()).isGreaterThan(DeviceKvCache.INITIAL_SEQ_CAPACITY);
			assertThat(kv.validTokens()).isEqualTo(past);
			assertThat(kv.readableThrough(past)).isTrue();
		} finally {
			kv.close();
		}
	}

	@Test
	@DisplayName("a retired mirror is neither live nor readable, however much it holds")
	void retired_mirror_is_not_readable() {
		DeviceKvCache kv = new DeviceKvCache(ctx, KV_DIM);
		for (int pos = 0; pos < 8; pos++)
			kv.appendToken(pos, row(pos), row(pos));
		assertThat(kv.readableThrough(8)).isTrue();

		kv.close();

		assertThat(kv.live()).isFalse();
		assertThat(kv.readableThrough(8)).isFalse();
		assertThat(kv.readableThrough(1)).isFalse();
	}

	private static float[] row(int pos) {
		float[] out = new float[KV_DIM];
		for (int i = 0; i < KV_DIM; i++)
			out[i] = pos + i / 100.0f;
		return out;
	}
}
