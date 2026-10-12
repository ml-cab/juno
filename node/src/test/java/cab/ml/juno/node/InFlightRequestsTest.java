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

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("InFlightRequests - a request is idle only when none of its passes runs")
class InFlightRequestsTest {

	@Test
	@DisplayName("nested and batched passes keep a request busy until the last one exits")
	void countsNestedAndBatchedPasses() {
		InFlightRequests r = new InFlightRequests();
		assertThat(r.idle("a")).isTrue();
		r.enter("a");
		r.enterAll(List.of("a", "b"));
		assertThat(r.idle("a")).isFalse();
		assertThat(r.idle("b")).isFalse();
		r.exitAll(List.of("a", "b"));
		assertThat(r.idle("a")).isFalse();
		assertThat(r.idle("b")).isTrue();
		r.exit("a");
		assertThat(r.idle("a")).isTrue();
		r.exit("a"); // an exit with nothing entered changes nothing
		assertThat(r.idle("a")).isTrue();
	}

	@Test
	@DisplayName("concurrent passes of one request leave it idle exactly when all have exited")
	void concurrentPasses() throws Exception {
		InFlightRequests r = new InFlightRequests();
		ExecutorService pool = Executors.newFixedThreadPool(8);
		CountDownLatch done = new CountDownLatch(8);
		for (int t = 0; t < 8; t++)
			pool.submit(() -> {
				for (int i = 0; i < 10_000; i++) {
					r.enter("x");
					r.exit("x");
				}
				done.countDown();
			});
		r.enter("x");
		assertThat(done.await(60, TimeUnit.SECONDS)).isTrue();
		assertThat(r.idle("x")).isFalse();
		r.exit("x");
		assertThat(r.idle("x")).isTrue();
		pool.shutdown();
	}
}
