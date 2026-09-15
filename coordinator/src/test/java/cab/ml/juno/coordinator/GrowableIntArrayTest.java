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
package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class GrowableIntArrayTest {

	@Test
	void empty_buffer_has_zero_size_and_empty_array() {
		GrowableIntArray buf = new GrowableIntArray();
		assertThat(buf.size()).isZero();
		assertThat(buf.toTrimmedArray()).isEmpty();
	}

	@Test
	void appended_values_are_returned_in_order() {
		GrowableIntArray buf = new GrowableIntArray();
		buf.append(10);
		buf.append(20);
		buf.append(30);
		assertThat(buf.size()).isEqualTo(3);
		assertThat(buf.toTrimmedArray()).containsExactly(10, 20, 30);
	}

	@Test
	void toTrimmedArray_is_a_defensive_copy() {
		GrowableIntArray buf = new GrowableIntArray();
		buf.append(1);
		buf.append(2);
		int[] snapshot = buf.toTrimmedArray();
		snapshot[0] = 999;
		buf.append(3);
		assertThat(buf.toTrimmedArray()).containsExactly(1, 2, 3);
	}

	@Test
	void grows_past_default_initial_capacity() {
		// Default constructor starts at capacity 16 — no existing coordinator test
		// (GenerationLoopTest, RequestSchedulerContinuousTest) generates more than 8
		// tokens per slot, so without this test the append() doubling branch
		// (buf.length * 2) was never exercised at all.
		GrowableIntArray buf = new GrowableIntArray();
		int count = 100;
		for (int i = 0; i < count; i++) {
			buf.append(i);
		}
		assertThat(buf.size()).isEqualTo(count);
		int[] result = buf.toTrimmedArray();
		assertThat(result).hasSize(count);
		for (int i = 0; i < count; i++) {
			assertThat(result[i]).isEqualTo(i);
		}
	}

	@Test
	void grows_past_explicit_small_initial_capacity() {
		GrowableIntArray buf = new GrowableIntArray(1);
		for (int i = 0; i < 5; i++) {
			buf.append(i * 2);
		}
		assertThat(buf.toTrimmedArray()).containsExactly(0, 2, 4, 6, 8);
	}
}
