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
package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class NodeHeapSizeTest {

	private static final long GIB = 1024L * 1024 * 1024;

	@TempDir
	Path tmp;

	@Test
	void derives_one_and_a_half_times_the_file_plus_two_gib_rounded_up() throws IOException {
		// a Mistral 7B Q4_K_M file: 4,368,440,864 bytes = 4.07 GiB; * 1.5 + 2 = 8.10 -> 9g
		assertThat(NodeHeapSize.forFileBytes(4_368_440_864L)).isEqualTo("9g");
		assertThat(NodeHeapSize.forFileBytes(4 * GIB)).isEqualTo("8g");
	}

	@Test
	void small_model_gets_the_four_gib_floor() {
		// TinyLlama Q4_K_M, 668,788,096 bytes -> 2.93 rounded up to 3, floored at 4
		assertThat(NodeHeapSize.forFileBytes(668_788_096L)).isEqualTo("4g");
	}

	@Test
	void huge_model_is_capped_at_forty_eight_gib() {
		assertThat(NodeHeapSize.forFileBytes(40 * GIB)).isEqualTo("48g");
	}

	@Test
	void explicit_property_wins_over_the_model_size() throws IOException {
		Path model = sparseFile(6 * GIB);
		assertThat(NodeHeapSize.resolve("5g", model.toString())).isEqualTo("5g");
	}

	@Test
	void unset_property_derives_from_the_model_file() throws IOException {
		Path model = sparseFile(6 * GIB);
		assertThat(NodeHeapSize.resolve(null, model.toString())).isEqualTo("11g");
		assertThat(NodeHeapSize.resolve("  ", model.toString())).isEqualTo("11g");
	}

	@Test
	void stub_mode_or_missing_file_falls_back_to_four_gib() {
		assertThat(NodeHeapSize.resolve(null, null)).isEqualTo("4g");
		assertThat(NodeHeapSize.resolve(null, tmp.resolve("absent.gguf").toString())).isEqualTo("4g");
	}

	private Path sparseFile(long bytes) throws IOException {
		Path p = tmp.resolve("model.gguf");
		try (RandomAccessFile f = new RandomAccessFile(p.toFile(), "rw")) {
			f.setLength(bytes);
		}
		return p;
	}
}
