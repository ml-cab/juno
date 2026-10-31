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
package cab.ml.juno.kit;


import static org.assertj.core.api.Assertions.assertThat;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** {@code --plan} and {@code --list} write, download and build nothing. */
class PlanTest {

	/** Every catalog model as a sparse file of the right size, so nothing is missing and nothing is fetched. */
	private static Path models(Path dir) throws Exception {
		Path m = Files.createDirectories(dir.resolve("models"));
		for (CatalogModel c : CatalogModel.ALL)
			try (RandomAccessFile f = new RandomAccessFile(m.resolve(c.file()).toFile(), "rw")) {
				f.setLength(c.bytes());
			}
		return m;
	}

	@Test
	void planShowsTheRunAndWritesNothing(@TempDir Path tmp) throws Exception {
		Path models = models(tmp);
		Path out = tmp.resolve("results");
		Path cache = tmp.resolve("cache");
		ByteArrayOutputStream console = new ByteArrayOutputStream();

		int rc = KitMain.run(List.of("--plan", "--suite", "quick", "--models-dir", models.toString(), "--cache-dir",
				cache.toString(), "--out", out.toString()), new PrintStream(console, true, StandardCharsets.UTF_8),
				new ByteArrayInputStream(new byte[0]));

		String text = console.toString(StandardCharsets.UTF_8);
		assertThat(rc).isZero();
		assertThat(text).contains("Plan for suite quick").contains("live.tinyllama").contains("Downloads: 0 MiB");
		assertThat(out).doesNotExist();
		assertThat(cache).doesNotExist();
		try (var s = Files.list(tmp)) {
			assertThat(s.map(p -> p.getFileName().toString()).toList()).containsExactly("models");
		}
	}

	@Test
	void listNamesEveryProbeAndModel(@TempDir Path tmp) throws Exception {
		Path models = models(tmp);
		ByteArrayOutputStream console = new ByteArrayOutputStream();
		int rc = KitMain.run(List.of("--list", "--models-dir", models.toString(), "--cache-dir",
				tmp.resolve("cache").toString(), "--out", tmp.resolve("results").toString()),
				new PrintStream(console, true, StandardCharsets.UTF_8), new ByteArrayInputStream(new byte[0]));
		String text = console.toString(StandardCharsets.UTF_8);
		assertThat(rc).isZero();
		for (Probe p : ProbeCatalog.all())
			assertThat(text).contains(p.id());
		for (CatalogModel m : CatalogModel.ALL)
			assertThat(text).contains(m.id());
		assertThat(tmp.resolve("results")).doesNotExist();
	}
}
