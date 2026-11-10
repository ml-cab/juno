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

import java.io.ByteArrayOutputStream;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.zip.ZipFile;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Packing logs a Windows probe JVM wrote in the machine's code page. On Windows the JVM writes a redirected stdout and
 * stderr in {@code native.encoding}, so the engine's em dash arrives as the single byte 0x97, which is not UTF-8.
 */
class ResultsPackerTest {

	private static final Charset CP1252 = Charset.forName("windows-1252");
	private static final KitBootstrap.Privacy WINDOWS = new KitBootstrap.Privacy("Jos\u00e9", "C:\\Users\\Jos\u00e9",
			List.of("JOSE-PC"));

	@Test
	void aLogThatIsNotUtf8IsPackedAndTheFilesAfterItAreToo(@TempDir Path tmp) throws Exception {
		Path run = Files.createDirectories(tmp.resolve("run"));
		Files.createDirectories(run.resolve("probes"));
		ByteArrayOutputStream log = new ByteArrayOutputStream();
		log.write("INFO: GGUF v3 ".getBytes(StandardCharsets.US_ASCII));
		log.write(0x97);
		log.write(" tensors=201\n".getBytes(StandardCharsets.US_ASCII));
		Files.write(run.resolve("probes").resolve("cpu.live.tinyllama.log"), log.toByteArray());
		Files.writeString(run.resolve("SUMMARY.txt"), "summary\n");
		Files.writeString(run.resolve("report.html"), "<html></html>\n");

		Path zip = ResultsPacker.pack(run, tmp.resolve("out.zip"), WINDOWS);

		try (ZipFile z = new ZipFile(zip.toFile())) {
			String packed = new String(z.getInputStream(z.getEntry("probes/cpu.live.tinyllama.log")).readAllBytes(),
					StandardCharsets.UTF_8);
			assertThat(packed).contains("INFO: GGUF v3 ").contains(" tensors=201");
			assertThat(z.getEntry("SUMMARY.txt")).isNotNull();
			assertThat(z.getEntry("report.html")).isNotNull();
		}
	}

	@Test
	void aHomePathInTheCodePageIsStillScrubbed() {
		byte[] log = "Loading C:\\Users\\Jos\u00e9\\.cache\\juno\\model.gguf on JOSE-PC \u2014 done\n".getBytes(CP1252);

		String out = WINDOWS.scrub(KitBootstrap.decode(log, CP1252));

		assertThat(out).contains("<home>\\.cache\\juno\\model.gguf on <host> \u2014 done").doesNotContain("Jos", "JOSE");
	}

	@Test
	void utf8TextStaysUtf8WhateverTheCodePage() {
		byte[] log = "GGUF v3 \u2014 tensors=201 \u0414\u0438\u043c\u0430\n".getBytes(StandardCharsets.UTF_8);

		assertThat(KitBootstrap.decode(log, CP1252)).isEqualTo("GGUF v3 \u2014 tensors=201 \u0414\u0438\u043c\u0430\n");
	}

	@Test
	void bytesNoCharsetCanReadStillDecodeWithoutThrowing() {
		byte[] log = { 'o', 'k', ' ', (byte) 0x81, (byte) 0xFF, '\n' };

		assertThat(KitBootstrap.decode(log, StandardCharsets.UTF_8)).startsWith("ok ").endsWith("\n");
	}

	@Test
	void readTextReadsAFileTheSameWay(@TempDir Path tmp) throws Exception {
		Path f = tmp.resolve("a.log");
		Files.write(f, new byte[] { 'a', (byte) 0x97, 'b', '\n' });

		assertThat(KitBootstrap.readText(f)).startsWith("a").endsWith("b\n");
	}
}
