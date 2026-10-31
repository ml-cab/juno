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
import static org.junit.jupiter.api.Assumptions.assumeFalse;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.List;
import java.util.stream.Stream;
import java.util.zip.ZipFile;

import javax.tools.JavaCompiler;
import javax.tools.ToolProvider;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class KitBootstrapTest {

	private static final Path SOURCE = Path.of("src/main/java/cab/ml/juno/kit/KitBootstrap.java");

	@Test
	void theBootstrapCompilesAgainstTheJdkAloneSoItRunsBeforeAnythingIsBuilt(@TempDir Path out) {
		// No class path and no source path: any reference outside the JDK and this one file fails to compile.
		JavaCompiler javac = ToolProvider.getSystemJavaCompiler();
		ByteArrayOutputStream err = new ByteArrayOutputStream();
		int rc = javac.run(null, null, err, "-proc:none", "-implicit:none", "-cp", "", "-sourcepath", "", "-d",
				out.toString(), SOURCE.toString());
		assertThat(rc).as(err.toString(StandardCharsets.UTF_8)).isZero();
	}

	@Test
	void theBootstrapSourceUsesNoSyntaxNewerThanJava17SoOldJdksReachTheVersionMessage(@TempDir Path out) {
		JavaCompiler javac = ToolProvider.getSystemJavaCompiler();
		ByteArrayOutputStream err = new ByteArrayOutputStream();
		int rc = javac.run(null, null, err, "-proc:none", "--release", "17", "-Xlint:-options", "-cp", "", "-d",
				out.toString(), SOURCE.toString());
		assertThat(rc).as(err.toString(StandardCharsets.UTF_8)).isZero();
	}

	@Test
	void anOldJdkIsRefusedWithInstructions() {
		assertThat(KitBootstrap.jdkProblem(25)).isNull();
		assertThat(KitBootstrap.jdkProblem(26)).isNull();
		assertThat(KitBootstrap.jdkProblem(21)).contains("25").contains("21");
	}

	@Test
	void aFailedBuildStillProducesAResultsZipHoldingTheBuildLog(@TempDir Path root) throws Exception {
		assumeFalse(System.getProperty("os.name").toLowerCase().contains("win"), "the fake Maven is a shell script");
		Path mvn = root.resolve("fake-mvn");
		Files.writeString(mvn, "#!/bin/sh\necho \"[INFO] Building juno-kit in " + System.getProperty("user.home")
				+ "/juno\"\necho \"[ERROR] COMPILATION ERROR\"\nexit 1\n");
		Files.setPosixFilePermissions(mvn, PosixFilePermissions.fromString("rwx------"));
		ByteArrayOutputStream console = new ByteArrayOutputStream();

		int rc = KitBootstrap.run(root, List.of("--yes"), List.of(mvn.toString()),
				new PrintStream(console, true, StandardCharsets.UTF_8));

		assertThat(rc).isEqualTo(1);
		Path zip;
		try (Stream<Path> s = Files.list(root.resolve("juno-kit-results"))) {
			zip = s.filter(p -> p.getFileName().toString().endsWith(".zip")).findFirst().orElseThrow();
		}
		assertThat(zip.getFileName().toString()).startsWith("juno-kit-results-");
		try (ZipFile z = new ZipFile(zip.toFile())) {
			String log = new String(z.getInputStream(z.getEntry("build.log")).readAllBytes(), StandardCharsets.UTF_8);
			assertThat(log).contains("COMPILATION ERROR").contains("<home>/juno")
					.doesNotContain(System.getProperty("user.home"));
			assertThat(z.getEntry("environment.txt")).isNotNull();
			String manifest = new String(z.getInputStream(z.getEntry("manifest.json")).readAllBytes(),
					StandardCharsets.UTF_8);
			assertThat(manifest).contains("\"kind\":\"build-failure\"");
		}
		assertThat(console.toString(StandardCharsets.UTF_8)).contains(zip.getFileName().toString());
	}
}
