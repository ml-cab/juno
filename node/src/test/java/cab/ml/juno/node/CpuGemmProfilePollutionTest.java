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
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.Test;

/**
 * The batched matmul allocates nothing per call at prefill width after another vector shape has run through the
 * JDK's Vector API methods in the same JVM.
 *
 * <p>Those methods keep one type profile for every caller. A Vector API call on a vector the kernel carries from one
 * loop iteration to the next is bound through that profile; once a second shape had been profiled there, C2 boxed the
 * kernel's accumulators and every vector operation allocated (about 8.5 GB per call here, and in a server the prefill
 * ran 2x to 5.5x slower in some JVMs and not others). Any Vector API code at another width can be the second shape,
 * such as the float kernels at {@code SPECIES_PREFERRED} on a 256-bit host. The check runs in a child JVM because the
 * profiles it pollutes are shared with every later test.
 */
class CpuGemmProfilePollutionTest {

	@Test
	void batchedMatmulAllocatesNothingAfterAnotherShapeRan() throws IOException, InterruptedException {
		List<String> cmd = List.of(ProcessHandle.current().info().command().orElse("java"), "--add-modules",
				"jdk.incubator.vector", "-Xmx2g", "-cp", System.getProperty("java.class.path"),
				CpuGemmProfilePollutionMain.class.getName());
		Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
		p.getOutputStream().close();
		String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
		assertThat(p.waitFor(600, TimeUnit.SECONDS)).as("child JVM finished").isTrue();
		assertThat(p.exitValue()).as("child JVM exit code; output:%n%s", out).isZero();
		String line = out.lines().filter(l -> l.startsWith("bytes/call ")).findFirst().orElse(null);
		assertThat(line).as("the child's reading; output:%n%s", out).isNotNull();
		// One boxed vector per sub-block of one row is already megabytes per call at this size; the clean kernel's
		// bookkeeping is about a kilobyte.
		assertThat(Long.parseLong(line.substring("bytes/call ".length())))
				.as("bytes allocated per batched matmul call, all threads, after another vector shape ran")
				.isLessThan(64 * 1024);
	}
}
