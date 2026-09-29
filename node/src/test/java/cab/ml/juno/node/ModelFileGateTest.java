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

import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.nio.file.Path;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The model-file gate every entry point runs before it reads anything else from a file, so an
 * architecture Juno has no handler for is the first reason given, ahead of any tokenizer or
 * config error the same file would also raise.
 */
class ModelFileGateTest {

	@TempDir
	Path dir;

	@Test
	@DisplayName("architectures with no verified handler are refused by name")
	void unverifiedArchitectures_refusedByName() throws IOException {
		// The four real files on disk that declare these also declare pre-tokenizer types
		// Juno does not implement, which is why the architecture has to be checked first.
		for (String arch : new String[] { "qwen35", "gemma4", "mistral3", "minimax-m2" }) {
			Path gguf = MetadataOnlyGguf.write(dir, arch);
			assertThatThrownBy(() -> ModelFileGate.requireLoadable(gguf))
					.isInstanceOf(UnsupportedModelException.class)
					.hasMessageContaining("Unsupported model architecture '" + arch + "'");
		}
	}

	@Test
	@DisplayName("every architecture the handler loader dispatches passes")
	void supportedArchitectures_pass() throws IOException {
		for (String arch : new String[] { "llama", "mistral", "qwen2", "phi2", "phi3", "qwen3", "qwen3moe" }) {
			Path gguf = MetadataOnlyGguf.write(dir, arch);
			assertThatCode(() -> ModelFileGate.requireLoadable(gguf)).doesNotThrowAnyException();
		}
	}

	@Test
	@DisplayName("the declared value is compared the way the handler loader reads it")
	void declaredValue_isNormalizedBeforeTheCheck() throws IOException {
		assertThatCode(() -> ModelFileGate.requireLoadable(MetadataOnlyGguf.write(dir, " LLaMA ")))
				.doesNotThrowAnyException();
		Path gemma = MetadataOnlyGguf.write(dir, "Gemma4");
		assertThatThrownBy(() -> ModelFileGate.requireLoadable(gemma))
				.hasMessageContaining("'gemma4'");
	}

	@Test
	@DisplayName("a file that cannot be read is an I/O failure, not a refused model")
	void unreadableFile_isNotReportedAsUnsupported() {
		Path missing = dir.resolve("absent.gguf");
		assertThatThrownBy(() -> ModelFileGate.requireLoadable(missing))
				.isInstanceOf(IOException.class)
				.isNotInstanceOf(UnsupportedModelException.class);
	}
}
