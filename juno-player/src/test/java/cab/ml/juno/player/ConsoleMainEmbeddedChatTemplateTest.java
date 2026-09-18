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

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.ChatTemplateFormatter;
import cab.ml.juno.tokenizer.EmbeddedChatTemplateRegistry;
import cab.ml.juno.tokenizer.GgufChatTemplateResolver;

/**
 * Regression test for the {@code ConsoleMain.registerEmbeddedChatTemplate}
 * bug fixed alongside this test: when a GGUF carries no
 * {@code tokenizer.chat_template} metadata (e.g. the moondream2/phi-2
 * llamafile vision fixture), the method used to register the *named
 * fallback* into {@link EmbeddedChatTemplateRegistry} under the model's raw
 * filename anyway. Because {@link ChatTemplateFormatter#forModelType}
 * consults the registry before falling through to
 * {@code ChatTemplate.forModelType}'s filename-substring matching, that
 * wrongly-registered fallback (resolved from {@code ChatModelType.fromPath},
 * which has no moondream/phi2 case and defaults to "chatml") permanently
 * shadowed the correct {@code moondream} template for the rest of the
 * process — every vision chat request got ChatML-formatted prompts instead
 * of the raw Q&A format moondream2/phi-2 was fine-tuned on, so the model's
 * first sampled token was immediately EOS (empty completions in production).
 */
class ConsoleMainEmbeddedChatTemplateTest {

	@AfterEach
	void clearRegistry() {
		EmbeddedChatTemplateRegistry.clear();
	}

	@Test
	void absent_embedded_template_does_not_shadow_filename_substring_fallback(@TempDir Path tmp) throws Exception {
		Path gguf = writeMinimalGguf(tmp, "moondream2-q5_k.llamafile", Map.of());

		invokeRegisterEmbeddedChatTemplate(gguf.toString());

		String filenameKey = gguf.getFileName().toString();
		assertThat(EmbeddedChatTemplateRegistry.lookup(filenameKey))
				.as("no embedded template exists — nothing should be registered, so downstream lookups fall "
						+ "through to filename-substring resolution")
				.isNull();

		String formatted = ChatTemplateFormatter.forModelType(filenameKey)
				.format(List.of(ChatMessage.user("What is in this image?")));
		assertThat(formatted).as("must still resolve to the moondream Q&A template via substring match")
				.startsWith("Question: What is in this image?").endsWith("\n\nAnswer:");
	}

	@Test
	void present_embedded_template_still_registers_under_both_keys(@TempDir Path tmp) throws Exception {
		String chatml = "{% for message in messages %}" + "{{ '<|im_start|>' + message['role'] + '\\n' "
				+ "+ message['content'] + '<|im_end|>' + '\\n' }}" + "{% endfor %}"
				+ "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";
		Path gguf = writeMinimalGguf(tmp, "some-chatml-model.gguf", Map.of(GgufChatTemplateResolver.METADATA_KEY, chatml));

		invokeRegisterEmbeddedChatTemplate(gguf.toString());

		String filenameKey = gguf.getFileName().toString();
		assertThat(EmbeddedChatTemplateRegistry.lookup(filenameKey))
				.as("a real embedded template must still be published for downstream call sites").isNotNull();
		assertThat(EmbeddedChatTemplateRegistry.lookup("chatml"))
				.as("...under the short ChatModelType key too").isNotNull();
	}

	// ── Reflection + minimal synthetic GGUF (same wire format as
	// cab.ml.juno.tokenizer.GgufChatTemplateResolverTest) ──────────────────────

	private static void invokeRegisterEmbeddedChatTemplate(String modelPath) throws Exception {
		Method m = ConsoleMain.class.getDeclaredMethod("registerEmbeddedChatTemplate", String.class);
		m.setAccessible(true);
		m.invoke(null, modelPath);
	}

	private static Path writeMinimalGguf(Path dir, String name, Map<String, String> stringMeta) throws IOException {
		Path file = dir.resolve(name);
		ByteArrayOutputStream out = new ByteArrayOutputStream();
		out.write("GGUF".getBytes(StandardCharsets.US_ASCII));
		writeInt32LE(out, 3); // version
		writeUInt64LE(out, 0); // tensor_count
		writeUInt64LE(out, stringMeta.size()); // metadata_kv_count
		for (Map.Entry<String, String> e : stringMeta.entrySet()) {
			writeGgufString(out, e.getKey());
			writeInt32LE(out, 8); // GGUF_METADATA_VALUE_TYPE_STRING
			writeGgufString(out, e.getValue());
		}
		Files.write(file, out.toByteArray());
		return file;
	}

	private static void writeInt32LE(ByteArrayOutputStream out, int v) throws IOException {
		out.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array());
	}

	private static void writeUInt64LE(ByteArrayOutputStream out, long v) throws IOException {
		out.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(v).array());
	}

	private static void writeGgufString(ByteArrayOutputStream out, String s) throws IOException {
		byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
		writeUInt64LE(out, bytes.length);
		out.write(bytes);
	}
}
