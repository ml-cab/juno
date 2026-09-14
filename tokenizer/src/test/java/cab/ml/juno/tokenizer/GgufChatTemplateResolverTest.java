package cab.ml.juno.tokenizer;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.api.io.TempDir;

import cab.ml.juno.node.GgufReader;

/**
 * Tests {@link GgufChatTemplateResolver}'s precedence: GGUF-embedded
 * {@code tokenizer.chat_template} (rendered via the restricted
 * {@link MiniJinjaTemplate} engine) when usable, named {@link ChatTemplate}
 * fallback otherwise — the "Spot check" item from
 * {@code docs/infra-plan/PLAN-Infra-Tier7.md}'s exit checklist.
 *
 * <p>Fixture-based tests below write minimal synthetic GGUF files (metadata
 * only, zero tensors) carrying adapted Llama-3 / Phi-3 / ChatML-style
 * {@code chat_template} strings. The {@code @EnabledIf}-gated tests at the
 * bottom additionally spot-check against real GGUF fixtures under
 * {@code models/} when present (Phi-3.5-mini, TinyLlama both carry a real
 * embedded {@code tokenizer.chat_template} in this repo's local test
 * fixtures) — see {@code GgufTokenizerBosTest} for the same pattern.
 */
class GgufChatTemplateResolverTest {

	private static final String LLAMA3_LIKE = """
			{{- bos_token }}
			{%- for message in messages %}
			{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + (message['content'] | trim) + '<|eot_id|>' }}
			{%- endfor %}
			{%- if add_generation_prompt %}
			{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}
			{%- endif %}
			""";

	private static final String PHI3_LIKE = "{% for message in messages %}"
			+ "{% if message['role'] == 'system' %}{{ '<|system|>\\n' + message['content'] + '<|end|>\\n' }}"
			+ "{% elif message['role'] == 'user' %}{{ '<|user|>\\n' + message['content'] + '<|end|>\\n' }}"
			+ "{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\\n' + message['content'] + '<|end|>\\n' }}"
			+ "{% endif %}{% endfor %}" + "{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}";

	private static final String CHATML_LIKE = "{% for message in messages %}"
			+ "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
			+ "{% endfor %}" + "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

	/** Missing closing "}}" — must fail to parse and trigger the named fallback. */
	private static final String MALFORMED = "{% for message in messages %}{{ message['role'] + ':' + message['content'] %}{% endfor %}";

	private static final List<ChatMessage> CONVERSATION = List.of(ChatMessage.system("You are concise."),
			ChatMessage.user("Hi"), ChatMessage.assistant("Hello!"), ChatMessage.user("Bye"));

	// ── Minimal synthetic GGUF writer (metadata-only, zero tensors) ────────────

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

	// ── Fixture-based precedence tests ──────────────────────────────────────

	@Test
	void resolves_llama3_style_embedded_template(@TempDir Path tmp) throws IOException {
		Path f = writeMinimalGguf(tmp, "llama3.gguf",
				Map.of(GgufChatTemplateResolver.METADATA_KEY, LLAMA3_LIKE));
		try (GgufReader r = GgufReader.open(f)) {
			ChatTemplate t = GgufChatTemplateResolver.resolve(r, "llama3");
			String out = t.format(CONVERSATION);
			assertThat(out).isEqualTo("<|start_header_id|>system<|end_header_id|>\n\nYou are concise.<|eot_id|>"
					+ "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
					+ "<|start_header_id|>assistant<|end_header_id|>\n\nHello!<|eot_id|>"
					+ "<|start_header_id|>user<|end_header_id|>\n\nBye<|eot_id|>"
					+ "<|start_header_id|>assistant<|end_header_id|>\n\n");
		}
	}

	@Test
	void resolves_phi3_style_embedded_template(@TempDir Path tmp) throws IOException {
		Path f = writeMinimalGguf(tmp, "phi3.gguf", Map.of(GgufChatTemplateResolver.METADATA_KEY, PHI3_LIKE));
		try (GgufReader r = GgufReader.open(f)) {
			ChatTemplate t = GgufChatTemplateResolver.resolve(r, "phi3");
			String out = t.format(CONVERSATION);
			assertThat(out).isEqualTo("<|system|>\nYou are concise.<|end|>\n" + "<|user|>\nHi<|end|>\n"
					+ "<|assistant|>\nHello!<|end|>\n" + "<|user|>\nBye<|end|>\n" + "<|assistant|>\n");
		}
	}

	@Test
	void resolves_chatml_style_embedded_template(@TempDir Path tmp) throws IOException {
		Path f = writeMinimalGguf(tmp, "chatml.gguf", Map.of(GgufChatTemplateResolver.METADATA_KEY, CHATML_LIKE));
		try (GgufReader r = GgufReader.open(f)) {
			ChatTemplate t = GgufChatTemplateResolver.resolve(r, "chatml");
			String out = t.format(CONVERSATION);
			assertThat(out).isEqualTo("<|im_start|>system\nYou are concise.<|im_end|>\n"
					+ "<|im_start|>user\nHi<|im_end|>\n" + "<|im_start|>assistant\nHello!<|im_end|>\n"
					+ "<|im_start|>user\nBye<|im_end|>\n" + "<|im_start|>assistant\n");
		}
	}

	@Test
	void falls_back_to_named_template_when_metadata_absent(@TempDir Path tmp) throws IOException {
		Path f = writeMinimalGguf(tmp, "no-template.gguf", Map.of());
		try (GgufReader r = GgufReader.open(f)) {
			assertThat(GgufChatTemplateResolver.hasEmbeddedTemplate(r)).isFalse();
			ChatTemplate t = GgufChatTemplateResolver.resolve(r, "tinyllama");
			assertThat(t.modelType()).isEqualTo("tinyllama");
			assertThat(t.format(List.of(ChatMessage.user("hi")))).contains("<|user|>");
		}
	}

	@Test
	void falls_back_to_named_template_on_malformed_embedded_template(@TempDir Path tmp) throws IOException {
		Path f = writeMinimalGguf(tmp, "malformed.gguf", Map.of(GgufChatTemplateResolver.METADATA_KEY, MALFORMED));
		try (GgufReader r = GgufReader.open(f)) {
			assertThat(GgufChatTemplateResolver.hasEmbeddedTemplate(r)).isTrue();
			ChatTemplate t = GgufChatTemplateResolver.resolve(r, "tinyllama");
			// Must fall back cleanly — no crash, no silent corruption — to the named
			// tinyllama template, not the broken embedded one.
			assertThat(t.modelType()).isEqualTo("tinyllama");
			String out = t.format(List.of(ChatMessage.user("hi")));
			assertThat(out).contains("<|user|>").contains("<|assistant|>");
		}
	}

	@Test
	void falls_back_when_embedded_template_renders_blank(@TempDir Path tmp) throws IOException {
		// Parses fine but the probe conversation renders nothing — must still not
		// be selected as the effective template.
		Path f = writeMinimalGguf(tmp, "blank.gguf", Map.of(GgufChatTemplateResolver.METADATA_KEY, "{{ '' }}"));
		try (GgufReader r = GgufReader.open(f)) {
			ChatTemplate t = GgufChatTemplateResolver.resolve(r, "mistral");
			assertThat(t.modelType()).isEqualTo("mistral");
		}
	}

	@Test
	void resolveFromPath_matches_resolve_and_tolerates_missing_file(@TempDir Path tmp) throws IOException {
		Path f = writeMinimalGguf(tmp, "chatml2.gguf", Map.of(GgufChatTemplateResolver.METADATA_KEY, CHATML_LIKE));
		ChatTemplate t = GgufChatTemplateResolver.resolveFromPath(f.toString(), "chatml");
		assertThat(t.format(List.of(ChatMessage.user("hi")))).contains("<|im_start|>user");

		// A missing/unreadable file must fall back, never throw.
		ChatTemplate fallback = GgufChatTemplateResolver.resolveFromPath(tmp.resolve("nope.gguf").toString(),
				"gemma");
		assertThat(fallback.modelType()).isEqualTo("gemma");
	}

	// ── Real-model spot checks (exit checklist item 1) ──────────────────────

	private static boolean phiModelPresent() {
		return modelPath("Phi-3.5-mini-instruct-Q4_K_M.gguf") != null;
	}

	private static boolean tinyLlamaModelPresent() {
		return modelPath("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf") != null;
	}

	private static Path modelPath(String filename) {
		Path p1 = Path.of("models", filename);
		if (p1.toFile().exists())
			return p1;
		Path p2 = Path.of("..", "models", filename);
		return p2.toFile().exists() ? p2 : null;
	}

	@Test
	@EnabledIf("phiModelPresent")
	void real_phi3_5_gguf_embedded_template_formats_multi_turn_chat() throws Exception {
		try (GgufReader r = GgufReader.open(modelPath("Phi-3.5-mini-instruct-Q4_K_M.gguf"))) {
			assertThat(GgufChatTemplateResolver.hasEmbeddedTemplate(r))
					.as("Phi-3.5-mini-instruct-Q4_K_M.gguf is expected to carry tokenizer.chat_template").isTrue();
			ChatTemplate t = GgufChatTemplateResolver.resolve(r, "phi3");
			String out = t.format(CONVERSATION);
			assertThat(out).isNotBlank();
			assertThat(out).contains("You are concise.").contains("Hi").contains("Hello!").contains("Bye");
			// Ordering must be preserved — a garbled template would interleave or drop turns.
			assertThat(out.indexOf("You are concise.")).isLessThan(out.indexOf("Hi"));
			assertThat(out.indexOf("Hi")).isLessThan(out.indexOf("Hello!"));
			assertThat(out.indexOf("Hello!")).isLessThan(out.indexOf("Bye"));
		}
	}

	@Test
	@EnabledIf("tinyLlamaModelPresent")
	void real_tinyllama_gguf_embedded_template_formats_multi_turn_chat() throws Exception {
		try (GgufReader r = GgufReader.open(modelPath("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"))) {
			assertThat(GgufChatTemplateResolver.hasEmbeddedTemplate(r)).isTrue();
			ChatTemplate t = GgufChatTemplateResolver.resolve(r, "tinyllama");
			String out = t.format(CONVERSATION);
			assertThat(out).isNotBlank();
			assertThat(out).contains("You are concise.").contains("Hi").contains("Hello!").contains("Bye");
		}
	}
}
