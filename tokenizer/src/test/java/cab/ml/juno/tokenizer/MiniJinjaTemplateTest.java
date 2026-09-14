package cab.ml.juno.tokenizer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

/**
 * White-box tests for the restricted Jinja engine used to render
 * GGUF-embedded {@code tokenizer.chat_template} strings. Package-private
 * class — tests live in the same package by design (see {@code
 * GgufChatTemplateResolverTest} for the public entry-point fixture tests).
 */
class MiniJinjaTemplateTest {

	private static Map<String, Object> messagesContext(List<Map<String, Object>> messages, String bos, String eos,
			boolean addGenPrompt) {
		Map<String, Object> ctx = new HashMap<>();
		ctx.put("messages", messages);
		ctx.put("bos_token", bos);
		ctx.put("eos_token", eos);
		ctx.put("add_generation_prompt", addGenPrompt);
		return ctx;
	}

	private static Map<String, Object> msg(String role, String content) {
		Map<String, Object> m = new HashMap<>();
		m.put("role", role);
		m.put("content", content);
		return m;
	}

	@Test
	void renders_plain_text_unchanged() {
		MiniJinjaTemplate t = new MiniJinjaTemplate("hello world");
		assertThat(t.render(Map.of())).isEqualTo("hello world");
	}

	@Test
	void renders_variable_output_and_string_concat() {
		MiniJinjaTemplate t = new MiniJinjaTemplate("{{ 'a' + name + 'c' }}");
		assertThat(t.render(Map.of("name", "b"))).isEqualTo("abc");
	}

	@Test
	void unknown_variable_stringifies_to_empty() {
		MiniJinjaTemplate t = new MiniJinjaTemplate("[{{ missing }}]");
		assertThat(t.render(Map.of())).isEqualTo("[]");
	}

	@Test
	void member_access_by_dot_and_bracket() {
		MiniJinjaTemplate t = new MiniJinjaTemplate("{{ m.role }}:{{ m['content'] }}");
		assertThat(t.render(Map.of("m", msg("user", "hi")))).isEqualTo("user:hi");
	}

	@Test
	void trim_filter_strips_whitespace() {
		MiniJinjaTemplate t = new MiniJinjaTemplate("[{{ m['content'] | trim }}]");
		assertThat(t.render(Map.of("m", msg("user", "  hi  ")))).isEqualTo("[hi]");
	}

	@Test
	void for_loop_iterates_and_binds_loop_var() {
		MiniJinjaTemplate t = new MiniJinjaTemplate("{% for m in messages %}({{ m.role }}){% endfor %}");
		List<Map<String, Object>> msgs = List.of(msg("system", "s"), msg("user", "u"));
		assertThat(t.render(messagesContext(msgs, "", "", true))).isEqualTo("(system)(user)");
	}

	@Test
	void loop_first_and_last_are_exposed() {
		MiniJinjaTemplate t = new MiniJinjaTemplate(
				"{% for m in messages %}{% if loop.first %}[{% endif %}{{ m.role }}{% if loop.last %}]{% endif %}{% endfor %}");
		List<Map<String, Object>> msgs = List.of(msg("a", ""), msg("b", ""), msg("c", ""));
		assertThat(t.render(messagesContext(msgs, "", "", true))).isEqualTo("[abc]");
	}

	@Test
	void if_elif_else_chain_selects_matching_branch() {
		String src = "{% if r == 'system' %}S{% elif r == 'user' %}U{% else %}O{% endif %}";
		MiniJinjaTemplate t = new MiniJinjaTemplate(src);
		assertThat(t.render(Map.of("r", "system"))).isEqualTo("S");
		assertThat(t.render(Map.of("r", "user"))).isEqualTo("U");
		assertThat(t.render(Map.of("r", "assistant"))).isEqualTo("O");
	}

	@Test
	void and_or_not_short_circuit_correctly() {
		MiniJinjaTemplate t = new MiniJinjaTemplate("{% if a and not b %}Y{% else %}N{% endif %}");
		assertThat(t.render(Map.of("a", true, "b", false))).isEqualTo("Y");
		assertThat(t.render(Map.of("a", true, "b", true))).isEqualTo("N");
		assertThat(t.render(Map.of("a", false, "b", false))).isEqualTo("N");
	}

	@Test
	void whitespace_control_dashes_strip_surrounding_newlines_and_indentation() {
		// A realistic multi-line, indented template — {%- / -%} must strip the
		// newline+indentation around tags so the rendered output has none of the
		// source file's own formatting whitespace, matching real chat_template
		// authoring style (Llama-3 / Phi-3 style GGUF metadata).
		String src = """
				{{- bos_token }}
				{%- for message in messages %}
				{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + (message['content'] | trim) + '<|eot_id|>' }}
				{%- endfor %}
				{%- if add_generation_prompt %}
				{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}
				{%- endif %}
				""";
		MiniJinjaTemplate t = new MiniJinjaTemplate(src);
		List<Map<String, Object>> msgs = List.of(msg("system", "Sys msg"), msg("user", "Hi"), msg("assistant", "Yo"));
		String out = t.render(messagesContext(msgs, "", "", true));

		assertThat(out).isEqualTo("<|start_header_id|>system<|end_header_id|>\n\nSys msg<|eot_id|>"
				+ "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
				+ "<|start_header_id|>assistant<|end_header_id|>\n\nYo<|eot_id|>"
				+ "<|start_header_id|>assistant<|end_header_id|>\n\n");
	}

	@Test
	void default_trim_blocks_and_lstrip_blocks_apply_without_explicit_dashes() {
		// Real TinyLlama/Zephyr-style chat_template (verbatim from the GGUF metadata) —
		// unlike the Llama-3 fixture above, this one has no {%- / -%} markers at all and
		// relies entirely on Jinja2's environment defaults (trim_blocks=True,
		// lstrip_blocks=True), which is how HF's apply_chat_template and llama.cpp's own
		// renderer treat it. Without honoring those defaults, every block-tag line leaves
		// a blank line behind, and the noise compounds every turn — exactly the bug this
		// test guards against (observed as garbled generation after a few chat turns).
		String src = """
				{% for message in messages %}
				{% if message['role'] == 'user' %}
				{{ '<|user|>
				' + message['content'] + eos_token }}
				{% elif message['role'] == 'system' %}
				{{ '<|system|>
				' + message['content'] + eos_token }}
				{% elif message['role'] == 'assistant' %}
				{{ '<|assistant|>
				'  + message['content'] + eos_token }}
				{% endif %}
				{% if loop.last and add_generation_prompt %}
				{{ '<|assistant|>' }}
				{% endif %}
				{% endfor %}
				""";
		MiniJinjaTemplate t = new MiniJinjaTemplate(src);
		List<Map<String, Object>> msgs = List.of(msg("user", "Hi"), msg("assistant", "Hello!"), msg("user", "Bye"));
		String out = t.render(messagesContext(msgs, "", "</s>", true));

		// Note the trailing "\n": per Jinja2's own docs, trim_blocks only eats the
		// newline that follows a *block* tag ({% %}), never one following a
		// variable/output tag ({{ }}) — so the newline between the final
		// {{ '<|assistant|>' }} output and the {% endif %} that follows it is
		// genuine template output, not whitespace noise. This is a deliberately
		// exact (not just substring) assertion so a regression in that subtlety
		// fails loudly.
		assertThat(out).isEqualTo(
				"<|user|>\nHi</s>\n<|assistant|>\nHello!</s>\n<|user|>\nBye</s>\n<|assistant|>\n");
	}

	@Test
	void bos_token_is_substituted() {
		MiniJinjaTemplate t = new MiniJinjaTemplate("{{ bos_token }}{{ 'x' }}");
		assertThat(t.render(Map.of("bos_token", "<BOS>"))).isEqualTo("<BOS>x");
	}

	// ── Failure paths — must throw MiniJinjaException, never a raw exception ────

	@Test
	void missing_endfor_throws() {
		assertThatThrownBy(() -> new MiniJinjaTemplate("{% for m in messages %}{{ m.role }}"))
				.isInstanceOf(MiniJinjaException.class);
	}

	@Test
	void unterminated_output_tag_throws() {
		assertThatThrownBy(() -> new MiniJinjaTemplate("{{ messages")).isInstanceOf(MiniJinjaException.class);
	}

	@Test
	void unsupported_tag_throws() {
		assertThatThrownBy(() -> new MiniJinjaTemplate("{% set x = 1 %}")).isInstanceOf(MiniJinjaException.class);
	}

	@Test
	void unsupported_filter_throws() {
		assertThatThrownBy(() -> new MiniJinjaTemplate("{{ x | upper_snake }}")).isInstanceOf(MiniJinjaException.class);
	}

	@Test
	void filter_with_arguments_throws() {
		assertThatThrownBy(() -> new MiniJinjaTemplate("{{ x | default('n/a') }}"))
				.isInstanceOf(MiniJinjaException.class);
	}

	@Test
	void non_list_for_target_throws_at_render_time() {
		MiniJinjaTemplate t = new MiniJinjaTemplate("{% for m in messages %}{{ m }}{% endfor %}");
		assertThatThrownBy(() -> t.render(Map.of("messages", "not-a-list"))).isInstanceOf(MiniJinjaException.class);
	}

	@Test
	void non_numeric_ordering_comparison_throws() {
		MiniJinjaTemplate t = new MiniJinjaTemplate("{% if a < b %}Y{% endif %}");
		assertThatThrownBy(() -> t.render(Map.of("a", "x", "b", "y"))).isInstanceOf(MiniJinjaException.class);
	}
}
