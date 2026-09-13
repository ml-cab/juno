package cab.ml.juno.sampler;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

class GrammarSessionTest {

	@Test
	void greedy_prefers_illegal_token_but_grammar_forces_literal() {
		GbnfGrammar g = GbnfGrammar.parse("root ::= \"ab\"");
		List<String> vocab = List.of("<eos>", "a", "b", "x", "ab");
		GrammarSession session = GrammarSession.open(g, id -> utf8(vocab.get(id)), 0, vocab.size());
		Sampler sampler = Sampler.create();
		SamplingParams params = SamplingParams.deterministic().withMaxTokens(8);

		float[] step1 = { 0f, 1f, 1f, 50f, 1f };
		int t1 = sampler.sample(step1, params, new int[0], null, session);
		assertThat(vocab.get(t1)).isIn("a", "ab");

		if (vocab.get(t1).equals("ab")) {
			assertThat(session.complete()).isTrue();
			return;
		}
		float[] step2 = { 0f, 50f, 1f, 50f, 1f };
		int t2 = sampler.sample(step2, params, new int[] { t1 }, null, session);
		assertThat(vocab.get(t2)).isEqualTo("b");
		assertThat(session.complete()).isTrue();
	}

	@Test
	void unconstrained_when_session_is_null() {
		Sampler sampler = Sampler.create();
		float[] logits = { 0.1f, 0.5f, 10f };
		int tok = sampler.sample(logits, SamplingParams.deterministic(), new int[0], null, null);
		assertThat(tok).isEqualTo(2);
	}

	@Test
	void json_object_schema_emits_parseable_object() {
		GbnfGrammar g = JsonSchemaToGbnf.compileGrammar("""
				{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}
				""");
		List<String> vocab = jsonPieces();
		vocab.add("<eos>");
		int eos = vocab.size() - 1;
		GrammarSession session = GrammarSession.open(g, id -> utf8(vocab.get(id)), eos, vocab.size());
		Sampler sampler = Sampler.create();
		SamplingParams params = SamplingParams.deterministic().withMaxTokens(32);
		StringBuilder out = new StringBuilder();
		int[] hist = new int[0];
		for (int step = 0; step < 32; step++) {
			float[] logits = adversarialLogits(vocab.size(), eos);
			int tok = sampler.sample(logits, params, hist, null, session);
			if (tok == eos)
				break;
			out.append(vocab.get(tok));
			hist = append(hist, tok);
			if (session.complete())
				break;
		}
		assertThat(out).contains("\"ok\"");
		assertThat(GbnfGrammar.parse(JsonSchemaToGbnf.compile("""
				{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}
				""")).isComplete(out.toString().getBytes(StandardCharsets.UTF_8))).isTrue();
	}

	private static byte[] utf8(String s) {
		if ("<eos>".equals(s))
			return GrammarSession.EMPTY;
		return s.getBytes(StandardCharsets.UTF_8);
	}

	private static List<String> jsonPieces() {
		return new ArrayList<>(List.of("{", "}", ":", ",", "\"ok\"", "true", "false", "null", "0", "1"));
	}

	private static float[] adversarialLogits(int n, int eos) {
		float[] l = new float[n];
		for (int i = 0; i < n; i++)
			l[i] = (i == eos) ? -1f : 100f - i;
		return l;
	}

	private static int[] append(int[] hist, int tok) {
		int[] n = java.util.Arrays.copyOf(hist, hist.length + 1);
		n[hist.length] = tok;
		return n;
	}
}
