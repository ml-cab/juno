package cab.ml.juno.sampler;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

/**
 * Fixture eval: greedy sampling under json_schema with an adversarial logit
 * prior. Gate: ≥95% of the fixed set must be parseable JSON and complete under
 * the compiled grammar.
 */
class GrammarEvalTest {

	private static final String[] SCHEMAS = {
			"{\"type\":\"object\",\"properties\":{\"ok\":{\"type\":\"boolean\"}},\"required\":[\"ok\"]}",
			"{\"type\":\"object\",\"properties\":{\"n\":{\"type\":\"integer\"}},\"required\":[\"n\"]}",
			"{\"type\":\"object\",\"properties\":{\"s\":{\"type\":\"string\"}},\"required\":[\"s\"]}",
			"{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"number\"}},\"required\":[\"x\"]}",
			"{\"type\":\"object\",\"properties\":{\"flag\":{\"type\":\"boolean\"},\"n\":{\"type\":\"integer\"}},\"required\":[\"flag\"]}",
			"{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"age\":{\"type\":\"integer\"}},\"required\":[\"name\"]}",
			"{\"type\":\"object\",\"properties\":{\"tone\":{\"enum\":[\"low\",\"high\"]}},\"required\":[\"tone\"]}",
			"{\"type\":\"object\",\"properties\":{\"vals\":{\"type\":\"array\",\"items\":{\"type\":\"integer\"}}},\"required\":[\"vals\"]}",
			"{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"boolean\"},\"b\":{\"type\":\"boolean\"}},\"required\":[\"a\",\"b\"]}",
			"{\"type\":\"array\",\"items\":{\"type\":\"integer\"}}",
			"{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}",
			"{\"type\":\"object\",\"properties\":{\"p\":{\"type\":\"null\"}},\"required\":[\"p\"]}",
			"{\"type\":\"object\",\"properties\":{\"k\":{\"const\":\"v\"}},\"required\":[\"k\"]}",
			"{\"type\":\"object\",\"properties\":{\"xs\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}},\"required\":[\"xs\"]}",
			"{\"type\":\"object\",\"properties\":{\"nested\":{\"type\":\"object\",\"properties\":{\"z\":{\"type\":\"boolean\"}},\"required\":[\"z\"]}},\"required\":[\"nested\"]}",
			"{\"type\":\"object\",\"properties\":{\"id\":{\"type\":\"integer\"},\"label\":{\"type\":\"string\"}},\"required\":[\"id\"]}",
			"{\"type\":\"object\",\"properties\":{\"on\":{\"type\":\"boolean\"}}}",
			"{\"type\":\"object\",\"properties\":{\"mode\":{\"enum\":[\"a\",\"b\",\"c\"]}},\"required\":[\"mode\"]}",
			"{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"number\"},\"r\":{\"type\":\"integer\"}},\"required\":[\"q\"]}",
			"{\"type\":\"object\",\"properties\":{\"title\":{\"type\":\"string\"},\"done\":{\"type\":\"boolean\"}},\"required\":[\"title\",\"done\"]}"
	};

	@Test
	void at_least_95_percent_valid_json() {
		List<String> vocab = vocab();
		int eos = vocab.size() - 1;
		int ok = 0;
		List<String> failures = new ArrayList<>();
		Sampler sampler = Sampler.create();
		SamplingParams params = SamplingParams.deterministic().withMaxTokens(48);
		for (int i = 0; i < SCHEMAS.length; i++) {
			GbnfGrammar g = JsonSchemaToGbnf.compileGrammar(SCHEMAS[i]);
			GrammarSession session = GrammarSession.open(g, id -> piece(vocab.get(id)), eos, vocab.size());
			StringBuilder out = new StringBuilder();
			int[] hist = new int[0];
			for (int step = 0; step < 48; step++) {
				float[] logits = new float[vocab.size()];
				for (int t = 0; t < vocab.size(); t++)
					logits[t] = (t == eos) ? -2f : 200f - t;
				int tok = sampler.sample(logits, params, hist, null, session);
				if (tok == eos)
					break;
				out.append(vocab.get(tok));
				int[] n = java.util.Arrays.copyOf(hist, hist.length + 1);
				n[hist.length] = tok;
				hist = n;
				if (session.complete())
					break;
			}
			String text = out.toString();
			boolean complete = g.isComplete(text.getBytes(StandardCharsets.UTF_8));
			boolean parsed = false;
			try {
				MiniJson.parse(text);
				parsed = true;
			} catch (RuntimeException ignored) {
			}
			if (complete && parsed)
				ok++;
			else
				failures.add("#" + i + " complete=" + complete + " parsed=" + parsed + " text=" + text);
		}
		double rate = ok / (double) SCHEMAS.length;
		assertThat(rate)
				.as("valid JSON rate %s / %s failures=%s", ok, SCHEMAS.length, failures)
				.isGreaterThanOrEqualTo(0.95);
	}

	private static List<String> vocab() {
		List<String> v = new ArrayList<>();
		v.addAll(List.of("{", "}", "[", "]", ":", ",", "true", "false", "null", "0", "1", "2", "3.5", "-",
				"\"ok\"", "\"n\"", "\"s\"", "\"x\"", "\"flag\"", "\"name\"", "\"age\"", "\"tone\"", "\"low\"",
				"\"high\"", "\"vals\"", "\"a\"", "\"b\"", "\"c\"", "\"city\"", "\"p\"", "\"k\"", "\"v\"", "\"xs\"",
				"\"nested\"", "\"z\"", "\"id\"", "\"label\"", "\"on\"", "\"mode\"", "\"q\"", "\"r\"", "\"title\"",
				"\"done\"", "\"Ada\"", "\"hi\""));
		v.add("<eos>");
		return v;
	}

	private static byte[] piece(String s) {
		if ("<eos>".equals(s))
			return GrammarSession.EMPTY;
		return s.getBytes(StandardCharsets.UTF_8);
	}
}
