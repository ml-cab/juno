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


import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

/**
 * The constrained-decoding smoke's checks, through the launcher in child JVMs on the CPU:
 * <ul>
 * <li>the CLI fails closed on an unsupported JSON Schema keyword and on a grammar and a schema together;</li>
 * <li>over OpenAI REST, {@code json_schema} and {@code json_object} answers parse, {@code x_juno_grammar}
 * answers yes or no, a grammar with {@code response_format} and an unsupported schema keyword are HTTP 400
 * naming the cause;</li>
 * <li>{@code --grammar-file} and {@code --json-schema-file} constrain requests that carry no constraint;</li>
 * <li>with a LoRA overlay and two requests in parallel, both schema answers parse;</li>
 * <li>each server's JFR recording holds a constrained-request event per constrained request.</li>
 * </ul>
 * The grammars are the sample files the documentation ships, written into the probe's directory.
 */
final class GrammarProbe implements KitProbe {

	static final String YES_NO = "root ::= \"yes\" | \"no\"\n";
	static final String OK_SCHEMA = "{\"type\":\"object\",\"properties\":{\"ok\":{\"type\":\"boolean\"}},\"required\":[\"ok\"]}";
	static final String BAD_SCHEMA = "{\"type\":\"string\",\"pattern\":\"^[a-z]+$\"}";
	static final String PROMPT_JSON = "Reply with JSON only: an object with boolean field ok.";
	static final String PROMPT_YES_NO = "Answer with a single word, yes or no: is 2 even?";
	private static final Duration START = Duration.ofMinutes(10);

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		Path model = Path.of(args.get("model"));
		String heap = args.getOrDefault("serverHeap", "4g");
		Path work = Path.of(args.getOrDefault("logDir", ".")).resolve(id);
		Files.createDirectories(work);
		Path yesNo = Files.writeString(work.resolve("yes-no.gbnf"), YES_NO);
		Path okSchema = Files.writeString(work.resolve("ok.schema.json"), OK_SCHEMA);
		Path badSchema = Files.writeString(work.resolve("pattern-unsupported.schema.json"), BAD_SCHEMA);
		List<ProbeReport.Check> checks = new ArrayList<>();
		Map<String, Object> facts = new LinkedHashMap<>();

		JunoProcess.Result r = JunoProcess.runOnce(work.resolve("cli-schema-unsupported"), heap,
				List.of("--model-path", model.toString(), "--cpu", "--json-schema-file", badSchema.toString()), "",
				START);
		String out = r.output().toLowerCase(Locale.ROOT);
		checks.add(new ProbeReport.Check("CLI: an unsupported JSON Schema keyword fails closed",
				r.exit() != 0 && (out.contains("pattern") || out.contains("failed to load constrained")),
				"exit " + r.exit()));
		r = JunoProcess.runOnce(work.resolve("cli-mutually-exclusive"), heap, List.of("--model-path", model.toString(),
				"--cpu", "--grammar-file", yesNo.toString(), "--json-schema-file", okSchema.toString()), "", START);
		checks.add(new ProbeReport.Check("CLI: --grammar-file and --json-schema-file together fail closed",
				r.exit() != 0 && r.output().toLowerCase(Locale.ROOT).contains("mutually exclusive"), "exit " + r.exit()));

		try (JunoProcess s = JunoProcess.serve(work.resolve("api-base"), heap, serverArgs(model), START)) {
			String m = s.modelId();
			JunoProcess.Answer a = s.post(PATH, body(m, PROMPT_JSON, 32).set("response_format", schemaFormat())
					.toString());
			checks.add(new ProbeReport.Check("OpenAI: json_schema answer is parseable JSON with a boolean ok",
					a.status() == 200 && okObject(content(a)), describe(a)));
			a = s.post(PATH, body(m, PROMPT_JSON, 48).set("response_format", type("json_object")).toString());
			checks.add(new ProbeReport.Check("OpenAI: json_object answer is a parseable JSON object",
					a.status() == 200 && object(content(a)) != null, describe(a)));
			a = s.post(PATH, body(m, PROMPT_YES_NO, 8).put("x_juno_grammar", "root ::= \"yes\" | \"no\"").toString());
			checks.add(new ProbeReport.Check("OpenAI: x_juno_grammar answers yes or no",
					a.status() == 200 && yesOrNo(content(a)), describe(a)));
			ObjectNode both = body(m, PROMPT_JSON, 8);
			both.set("response_format", type("json_object"));
			a = s.post(PATH, both.put("x_juno_grammar", "root ::= \"a\"").toString());
			checks.add(new ProbeReport.Check("OpenAI: x_juno_grammar with response_format is HTTP 400 naming x_juno_grammar",
					a.status() == 400 && a.body().contains("x_juno_grammar"), describe(a)));
			ObjectNode bad = Json.MAPPER.createObjectNode().put("type", "json_schema");
			bad.putObject("json_schema").put("name", "bad").set("schema", Json.MAPPER.readTree(BAD_SCHEMA));
			a = s.post(PATH, body(m, "x", 8).set("response_format", bad).toString());
			checks.add(new ProbeReport.Check("OpenAI: an unsupported JSON Schema keyword is HTTP 400 naming it",
					a.status() == 400 && a.body().toLowerCase(Locale.ROOT).contains("pattern"), describe(a)));
			s.close();
			jfr(s, 3, "response_format server", checks);
		}

		List<String> withGrammar = serverArgs(model);
		withGrammar.addAll(List.of("--grammar-file", yesNo.toString()));
		try (JunoProcess s = JunoProcess.serve(work.resolve("cli-gbnf"), heap, withGrammar, START)) {
			JunoProcess.Answer a = s.post(PATH, body(s.modelId(), PROMPT_YES_NO, 8).toString());
			checks.add(new ProbeReport.Check("--grammar-file: an unconstrained request answers yes or no",
					a.status() == 200 && yesOrNo(content(a)), describe(a)));
			s.close();
			jfr(s, 1, "--grammar-file server", checks);
		}

		List<String> withSchema = serverArgs(model);
		withSchema.addAll(List.of("--json-schema-file", okSchema.toString()));
		try (JunoProcess s = JunoProcess.serve(work.resolve("cli-schema"), heap, withSchema, START)) {
			JunoProcess.Answer a = s.post(PATH, body(s.modelId(), PROMPT_JSON, 32).toString());
			checks.add(new ProbeReport.Check("--json-schema-file: an unconstrained request is parseable JSON with a boolean ok",
					a.status() == 200 && okObject(content(a)), describe(a)));
			s.close();
			jfr(s, 1, "--json-schema-file server", checks);
		}

		Path adapter = KitAdapters.trainSmall(model, work.resolve("kit-overlay.lora"));
		List<String> withLora = serverArgs(model);
		withLora.addAll(List.of("--parallel", "2", "--lora-play", adapter.toString()));
		try (JunoProcess s = JunoProcess.serve(work.resolve("lora-parallel"), heap, withLora, START)) {
			String body = body(s.modelId(), PROMPT_JSON, 32).set("response_format", schemaFormat()).toString();
			List<CompletableFuture<JunoProcess.Answer>> two = new ArrayList<>();
			for (int i = 0; i < 2; i++)
				two.add(CompletableFuture.supplyAsync(() -> {
					try {
						return s.post(PATH, body);
					} catch (Exception e) {
						return new JunoProcess.Answer(-1, String.valueOf(e.getMessage()));
					}
				}, Thread::startVirtualThread));
			JunoProcess.Answer a1 = two.get(0).join();
			JunoProcess.Answer a2 = two.get(1).join();
			checks.add(new ProbeReport.Check("LoRA overlay with 2 parallel: both json_schema answers parseable",
					a1.status() == 200 && a2.status() == 200 && okObject(content(a1)) && okObject(content(a2)),
					describe(a1) + " / " + describe(a2)));
			facts.put("loraLogMentionsOverlay", s.log().toLowerCase(Locale.ROOT).contains("lora"));
			s.close();
			jfr(s, 2, "LoRA overlay with 2 parallel", checks);
		}
		return new ProbeReport(id, ProbeKind.CHECK, checks.stream().allMatch(ProbeReport.Check::passed), checks,
				List.of(), facts);
	}

	static final String PATH = "/v1/chat/completions";

	/** The smoke's server: greedy, 32 tokens, CPU, with a JFR recording and verbose engine logging. */
	private static List<String> serverArgs(Path model) {
		List<String> a = JunoProcess.localArgs(model, 32);
		a.addAll(List.of("--cpu", "--jfr", "30m", "--verbose"));
		return a;
	}

	private static void jfr(JunoProcess s, int atLeast, String server, List<ProbeReport.Check> checks)
			throws Exception {
		long n = JfrEvents.count(s.recordings(), JfrEvents.GRAMMAR_CONSTRAINED);
		checks.add(new ProbeReport.Check("JFR: at least " + atLeast + " constrained request" + (atLeast == 1 ? "" : "s")
				+ ", " + server, n >= atLeast, n + " events in " + s.recordings().size() + " recordings"));
	}

	static ObjectNode body(String model, String prompt, int maxTokens) {
		ObjectNode b = Json.MAPPER.createObjectNode().put("model", model);
		b.putArray("messages").addObject().put("role", "user").put("content", prompt);
		return b.put("max_tokens", maxTokens).put("temperature", 0);
	}

	private static ObjectNode type(String t) {
		return Json.MAPPER.createObjectNode().put("type", t);
	}

	private static ObjectNode schemaFormat() throws Exception {
		ObjectNode f = type("json_schema");
		f.putObject("json_schema").put("name", "ok").set("schema", Json.MAPPER.readTree(OK_SCHEMA));
		return f;
	}

	static String content(JunoProcess.Answer a) {
		try {
			return a.status() == 200 ? a.json().path("choices").path(0).path("message").path("content").asText("")
					: "";
		} catch (Exception e) {
			return "";
		}
	}

	static String describe(JunoProcess.Answer a) {
		String c = content(a);
		String shown = a.status() == 200 ? c : a.body();
		return "HTTP " + a.status() + ": " + (shown.length() > 200 ? shown.substring(0, 200) + "..." : shown);
	}

	static JsonNode object(String text) {
		try {
			JsonNode n = Json.MAPPER.readTree(text.strip());
			return n != null && n.isObject() ? n : null;
		} catch (Exception e) {
			return null;
		}
	}

	/** The smoke's rule: a JSON object, and if it has {@code ok}, a boolean one. */
	static boolean okObject(String text) {
		JsonNode n = object(text);
		return n != null && (!n.has("ok") || n.get("ok").isBoolean());
	}

	static boolean yesOrNo(String text) {
		String t = text.strip().toLowerCase(Locale.ROOT);
		return t.equals("yes") || t.equals("no");
	}
}
