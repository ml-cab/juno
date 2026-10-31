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


import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

/**
 * The function-calling smoke's checks, through the launcher in child JVMs on the CPU. On a ChatML model
 * (Qwen2.5): {@code tool_choice} required, named and none; a {@code role=tool} turn continuing with text; tools
 * with {@code response_format} or {@code x_juno_grammar} refused with HTTP 400; the streamed path carrying the
 * call; two required requests in parallel; and the JFR recording holding the constrained requests. On
 * TinyLlama, whose chat template cannot carry tools, tools are refused with HTTP 400 naming the template, with
 * and without a LoRA overlay.
 */
final class ToolsProbe implements KitProbe {

	static final String CHATML_MODEL = "qwen2.5-3b";
	static final String REFUSING_MODEL = "tinyllama-1.1b";
	static final String TOOLS = "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":"
			+ "\"Current weather for a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":"
			+ "{\"type\":\"string\"}},\"required\":[\"city\"]}}}]";
	static final String ASK = "Weather in Boston?";
	private static final Duration START = Duration.ofMinutes(10);

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		Path chatml = Path.of(args.get("model." + CHATML_MODEL));
		Path refusing = Path.of(args.get("model." + REFUSING_MODEL));
		String chatmlHeap = args.getOrDefault("serverHeap", "6g");
		String refusingHeap = args.getOrDefault("serverHeap", "4g");
		Path work = Path.of(args.getOrDefault("logDir", ".")).resolve(id);
		List<ProbeReport.Check> checks = new ArrayList<>();

		try (JunoProcess s = JunoProcess.serve(work.resolve("chatml"), chatmlHeap, serverArgs(chatml), START)) {
			String m = s.modelId();
			JunoProcess.Answer required = s.post(GrammarProbe.PATH, tools(m, ASK, 48).put("tool_choice", "required")
					.toString());
			JsonNode call = call(required);
			checks.add(new ProbeReport.Check("ChatML: tool_choice required returns a get_weather call with JSON arguments",
					isWeather(call) && finish(required).equals("tool_calls")
							&& GrammarProbe.object(call.path("function").path("arguments").asText("")) != null,
					GrammarProbe.describe(required) + callText(call)));
			ObjectNode named = tools(m, ASK, 48);
			named.putObject("tool_choice").put("type", "function").putObject("function").put("name", "get_weather");
			JunoProcess.Answer a = s.post(GrammarProbe.PATH, named.toString());
			checks.add(new ProbeReport.Check("ChatML: a named tool_choice returns a get_weather call",
					isWeather(call(a)), GrammarProbe.describe(a) + callText(call(a))));
			a = s.post(GrammarProbe.PATH, tools(m, "Say hi in one word.", 16).put("tool_choice", "none").toString());
			checks.add(new ProbeReport.Check("ChatML: tool_choice none returns no tool call",
					a.status() == 200 && !hasCalls(a), GrammarProbe.describe(a)));
			if (required.status() == 200) {
				ObjectNode turn = tools(m, ASK, 24).put("tool_choice", "none");
				ArrayNode msgs = turn.putArray("messages");
				msgs.addObject().put("role", "user").put("content", ASK);
				msgs.add(required.json().path("choices").path(0).path("message"));
				msgs.addObject().put("role", "tool").put("tool_call_id", call.path("id").asText("call_0"))
						.put("content", "72F");
				a = s.post(GrammarProbe.PATH, turn.toString());
				checks.add(new ProbeReport.Check("ChatML: a role=tool turn continues with text",
						a.status() == 200 && !hasCalls(a) && !GrammarProbe.content(a).isEmpty(), GrammarProbe.describe(a)));
			} else {
				checks.add(new ProbeReport.Check("ChatML: a role=tool turn continues with text", false,
						"not run: the required request failed"));
			}
			ObjectNode withFormat = tools(m, "x", 8);
			withFormat.putObject("response_format").put("type", "json_object");
			a = s.post(GrammarProbe.PATH, withFormat.toString());
			checks.add(new ProbeReport.Check("ChatML: tools with response_format json_object is HTTP 400",
					a.status() == 400 && a.body().contains("response_format"), GrammarProbe.describe(a)));
			a = s.post(GrammarProbe.PATH, tools(m, "x", 8).put("x_juno_grammar", "root ::= \"a\"").toString());
			checks.add(new ProbeReport.Check("ChatML: tools with x_juno_grammar is HTTP 400",
					a.status() == 400 && a.body().contains("x_juno_grammar"), GrammarProbe.describe(a)));
			a = s.post(GrammarProbe.PATH, tools(m, ASK, 48).put("tool_choice", "required").put("stream", true)
					.toString());
			checks.add(streamCheck(a));
			s.close();
			long n = JfrEvents.count(s.recordings(), JfrEvents.GRAMMAR_CONSTRAINED);
			checks.add(new ProbeReport.Check("JFR: at least 2 constrained requests, ChatML server", n >= 2,
					n + " events"));
		}

		try (JunoProcess s = JunoProcess.serve(work.resolve("tinyllama"), refusingHeap, serverArgs(refusing), START)) {
			JunoProcess.Answer a = s.post(GrammarProbe.PATH, tools(s.modelId(), "Weather?", 8).toString());
			checks.add(new ProbeReport.Check("TinyLlama template: tools are HTTP 400 naming the template",
					a.status() == 400 && a.body().toLowerCase(Locale.ROOT).contains("tinyllama"),
					GrammarProbe.describe(a)));
		}

		Path adapter = KitAdapters.trainSmall(refusing, work.resolve("kit-overlay.lora"));
		List<String> withLora = serverArgs(refusing);
		withLora.addAll(List.of("--lora-play", adapter.toString()));
		try (JunoProcess s = JunoProcess.serve(work.resolve("tinyllama-lora"), refusingHeap, withLora, START)) {
			JunoProcess.Answer a = s.post(GrammarProbe.PATH, tools(s.modelId(), "Weather?", 8).toString());
			checks.add(new ProbeReport.Check("TinyLlama template with a LoRA overlay: tools are HTTP 400 naming the template",
					a.status() == 400 && a.body().toLowerCase(Locale.ROOT).contains("tinyllama"),
					GrammarProbe.describe(a)));
		}

		List<String> parallel = serverArgs(chatml);
		parallel.addAll(List.of("--parallel", "2"));
		try (JunoProcess s = JunoProcess.serve(work.resolve("chatml-parallel"), chatmlHeap, parallel, START)) {
			String body = tools(s.modelId(), ASK, 48).put("tool_choice", "required").toString();
			List<CompletableFuture<JunoProcess.Answer>> two = new ArrayList<>();
			for (int i = 0; i < 2; i++)
				two.add(CompletableFuture.supplyAsync(() -> {
					try {
						return s.post(GrammarProbe.PATH, body);
					} catch (Exception e) {
						return new JunoProcess.Answer(-1, String.valueOf(e.getMessage()));
					}
				}, Thread::startVirtualThread));
			JunoProcess.Answer a1 = two.get(0).join();
			JunoProcess.Answer a2 = two.get(1).join();
			checks.add(new ProbeReport.Check("ChatML with 2 parallel: both required requests return a get_weather call",
					isWeather(call(a1)) && isWeather(call(a2)),
					GrammarProbe.describe(a1) + callText(call(a1)) + " / " + GrammarProbe.describe(a2) + callText(call(a2))));
		}
		return new ProbeReport(id, ProbeKind.CHECK, checks.stream().allMatch(ProbeReport.Check::passed), checks,
				List.of(), Map.of());
	}

	private static List<String> serverArgs(Path model) {
		List<String> a = JunoProcess.localArgs(model, 32);
		a.addAll(List.of("--cpu", "--jfr", "30m", "--verbose"));
		return a;
	}

	private static ObjectNode tools(String model, String prompt, int maxTokens) throws Exception {
		ObjectNode b = GrammarProbe.body(model, prompt, maxTokens);
		b.set("tools", Json.MAPPER.readTree(TOOLS));
		return b;
	}

	private static JsonNode call(JunoProcess.Answer a) {
		try {
			return a.status() == 200 ? a.json().path("choices").path(0).path("message").path("tool_calls").path(0)
					: Json.MAPPER.missingNode();
		} catch (Exception e) {
			return Json.MAPPER.missingNode();
		}
	}

	private static boolean isWeather(JsonNode call) {
		return call.path("function").path("name").asText("").equals("get_weather");
	}

	private static String callText(JsonNode call) {
		return call.isMissingNode() ? "" : "; call " + call.path("function");
	}

	private static String finish(JunoProcess.Answer a) {
		try {
			return a.json().path("choices").path(0).path("finish_reason").asText("");
		} catch (Exception e) {
			return "";
		}
	}

	private static boolean hasCalls(JunoProcess.Answer a) {
		try {
			JsonNode tc = a.json().path("choices").path(0).path("message").path("tool_calls");
			return !tc.isMissingNode() && !tc.isNull() && !(tc.isArray() && tc.isEmpty());
		} catch (Exception e) {
			return false;
		}
	}

	/**
	 * The smoke's rule for the streamed path: the stream carries the call with {@code finish_reason}
	 * {@code tool_calls}, or it is the buffered form (two to four chunks, the assistant role among them), which
	 * is what a call cut off by the token limit streams.
	 */
	static ProbeReport.Check streamCheck(JunoProcess.Answer a) {
		String sse = a.body();
		long chunks = sse.lines().filter(l -> l.startsWith("data: {")).count();
		boolean call = sse.matches("(?s).*finish_reason\": ?\"tool_calls\".*") && sse.contains("\"tool_calls\"");
		boolean buffered = chunks >= 2 && chunks <= 4 && sse.contains("\"role\":\"assistant\"");
		return new ProbeReport.Check("ChatML: streamed tool_choice required carries the tool call",
				a.status() == 200 && (call || buffered),
				"HTTP " + a.status() + ", " + chunks + " chunks" + (call ? ", call streamed" : buffered ? ", buffered" : ""));
	}
}
