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

import cab.ml.juno.node.LoraMerge;
import cab.ml.juno.player.ChatModelType;
import cab.ml.juno.player.LoraCliOptions;
import cab.ml.juno.player.LoraTrainer;
import cab.ml.juno.player.LoraTrainingLoop;

/**
 * The LoRA script's name-recall scenario through Juno's Java APIs: a fresh adapter trained in this JVM on one
 * question and answer with the LoRA command's default settings until its loss target (training time per pass is
 * the reading), then played back by the launcher with {@code --lora-play} and, after {@link LoraMerge} bakes it
 * into a copy of the model, by the launcher serving the merged file alone. Recall is the script's: the greedy
 * reply names Juno. Each server answers the question once untimed, then once timed; playback t/s is completion
 * tokens over the timed request. The GPU entry trains on the device Juno picks and serves with {@code --gpu}.
 */
final class LoraProbe implements KitProbe {

	static final String QUESTION = "What is your name?";
	static final String ANSWER = "My name is Juno";
	private static final Duration START = Duration.ofMinutes(10);

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		Path model = Path.of(args.get("model"));
		String heap = args.getOrDefault("serverHeap", "4g");
		Path work = Path.of(args.getOrDefault("logDir", ".")).resolve(id);
		boolean gpu = Boolean.parseBoolean(args.getOrDefault("gpu", "false"));
		int maxTokens = KitProbe.intArg(args, "maxTokens", 32);
		Files.createDirectories(work);
		Path adapter = work.resolve("adapter.lora");
		Path merged = work.resolve("merged.gguf");
		Files.deleteIfExists(adapter);
		Files.deleteIfExists(merged);
		LoraCliOptions o = new LoraCliOptions();
		o.trainDevice = gpu ? "auto" : "cpu";
		float lossTarget = Float.parseFloat(args.getOrDefault("lossTarget", String.valueOf(o.lossTargetQa)));
		int maxIters = KitProbe.intArg(args, "maxIters", o.maxItersQa);
		List<ProbeReport.Check> checks = new ArrayList<>();
		List<ProbeReport.Reading> readings = new ArrayList<>();
		Map<String, Object> facts = new LinkedHashMap<>();

		LoraTrainingLoop.TrainingResult tr;
		double trainMs;
		try (LoraTrainer t = LoraTrainer.open(model, adapter, o.toTrainingConfig())) {
			long t0 = System.nanoTime();
			tr = t.trainQaPairUntilResult(QUESTION, ANSWER, ChatModelType.fromPath(model.toString()), lossTarget,
					maxIters, o.earlyStop);
			trainMs = (System.nanoTime() - t0) / 1e6;
			t.save();
			facts.put("trainDevice", t.config().trainDevice());
		}
		checks.add(new ProbeReport.Check("training reaches the loss target", tr.targetReached(), String.format(
				Locale.ROOT, "%s after %d passes, loss %.4f (target %.2f)", tr.stopReason(), tr.passCount(),
				tr.finalTrainLoss(), lossTarget)));
		readings.add(new ProbeReport.Reading("training time per pass", "ms",
				KitProbe.one(tr.passCount() > 0 ? trainMs / tr.passCount() : Double.NaN), false, false));
		facts.put("passes", tr.passCount());
		facts.put("optimizerUpdates", tr.optimizerUpdateCount());
		facts.put("finalLoss", (double) tr.finalTrainLoss());
		facts.put("stopReason", tr.stopReason().name());

		List<String> playArgs = JunoProcess.localArgs(model, maxTokens);
		playArgs.addAll(List.of(gpu ? "--gpu" : "--cpu", "--lora-play", adapter.toAbsolutePath().toString()));
		Played play = play(work.resolve("lora-play"), heap, playArgs, maxTokens);
		checks.add(new ProbeReport.Check("--lora-play: the adapter answers with the trained reply", play.recalls(),
				play.detail()));
		readings.add(new ProbeReport.Reading("playback t/s, --lora-play", "t/s", KitProbe.one(play.tps), true, false));
		facts.put("reply.loraPlay", play.reply);

		LoraMerge.Result m = LoraMerge.merge(model, adapter, merged);
		facts.put("mergeAdaptersApplied", m.adaptersApplied());
		facts.put("mergeTensorsPatched", m.tensorsPatched().size());
		facts.put("mergeSkipped", m.skipped().size());
		List<String> mergedArgs = JunoProcess.localArgs(merged, maxTokens);
		mergedArgs.add(gpu ? "--gpu" : "--cpu");
		Played mp = play(work.resolve("merged"), heap, mergedArgs, maxTokens);
		checks.add(new ProbeReport.Check("the merged model loads and answers", mp.status == 200 && !mp.reply.isBlank(),
				mp.error != null ? mp.error : "HTTP " + mp.status + ", " + mp.reply.length() + " characters"));
		checks.add(new ProbeReport.Check("the merged model answers with the trained reply", mp.recalls(),
				mp.detail()));
		readings.add(new ProbeReport.Reading("playback t/s, merged model", "t/s", KitProbe.one(mp.tps), true, false));
		facts.put("reply.merged", mp.reply);
		Files.deleteIfExists(merged);
		facts.put("backend", gpu ? "gpu" : "cpu");
		return new ProbeReport(id, ProbeKind.CHECK, checks.stream().allMatch(ProbeReport.Check::passed), checks,
				readings, facts);
	}

	/** One server's answer to the question: status, greedy reply, timed t/s, or why it did not start. */
	private record Played(int status, String reply, double tps, String error) {

		boolean recalls() {
			return status == 200 && reply.toLowerCase(Locale.ROOT).contains("juno");
		}

		String detail() {
			return error != null ? error : "reply '" + reply + "'";
		}
	}

	private static Played play(Path dir, String heap, List<String> args, int maxTokens) {
		try (JunoProcess s = JunoProcess.serve(dir, heap, args, START)) {
			String body = GrammarProbe.body(s.modelId(), QUESTION, maxTokens).toString();
			s.post(GrammarProbe.PATH, body);
			long t0 = System.nanoTime();
			JunoProcess.Answer a = s.post(GrammarProbe.PATH, body);
			double ms = (System.nanoTime() - t0) / 1e6;
			int tokens = a.status() == 200 ? a.json().path("usage").path("completion_tokens").asInt(0) : 0;
			return new Played(a.status(), GrammarProbe.content(a), tokens > 0 ? tokens / (ms / 1000.0) : Double.NaN,
					a.status() == 200 ? null : GrammarProbe.describe(a));
		} catch (Exception e) {
			return new Played(0, "", Double.NaN, "the server did not answer: " + e.getMessage());
		}
	}
}
