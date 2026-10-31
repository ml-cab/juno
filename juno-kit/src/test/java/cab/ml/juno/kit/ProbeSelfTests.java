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


import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Every probe, run in-process on synthetic inputs: no model file and no GPU. The live checks run with the CPU
 * backend (nodes included); the GPU probes run their mechanics with the GPU switched off, where their
 * device-only checks must say so rather than pass. The GPU throughput probe is the CPU one with
 * {@code gpu=true}, so {@link #throughputProbeReadsBothLanesOffTheJfrSpans} covers it; the unit-test probe's
 * report parsing and command are in {@link SurefireReportsTest}.
 */
class ProbeSelfTests {

	@Test
	void envProbeReportsTheJvmsVectorWidthsAndJunosBackendProbes() {
		ProbeReport r = new EnvProbe().run("env", Map.of());
		assertThat(r.kind()).isEqualTo(ProbeKind.READING);
		assertThat((Integer) r.facts().get("intVectorBits")).isGreaterThanOrEqualTo(64);
		assertThat(r.facts()).containsKeys("floatVectorBits", "useAvx", "cuda", "rocm", "cpuThreads");
	}

	@Test
	void calibrationProbeTimesTheShippedKernel() {
		ProbeReport r = new CalibrationProbe().run("calibration",
				Map.of("rows", "256", "cols", "256", "reps", "2", "targetMs", "5", "warmupMs", "5"));
		assertThat(r.readings()).singleElement().satisfies(rd -> {
			assertThat(rd.name()).isEqualTo(CalibrationProbe.READING);
			assertThat(rd.values()).hasSize(2).allSatisfy(v -> assertThat(v).isPositive());
			assertThat(rd.scored()).isTrue();
		});
	}

	@Test
	void kernelWidthProbeTimesBothWidthsInSeparateJvms() throws Exception {
		ProbeReport r = new KernelWidthProbe().run("cpu.kernel-width", Map.of("shapes", "12:256:256", "threads", "1",
				"alternations", "1", "targetMs", "5", "warmupMs", "5"));
		assertThat(r.readings()).extracting(ProbeReport.Reading::name)
				.containsExactly("Q4_K 256x256 threads 1 128-bit", "Q4_K 256x256 threads 1 256-bit");
		assertThat(r.readings()).allSatisfy(rd -> assertThat(rd.values()).singleElement()
				.satisfies(v -> assertThat(v).isPositive()));
		assertThat(r.facts()).containsKey("Q4_K 256x256 threads 1 256-bit over 128-bit time");
	}

	@Test
	void throughputProbeReadsBothLanesOffTheJfrSpans(@TempDir Path dir) throws Exception {
		Path model = SyntheticGguf.write(dir);
		ProbeReport r = new ThroughputProbe().run("cpu.throughput.synthetic", Map.of("model", model.toString(),
				"nPrompt", "32", "nGen", "8", "warmups", "1", "reps", "2", "threads", "2"));
		ProbeReport.Reading pp = reading(r, "prefill t/s");
		ProbeReport.Reading tg = reading(r, "generation t/s");
		assertThat(pp.scored()).isTrue();
		assertThat(tg.scored()).isTrue();
		assertThat(pp.values()).isNotEmpty().allSatisfy(v -> assertThat(v).isPositive());
		assertThat(tg.values()).isNotEmpty().allSatisfy(v -> assertThat(v).isPositive());
		assertThat(reading(r, "generation allocation per token in recording").scored()).isFalse();
		assertThat(r.facts().get("generatedTokens")).isEqualTo(8);
		assertThat((Integer) r.facts().get("prefillPromptTokens")).isBetween(28, 36);
	}

	@Test
	void liveChecksProbeReportsEachCheckOfTheSuitesItRan(@TempDir Path dir) throws Exception {
		Path model = SyntheticGguf.write(dir);
		ProbeReport r = new LiveChecksProbe().run("live.synthetic",
				Map.of("model", model.toString(), "suites", "TENSOR", "backend", "cpu"));
		assertThat(r.kind()).isEqualTo(ProbeKind.CHECK);
		assertThat(r.checks()).extracting(ProbeReport.Check::name).containsExactly("7. tensor-parallel generation",
				"8. tensor-parallel greedy determinism");
		assertThat(r.passed()).isEqualTo(r.checks().stream().allMatch(ProbeReport.Check::passed));
		assertThat(r.facts()).containsEntry("architecture", "llama").containsEntry("backend", "CPU");
	}

	@Test
	void residencyProbeRunsBothModesInTheirOwnJvmsAndComparesThem(@TempDir Path dir) throws Exception {
		Path model = SyntheticGguf.write(dir);
		ProbeReport r = new GpuResidencyProbe().run("gpu.residency.synthetic", Map.of("model", model.toString(),
				"requests", "3", "nGen", "4", "memNGen", "2", "gpu", "false", "childHeap", "1g", "logDir", dir.toString()));
		assertThat(r.checks()).extracting(ProbeReport.Check::name).containsExactly(
				"region off: every request answers, the last equals the first, shorter ones are its opening",
				"region on: every request answers, the last equals the first, shorter ones are its opening",
				"region on: activation or decline reported", "greedy output identical with the region on and off",
				"device memory does not grow per request", "pipeline cluster, region on: the forked nodes answer",
				"pipeline cluster: no node JVM left", "tensor cluster, region on: the forked nodes answer",
				"tensor cluster: no node JVM left");
		assertThat(r.checks().get(0).passed()).isTrue();
		assertThat(r.checks().get(1).passed()).isTrue();
		assertThat(r.checks().get(3).passed()).isTrue();
		assertThat(r.checks().get(4).detail()).contains("no device memory reading");
		assertThat(r.checks().subList(5, 9)).allSatisfy(c -> assertThat(c.passed()).as(c.name() + ": " + c.detail())
				.isTrue());
		assertThat(r.facts()).containsKeys("pipeline.equalsLocal", "tensor.equalsLocal");
	}

	@Test
	void packedProbeChecksTheRequestShapeAndSaysThePackedPathIsNotActiveOffTheGpu(@TempDir Path dir)
			throws Exception {
		Path model = SyntheticGguf.write(dir);
		ProbeReport r = new PackedKquantProbe().run("gpu.packed.synthetic", Map.of("model", model.toString(),
				"nPrompt", "32", "nGen", "8", "gpu", "false", "schedule", "static"));
		assertThat(r.checks()).hasSize(5);
		assertThat(r.checks().subList(0, 4)).allSatisfy(c -> assertThat(c.passed()).as(c.name() + ": " + c.detail())
				.isTrue());
		assertThat(r.checks().get(4).passed()).isFalse();
		assertThat(r.checks().get(4).detail()).contains("does not show the packed path active");
		assertThat(r.passed()).isFalse();
		assertThat(r.facts()).containsKeys("ttftMs", "peakDeviceMiB");
	}

	@Test
	void consistencyProbeRejectsUnknownArchitecturesAndKeepsBatchedOutputEqualToSingle(@TempDir Path dir)
			throws Exception {
		Path model = SyntheticGguf.write(dir);
		ProbeReport r = new ConsistencyProbe().run("consistency.synthetic", Map.of("model", model.toString(),
				"architectures", "qwen35", "serverHeap", "1g", "logDir", dir.toString()));
		assertThat(r.checks()).extracting(ProbeReport.Check::name).contains(
				"local mode refuses architecture qwen35 and names it",
				"pipeline cluster refuses architecture qwen35, names it, leaves no node JVM",
				"tensor cluster refuses architecture qwen35, names it, leaves no node JVM",
				"local mode loads a supported model and exits cleanly",
				"static batching, cpu: OpenAI round 1, prompt 0 equals its single-request output",
				"static batching, cpu: native round 2, prompt 2 equals its single-request output",
				"static batching, cpu: a session request after batched stateless traffic equals single-request output",
				"static batching, cpu: no prefix-cache hit for stateless traffic",
				"static batching, cpu: native streaming joins to the native blocking answer",
				"continuous: a session request after the same stateless request equals it");
		assertThat(r.checks()).allSatisfy(c -> assertThat(c.passed()).as(c.name() + ": " + c.detail()).isTrue());
	}

	@Test
	void grammarProbeFailsClosedOnBadRequestsAndRecordsEveryConstrainedRequest(@TempDir Path dir) throws Exception {
		Path model = SyntheticGguf.write(dir);
		ProbeReport r = new GrammarProbe().run("grammar.synthetic",
				Map.of("model", model.toString(), "serverHeap", "1g", "logDir", dir.toString()));
		// What a meaningless model cannot change: refusals, the yes/no grammar's two words, the JFR counts.
		assertThat(r.checks()).filteredOn(c -> !c.name().contains("parseable")).allSatisfy(
				c -> assertThat(c.passed()).as(c.name() + ": " + c.detail()).isTrue());
		assertThat(r.checks()).extracting(ProbeReport.Check::name).contains(
				"CLI: an unsupported JSON Schema keyword fails closed",
				"CLI: --grammar-file and --json-schema-file together fail closed",
				"OpenAI: x_juno_grammar answers yes or no",
				"OpenAI: x_juno_grammar with response_format is HTTP 400 naming x_juno_grammar",
				"OpenAI: an unsupported JSON Schema keyword is HTTP 400 naming it",
				"JFR: at least 3 constrained requests, response_format server",
				"--grammar-file: an unconstrained request answers yes or no",
				"JFR: at least 1 constrained request, --grammar-file server",
				"JFR: at least 1 constrained request, --json-schema-file server",
				"JFR: at least 2 constrained requests, LoRA overlay with 2 parallel");
	}

	@Test
	void toolsProbeFailsClosedWhereTheTemplateCannotCarryToolsAndOnConflictingConstraints(@TempDir Path dir)
			throws Exception {
		Path tiny = SyntheticGguf.write(dir);
		Path qwen = java.nio.file.Files.copy(tiny, dir.resolve("qwen2.5-synthetic.gguf"));
		ProbeReport r = new ToolsProbe().run("tools.synthetic",
				Map.of("model", qwen.toString(), "model." + ToolsProbe.CHATML_MODEL, qwen.toString(),
						"model." + ToolsProbe.REFUSING_MODEL, tiny.toString(), "serverHeap", "1g", "logDir",
						dir.toString()));
		assertThat(r.checks()).extracting(ProbeReport.Check::name).contains(
				"ChatML: tools with response_format json_object is HTTP 400",
				"ChatML: tools with x_juno_grammar is HTTP 400",
				"TinyLlama template: tools are HTTP 400 naming the template",
				"TinyLlama template with a LoRA overlay: tools are HTTP 400 naming the template");
		assertThat(r.checks()).filteredOn(c -> c.name().contains("HTTP 400")).allSatisfy(
				c -> assertThat(c.passed()).as(c.name() + ": " + c.detail()).isTrue());
	}

	/**
	 * The synthetic model is llama-family, whose limit is the KV cache's 32,768 positions, so its conversation
	 * never reaches a limit: the shift path must answer, and the limit checks must fail saying why.
	 */
	@Test
	void contextPolicyProbeAnswersWithTheOptInAndSaysSoWhenNoLimitWasReached(@TempDir Path dir) throws Exception {
		Path model = SyntheticGguf.write(dir);
		ProbeReport r = new ContextPolicyProbe().run("context-policy.synthetic",
				Map.of("model", model.toString(), "limit", String.valueOf(SyntheticGguf.CONTEXT), "turns", "3",
						"notesPerTurn", "10", "fitNotes", "20", "replyTokens", "4", "decodeTokens", "200",
						"serverHeap", "1g", "logDir", dir.toString()));
		assertThat(r.checks()).extracting(ProbeReport.Check::name).containsExactly(
				"static: with x_juno_context_shift every one of 3 turns answers",
				"static: a request held open past the limit shifts during decode",
				"static: without the opt-in a later turn fails at the limit, naming the original context length",
				"continuous: with x_juno_context_shift every one of 3 turns answers",
				"continuous: a request held open past the limit shifts during decode",
				"continuous: without the opt-in a later turn fails at the limit, naming the original context length",
				"static, --context-shift on: the crossing turn answers by default and fails with an explicit false");
		assertThat(r.checks()).filteredOn(c -> c.name().contains("x_juno_context_shift") || c.name().contains("decode"))
				.hasSize(4).allSatisfy(c -> assertThat(c.passed()).as(c.name() + ": " + c.detail()).isTrue());
		assertThat(r.checks()).filteredOn(c -> c.name().contains("without the opt-in")).hasSize(2)
				.allSatisfy(c -> assertThat(c.detail()).contains("every turn answered"));
		assertThat(r.checks().get(6).detail()).contains("not run");
		assertThat(r.passed()).isFalse();
	}

	@Test
	void longPromptProbeHitsEachLengthOnBothSchedulesAndStreamsTheSameText(@TempDir Path dir) throws Exception {
		Path model = SyntheticGguf.write(dir);
		ProbeReport r = new LongPromptProbe().run("long-prompt.synthetic",
				Map.of("model", model.toString(), "lengths", "512,1024", "calibB", "10", "maxTokens", "8",
						"serverHeap", "1g", "logDir", dir.toString()));
		assertThat(r.checks()).hasSize(12).extracting(ProbeReport.Check::name).contains(
				"static 512: HTTP 200, non-empty answer", "continuous 1024: prompt tokens within 10% of 1024",
				"continuous 1024: streamed text equals non-streamed");
		assertThat(r.checks()).allSatisfy(c -> assertThat(c.passed()).as(c.name() + ": " + c.detail()).isTrue());
		assertThat(r.facts()).containsKeys("static.512.ttftMs", "continuous.1024.text");
	}

	@Test
	void cpuHotPathProbeAnswersAtEachThreadCountWithIdenticalReplies(@TempDir Path dir) throws Exception {
		Path model = SyntheticGguf.write(dir);
		ProbeReport r = new CpuHotPathProbe().run("cpu.hot-path.synthetic",
				Map.of("model", model.toString(), "threads", "1,2", "serverHeap", "1g", "logDir", dir.toString()));
		assertThat(r.checks()).extracting(ProbeReport.Check::name).containsExactly(
				"threads 1: the server runs CPU matmuls on 1 threads", "threads 1: a short greedy request decodes",
				"threads 1: a prompt of about 512 tokens prefills and answers",
				"threads 2: the server runs CPU matmuls on 2 threads", "threads 2: a short greedy request decodes",
				"threads 2: a prompt of about 512 tokens prefills and answers",
				"greedy replies identical at 1 and 2 threads");
		assertThat(r.checks()).allSatisfy(c -> assertThat(c.passed()).as(c.name() + ": " + c.detail()).isTrue());
		assertThat(CpuHotPathProbe.defaultThreads(12)).containsExactly("4", "12");
		assertThat(CpuHotPathProbe.defaultThreads(4)).containsExactly("2", "4");
		assertThat(CpuHotPathProbe.defaultThreads(1)).containsExactly("1");
	}

	@Test
	void parallelProbeReadsAggregateThroughputAtEachSettingOverConcurrentClients(@TempDir Path dir) throws Exception {
		Path model = SyntheticGguf.write(dir);
		ProbeReport r = new ParallelProbe().run("parallel.synthetic",
				Map.of("model", model.toString(), "parallel", "1,2", "sessions", "2", "maxTokens", "6", "warmups", "1",
						"reps", "2", "serverHeap", "1g", "logDir", dir.toString()));
		assertThat(r.kind()).isEqualTo(ProbeKind.READING);
		for (String p : new String[] { "1", "2" }) {
			ProbeReport.Reading tps = reading(r, "aggregate t/s, --parallel " + p);
			assertThat(tps.scored()).isTrue();
			assertThat(tps.values()).hasSize(2).allSatisfy(v -> assertThat(v).isPositive());
			ProbeReport.Reading latency = reading(r, "mean request latency, --parallel " + p);
			assertThat(latency.scored()).isFalse();
			assertThat(latency.higherIsBetter()).isFalse();
			assertThat((Integer) r.facts().get("parallel" + p + ".distinctReplies")).isPositive();
		}
		assertThat(r.facts()).containsKey("parallel2OverParallel1");
	}

	@Test
	void scheduleProbeServesEachWorkloadOnBothSchedulesAndSeesSharedContinuousSteps(@TempDir Path dir)
			throws Exception {
		Path model = SyntheticGguf.write(dir);
		ProbeReport r = new ScheduleProbe().run("schedule.synthetic",
				Map.of("model", model.toString(), "sessions", "4", "maxTokens", "16", "parallel", "4", "turns", "3",
						"serverHeap", "1g", "logDir", dir.toString()));
		assertThat(r.kind()).isEqualTo(ProbeKind.CHECK);
		assertThat(r.checks()).extracting(ProbeReport.Check::name).containsExactly(
				"static: the server runs the static schedule", "static: every request answered",
				"continuous: the server runs the continuous schedule", "continuous: every request answered",
				"continuous: concurrent requests shared decode steps");
		assertThat(r.checks()).allSatisfy(c -> assertThat(c.passed()).as(c.name() + ": " + c.detail()).isTrue());
		for (String s : new String[] { "static", "continuous" }) {
			assertThat(reading(r, s + ": aggregate t/s").values()).singleElement()
					.satisfies(v -> assertThat(v).isPositive());
			assertThat(reading(r, s + ": mean TTFT").values()).singleElement()
					.satisfies(v -> assertThat(v).isPositive());
			assertThat(reading(r, s + ": mean TPOT").values()).singleElement()
					.satisfies(v -> assertThat(v).isPositive());
			assertThat(reading(r, s + ": first turn latency").values()).hasSize(1);
			assertThat(reading(r, s + ": mean later turn latency").values()).hasSize(1);
			assertThat(r.facts()).containsKeys(s + ".prefixLookups", s + ".prefixHits", s + ".prefixHitRate");
		}
	}

	@Test
	void mixedPrefillProbeTimesShortStreamsBesideALongPromptAndSeesItsChunksMixedIn(@TempDir Path dir)
			throws Exception {
		Path model = SyntheticGguf.write(dir);
		ProbeReport r = new MixedPrefillProbe().run("mixed-prefill.synthetic",
				Map.of("model", model.toString(), "nPrompt", "96", "shorts", "2", "maxTokens", "8", "prefillBatch",
						"16", "parallel", "4", "serverHeap", "1g", "logDir", dir.toString()));
		assertThat(r.checks()).extracting(ProbeReport.Check::name).containsExactly("mixed: every stream answered",
				"mixed: the long prompt was prefilled in chunks inside decode steps", "admit: every stream answered");
		assertThat(r.checks()).allSatisfy(c -> assertThat(c.passed()).as(c.name() + ": " + c.detail()).isTrue());
		for (String m : new String[] { "mixed", "admit" }) {
			assertThat(reading(r, m + ": short-request TTFT").values()).singleElement()
					.satisfies(v -> assertThat(v).isPositive());
			assertThat(reading(r, m + ": short-request TPOT").values()).singleElement()
					.satisfies(v -> assertThat(v).isPositive());
			assertThat(reading(r, m + ": long-request TTFT").values()).singleElement()
					.satisfies(v -> assertThat(v).isPositive());
		}
		assertThat(r.facts()).containsKeys("mixed.prefillChunks", "admit.prefillChunks");
	}

	@Test
	void prefillBatchProbeReadsPrefillThroughputAtEachChunkSizeOffTheServersRecording(@TempDir Path dir)
			throws Exception {
		Path model = SyntheticGguf.write(dir);
		ProbeReport r = new PrefillBatchProbe().run("prefill-batch.synthetic",
				Map.of("model", model.toString(), "values", "1,8", "nPrompt", "48", "maxTokens", "2", "reps", "2",
						"serverHeap", "1g", "logDir", dir.toString()));
		assertThat(r.kind()).isEqualTo(ProbeKind.READING);
		for (String v : new String[] { "1", "8" }) {
			ProbeReport.Reading pp = reading(r, "prefill t/s, --prefill-batch " + v);
			assertThat(pp.scored()).isTrue();
			assertThat(pp.values()).hasSize(2).allSatisfy(x -> assertThat(x).isPositive());
			assertThat((Integer) r.facts().get("prefillBatch" + v + ".promptTokens")).isPositive();
		}
	}

	/**
	 * A random model cannot learn the answer in two passes, so recall is not asserted here; the loss target is
	 * set where the first pass reaches it, and everything else (training, saving, playback through
	 * {@code --lora-play}, merging into a GGUF and serving the merged file) must work.
	 */
	@Test
	void loraProbeTrainsPlaysBackMergesAndServesTheMergedModel(@TempDir Path dir) throws Exception {
		Path model = SyntheticGguf.write(dir);
		ProbeReport r = new LoraProbe().run("lora.synthetic", Map.of("model", model.toString(), "lossTarget", "100",
				"maxIters", "2", "maxTokens", "8", "serverHeap", "1g", "logDir", dir.toString()));
		assertThat(r.kind()).isEqualTo(ProbeKind.CHECK);
		assertThat(r.checks()).extracting(ProbeReport.Check::name).containsExactly("training reaches the loss target",
				"--lora-play: the adapter answers with the trained reply", "the merged model loads and answers",
				"the merged model answers with the trained reply");
		assertThat(r.checks().get(0).passed()).as(r.checks().get(0).detail()).isTrue();
		assertThat(r.checks().get(2).passed()).as(r.checks().get(2).detail()).isTrue();
		assertThat(r.checks().get(1).detail()).startsWith("reply '");
		assertThat(reading(r, "training time per pass").values()).singleElement()
				.satisfies(v -> assertThat(v).isPositive());
		assertThat(reading(r, "playback t/s, --lora-play").values()).singleElement()
				.satisfies(v -> assertThat(v).isPositive());
		assertThat(reading(r, "playback t/s, merged model").values()).singleElement()
				.satisfies(v -> assertThat(v).isPositive());
		assertThat(r.facts()).containsEntry("trainDevice", "cpu").containsKeys("passes", "finalLoss",
				"mergeAdaptersApplied");
		assertThat((Integer) r.facts().get("mergeAdaptersApplied")).isPositive();
	}

	/**
	 * No synthetic vision model exists (a CLIP encoder and its projector beside a language model), so the probe
	 * runs its mechanics against a text-only model: servers at both thread counts start, the image goes out as a
	 * multipart request, and every caption check must fail with the server's answer rather than pass.
	 */
	@Test
	void visionProbeSendsTheImageAtEachThreadCountAndSaysSoWhenTheModelHasNoVision(@TempDir Path dir)
			throws Exception {
		Path model = SyntheticGguf.write(dir);
		Path image = dir.resolve("image.jpg");
		java.nio.file.Files.write(image, new byte[] { (byte) 0xFF, (byte) 0xD8, (byte) 0xFF, (byte) 0xD9 });
		ProbeReport r = new VisionProbe().run("vision.synthetic", Map.of("model", model.toString(), "image",
				image.toString(), "threads", "1,2", "serverHeap", "1g", "logDir", dir.toString()));
		assertThat(r.checks()).extracting(ProbeReport.Check::name).containsExactly(
				"threads 1: a caption for the benchmark prompt", "threads 1: prompt tokens between 700 and 800",
				"threads 1: a one-sentence caption within 600 s", "threads 2: a caption for the benchmark prompt",
				"threads 2: prompt tokens between 700 and 800", "threads 2: a one-sentence caption within 600 s",
				"captions identical at 1 and 2 threads");
		assertThat(r.checks().get(0).passed()).isFalse();
		assertThat(r.checks().get(0).detail()).startsWith("HTTP 404");
		assertThat(r.passed()).isFalse();
	}

	/**
	 * The launcher script runs the player's shaded jar, which {@code mvn test} alone does not build (the kit's
	 * bootstrap and {@code mvn verify} do), so this skips, saying why, when the jar is absent.
	 */
	@Test
	void launcherProbeStartsLocalModeThroughTheScriptAnswersAndExitsCleanly(@TempDir Path dir) throws Exception {
		Path root = Path.of("").toAbsolutePath().getParent();
		try (java.util.stream.Stream<Path> jars = java.nio.file.Files.list(root.resolve("juno-player/target"))) {
			org.junit.jupiter.api.Assumptions.assumeTrue(
					jars.anyMatch(p -> p.getFileName().toString().endsWith("-shaded.jar")),
					"the launcher needs juno-player's shaded jar: mvn package -pl juno-player -am");
		} catch (java.nio.file.NoSuchFileException e) {
			org.junit.jupiter.api.Assumptions.abort("juno-player is not built");
		}
		Path model = SyntheticGguf.write(dir);
		ProbeReport r = new LauncherProbe().run("launcher.synthetic", Map.of("model", model.toString(), "root",
				root.toString(), "serverHeap", "1g", "logDir", dir.toString()));
		assertThat(r.checks()).extracting(ProbeReport.Check::name).containsExactly(
				"local mode with --api-port answers a chat request", "the launcher exits cleanly on exit",
				"lora with --grammar-file warns that constrained decoding is a no-op for LoRA training");
		assertThat(r.checks()).allSatisfy(c -> assertThat(c.passed()).as(c.name() + ": " + c.detail()).isTrue());
		assertThat(r.facts()).containsKey("launcher");
	}

	@Test
	void qualityProbeScoresTheTextTeacherForcedWithBothKernelSettings(@TempDir Path dir) throws Exception {
		Path model = SyntheticGguf.write(dir);
		ProbeReport r = new QualityProbe().run("quality.synthetic", Map.of("model", model.toString(), "tokens", "256"));
		assertThat(r.readings()).extracting(ProbeReport.Reading::name).containsExactly("mean NLL, integer kernels",
				"mean NLL, float-activation kernels", "argmax agreement, first 256 positions");
		assertThat(r.readings()).allSatisfy(rd -> {
			assertThat(rd.scored()).isFalse();
			assertThat(rd.values()).singleElement().satisfies(v -> assertThat(v).isPositive());
		});
		assertThat(r.checks()).singleElement().satisfies(c -> {
			assertThat(c.name()).isEqualTo("integer-kernel NLL at most 1.01x the float-activation NLL");
			assertThat(c.passed()).as(c.detail()).isTrue();
		});
		assertThat(r.facts()).containsEntry("tokens", 256);
	}

	@Test
	void theKitsQualityTextIsTheNllTestsText() throws Exception {
		Path original = Path.of("..", "node", "src", "test", "resources", "cab", "ml", "juno", "node",
				"model-quality-nll.txt");
		assertThat(QualityProbe.text()).isEqualTo(java.nio.file.Files.readString(original));
	}

	private static ProbeReport.Reading reading(ProbeReport r, String name) {
		return r.readings().stream().filter(x -> x.name().equals(name)).findFirst().orElseThrow();
	}
}
