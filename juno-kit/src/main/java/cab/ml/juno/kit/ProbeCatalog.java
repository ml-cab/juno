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


import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Every probe the kit can run. Reference costs are seconds on the reference host (Xeon E5-1650 v2, 12
 * threads, GTX 1080), rounded up from its end-to-end runs of 2026-10-28; {@link #REFERENCE_CALIBRATION_MS} is
 * the first run's calibration median. The ROCm unit-test probe's cost is the CUDA one's. The kernel-width probe cannot run there (128-bit integer vectors), so its
 * cost is an estimate: 24 kernel-timing JVMs of about 7 s each.
 * Priority 0 runs in every run; otherwise probes that unblock a plan row come first under {@code --budget}.
 */
final class ProbeCatalog {

	/** The reference host's calibration median, ms per call of the Q4_K 4096x4096 decode kernel. */
	static final double REFERENCE_CALIBRATION_MS = 0.9075;

	private static final Set<String> QUICK_CPU_GPU = Set.of("quick", "cpu", "gpu");
	private static final Set<String> QUICK_CPU = Set.of("quick", "cpu");
	private static final Set<String> QUICK_GPU = Set.of("quick", "gpu");
	private static final Set<String> GPU = Set.of("gpu");
	private static final Set<String> CPU = Set.of("cpu");

	/** The modules whose tests need no model file and no GPU, as the build documentation lists them. */
	static final String UNIT_TEST_MODULES = "tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,vision,"
			+ "metrics,juno-player";

	/**
	 * Each GPU probe's own peak device memory on the reference host (GTX 1080, 8 GiB) in MiB, the probe and every
	 * process it starts, from the kit's runs there. A probe needs a GPU with at least this, rounded up to 512 MiB:
	 * with less it would fail on memory, not on what it checks.
	 */
	static final Map<String, Integer> MEASURED_DEVICE_PEAK_MIB = Map.ofEntries(Map.entry("gpu.live.tinyllama", 2748),
			Map.entry("gpu.throughput.tinyllama.512", 1134), Map.entry("gpu.throughput.tinyllama.2048", 1666),
			Map.entry("gpu.residency.tinyllama", 2742), Map.entry("gpu.packed-kquant.tinyllama.static", 1154),
			Map.entry("gpu.packed-kquant.tinyllama.continuous", 960), Map.entry("gpu.live.qwen2.5-3b", 6960),
			Map.entry("gpu.live.phi-3.5-mini", 7596),
			Map.entry("gpu.live.mistral-7b", 7306), Map.entry("gpu.live.qwen3-1.7b", 5328),
			Map.entry("gpu.consistency.tinyllama", 950), Map.entry("gpu.context-policy.phi-3.5-mini", 5038),
			Map.entry("gpu.long-prompt.tinyllama", 1142), Map.entry("gpu.long-prompt.mistral-7b", 5536),
			Map.entry("gpu.throughput.tinyllama-1.1b.128", 992), Map.entry("gpu.throughput.qwen2.5-3b.128", 2424),
			Map.entry("gpu.throughput.qwen2.5-3b.512", 2640), Map.entry("gpu.throughput.qwen2.5-3b.2048", 3420),
			Map.entry("gpu.throughput.phi-3.5-mini.128", 2812), Map.entry("gpu.throughput.phi-3.5-mini.512", 3234),
			Map.entry("gpu.throughput.phi-3.5-mini.2048", 4726),
			Map.entry("gpu.throughput.mistral-7b.128", 4660), Map.entry("gpu.throughput.mistral-7b.512", 4986),
			Map.entry("gpu.throughput.mistral-7b.2048", 6300), Map.entry("gpu.parallel.tinyllama", 1002),
			Map.entry("gpu.schedule.tinyllama", 1002), Map.entry("gpu.mixed-prefill.tinyllama", 936),
			Map.entry("gpu.prefill-batch.tinyllama", 922), Map.entry("gpu.lora.tinyllama", 4066),
			Map.entry("gpu.vision.moondream2", 118), Map.entry("gpu.unit-tests", 4144));

	static long deviceMinimum(int peakMiB) {
		return (long) ((peakMiB + 511) / 512 * 512) << 20;
	}

	private ProbeCatalog() {
	}

	static List<Probe> all() {
		return withMeasuredDeviceMemory(entries());
	}

	private static List<Probe> withMeasuredDeviceMemory(List<Probe> probes) {
		return probes.stream().map(p -> {
			Integer peak = MEASURED_DEVICE_PEAK_MIB.get(p.id());
			return peak == null ? p
					: new Probe(p.id(), p.title(), p.kind(), p.suites(),
							p.requirements().withDeviceMemory(deviceMinimum(peak)), p.models(), p.refCostSeconds(),
							p.priority(), p.mainClass(), p.args(), p.timeoutSeconds(), p.heap());
		}).toList();
	}

	private static List<Probe> entries() {
		CatalogModel tiny = CatalogModel.byId("tinyllama-1.1b");
		return List.of(
				new Probe("env", "Environment: JVM vector widths, Juno's CUDA and ROCm probes", ProbeKind.READING,
						QUICK_CPU_GPU, Requirements.cpu(0), List.of(), 5, 0, "EnvProbe", Map.of(), 300, "512m"),
				new Probe("calibration", "Calibration: Q4_K decode kernel, all threads", ProbeKind.READING,
						QUICK_CPU_GPU, Requirements.cpu(2L << 30), List.of(), 10, 0, "CalibrationProbe", Map.of(), 900,
						"2g"),
				new Probe("cpu.kernel-width", "CPU integer kernels, 128-bit against 256-bit, 1 and all threads",
						ProbeKind.READING, QUICK_CPU, Requirements.intVectorBits(256), List.of(), 170, 1,
						"KernelWidthProbe", Map.of(), 3600, "512m"),
				new Probe("cpu.live.tinyllama", "Live checks 1 to 10 on TinyLlama, every check and node on the CPU",
						ProbeKind.CHECK, QUICK_CPU, Requirements.cpu(8L << 30), List.of(tiny.id()), 250, 1,
						"LiveChecksProbe", Map.of("backend", "cpu"), 3600, "4g"),
				new Probe("cpu.throughput.tinyllama.128", "CPU throughput, TinyLlama, 128-token prompt",
						ProbeKind.READING, QUICK_CPU, Requirements.cpu(6L << 30), List.of(tiny.id()), 60, 1,
						"ThroughputProbe", Map.of("nPrompt", "128", "gpu", "false"), 2400, tiny.heap()),
				new Probe("cpu.throughput.tinyllama.512", "CPU throughput, TinyLlama, 512-token prompt",
						ProbeKind.READING, QUICK_CPU, Requirements.cpu(6L << 30), List.of(tiny.id()), 130, 2,
						"ThroughputProbe", Map.of("nPrompt", "512", "gpu", "false"), 3600, tiny.heap()),
				new Probe("gpu.live.tinyllama", "Live checks 1 to 10 on TinyLlama, every check and node on CUDA",
						ProbeKind.CHECK, QUICK_GPU, Requirements.cuda(8L << 30), List.of(tiny.id()), 100, 1,
						"LiveChecksProbe", Map.of("backend", "cuda"), 3600, "4g"),
				new Probe("gpu.throughput.tinyllama.512", "GPU throughput, TinyLlama, 512-token prompt",
						ProbeKind.READING, QUICK_GPU, Requirements.gpu(6L << 30), List.of(tiny.id()), 15, 1,
						"ThroughputProbe", Map.of("nPrompt", "512", "gpu", "true"), 2400, tiny.heap()),
				new Probe("gpu.throughput.tinyllama.2048", "GPU throughput, TinyLlama, 2048-token prompt",
						ProbeKind.READING, QUICK_GPU, Requirements.gpu(6L << 30), List.of(tiny.id()), 20, 2,
						"ThroughputProbe", Map.of("nPrompt", "2048", "gpu", "true"), 2400, tiny.heap()),
				new Probe("gpu.residency.tinyllama", "Device-resident decode region off against on, TinyLlama",
						ProbeKind.CHECK, QUICK_GPU, Requirements.cuda(8L << 30), List.of(tiny.id()), 20, 1,
						"GpuResidencyProbe", Map.of(), 3600, "512m"),
				new Probe("gpu.packed-kquant.tinyllama.static", "Packed K-quant prefill matmul, TinyLlama, static schedule",
						ProbeKind.CHECK, QUICK_GPU, Requirements.cuda(6L << 30), List.of(tiny.id()), 20, 2,
						"PackedKquantProbe", Map.of("schedule", "static"), 2400, tiny.heap()),
				new Probe("gpu.packed-kquant.tinyllama.continuous",
						"Packed K-quant prefill matmul, TinyLlama, continuous schedule", ProbeKind.CHECK, QUICK_GPU,
						Requirements.cuda(6L << 30), List.of(tiny.id()), 80, 2, "PackedKquantProbe",
						Map.of("schedule", "continuous"), 2400, tiny.heap()),
				live("qwen2.5-3b", "Qwen2.5-3B", false, "16g", 24, 505),
				live("phi-3.5-mini", "Phi-3.5-mini", false, "8g", 24, 870),
				live("mistral-7b", "Mistral 7B", false, "24g", 32, 1545),
				live("qwen3-1.7b", "Qwen3-1.7B", false, "16g", 20, 605),
				live("qwen2.5-3b", "Qwen2.5-3B", true, "16g", 24, 150),
				live("phi-3.5-mini", "Phi-3.5-mini", true, "12g", 24, 210),
				live("mistral-7b", "Mistral 7B", true, "24g", 32, 305),
				live("qwen3-1.7b", "Qwen3-1.7B", true, "16g", 20, 130),
				new Probe("consistency.tinyllama",
						"Unsupported architectures refused; static batching and sessions equal single requests, CPU",
						ProbeKind.CHECK, QUICK_CPU, Requirements.cpu(8L << 30), List.of(tiny.id()), 95, 1,
						"ConsistencyProbe", Map.of(), 3600, "512m"),
				new Probe("gpu.consistency.tinyllama", "Static batching equals single requests, GPU", ProbeKind.CHECK,
						QUICK_GPU, Requirements.gpu(8L << 30), List.of(tiny.id()), 15, 2, "ConsistencyProbe",
						Map.of("gpu", "true"), 3600, "512m"),
				new Probe("grammar.tinyllama", "Constrained decoding over REST and the CLI, with a LoRA overlay, CPU",
						ProbeKind.CHECK, QUICK_CPU, Requirements.cpu(10L << 30), List.of(tiny.id()), 100, 1,
						"GrammarProbe", Map.of(), 3600, tiny.heap()),
				new Probe("tools.qwen2.5-3b", "Tool calling on a ChatML model; refused on TinyLlama's template, CPU",
						ProbeKind.CHECK, CPU, Requirements.cpu(12L << 30),
						List.of(ToolsProbe.CHATML_MODEL, ToolsProbe.REFUSING_MODEL), 600, 2, "ToolsProbe", Map.of(),
						5400, tiny.heap()),
				new Probe("context-policy.phi-3.5-mini",
						"Context shift opt-in, decode-time shift and the limit error, both schedules, CPU", ProbeKind.CHECK,
						CPU, Requirements.cpu(20L << 30), List.of("phi-3.5-mini"), 11250, 3, "ContextPolicyProbe",
						Map.of("limit", "4096", "turns", "3", "notesPerTurn", "85", "serverHeap", "16g"), 33800, "512m"),
				new Probe("gpu.context-policy.phi-3.5-mini",
						"Context shift opt-in, decode-time shift and the limit error, both schedules, GPU", ProbeKind.CHECK,
						GPU, Requirements.gpu(20L << 30), List.of("phi-3.5-mini"), 260, 2, "ContextPolicyProbe",
						Map.of("limit", "4096", "turns", "3", "notesPerTurn", "85", "gpu", "true", "serverHeap", "16g"), 5400,
						"512m"),
				new Probe("long-prompt.tinyllama", "Prompts of 128, 512 and 2048 tokens, both schedules, CPU",
						ProbeKind.CHECK, QUICK_CPU, Requirements.cpu(8L << 30), List.of(tiny.id()), 630, 1,
						"LongPromptProbe", Map.of("serverHeap", "4g"), 3600, "512m"),
				new Probe("long-prompt.mistral-7b", "Prompts of 128, 512 and 2048 tokens, both schedules, CPU",
						ProbeKind.CHECK, CPU, Requirements.cpu(12L << 30), List.of("mistral-7b"), 3855, 2,
						"LongPromptProbe", Map.of("serverHeap", "8g"), 11600, "512m"),
				new Probe("gpu.long-prompt.tinyllama", "Prompts of 128, 512 and 2048 tokens, both schedules, GPU",
						ProbeKind.CHECK, QUICK_GPU, Requirements.gpu(8L << 30), List.of(tiny.id()), 90, 1,
						"LongPromptProbe", Map.of("gpu", "true", "serverHeap", "4g"), 3600, "512m"),
				new Probe("gpu.long-prompt.mistral-7b", "Prompts of 128, 512 and 2048 tokens, both schedules, GPU",
						ProbeKind.CHECK, GPU, Requirements.gpu(12L << 30), List.of("mistral-7b"), 95, 2,
						"LongPromptProbe", Map.of("gpu", "true", "serverHeap", "8g"), 5400, "512m"),
				hotPath("tinyllama-1.1b", QUICK_CPU, "4g", 8, 120, 1),
				hotPath("qwen2.5-3b", CPU, "6g", 10, 255, 2),
				hotPath("phi-3.5-mini", CPU, "6g", 10, 330, 2),
				hotPath("mistral-7b", CPU, "9g", 14, 580, 2),
				quality("tinyllama-1.1b", QUICK_CPU, 460, 1),
				quality("qwen2.5-3b", CPU, 1430, 2),
				quality("phi-3.5-mini", CPU, 4170, 2),
				quality("mistral-7b", CPU, 2840, 2),
				throughput("tinyllama-1.1b", 2048, false, CPU, 6, 465, 2),
				throughput("tinyllama-1.1b", 128, true, GPU, 6, 10, 2),
				throughput("qwen2.5-3b", 128, false, CPU, 8, 150, 2),
				throughput("qwen2.5-3b", 512, false, CPU, 8, 345, 2),
				throughput("qwen2.5-3b", 2048, false, CPU, 8, 1180, 3),
				throughput("phi-3.5-mini", 128, false, CPU, 10, 195, 2),
				throughput("phi-3.5-mini", 512, false, CPU, 10, 440, 2),
				throughput("phi-3.5-mini", 2048, false, CPU, 16, 2000, 3, "12g"),
				throughput("mistral-7b", 128, false, CPU, 14, 315, 2),
				throughput("mistral-7b", 512, false, CPU, 14, 750, 2),
				throughput("mistral-7b", 2048, false, CPU, 14, 2795, 3),
				throughput("qwen2.5-3b", 128, true, GPU, 8, 20, 2),
				throughput("qwen2.5-3b", 512, true, GPU, 8, 25, 2),
				throughput("qwen2.5-3b", 2048, true, GPU, 8, 35, 2),
				throughput("phi-3.5-mini", 128, true, GPU, 10, 15, 2),
				throughput("phi-3.5-mini", 512, true, GPU, 10, 20, 2),
				throughput("phi-3.5-mini", 2048, true, GPU, 16, 40, 2, "12g"),
				throughput("mistral-7b", 128, true, GPU, 14, 25, 2),
				throughput("mistral-7b", 512, true, GPU, 14, 25, 2),
				throughput("mistral-7b", 2048, true, GPU, 14, 50, 2),
				schedule("parallel", PARALLEL_TITLE, ProbeKind.READING, "ParallelProbe", false, 890),
				schedule("parallel", PARALLEL_TITLE, ProbeKind.READING, "ParallelProbe", true, 140),
				schedule("schedule", SCHEDULE_TITLE, ProbeKind.CHECK, "ScheduleProbe", false, 315),
				schedule("schedule", SCHEDULE_TITLE, ProbeKind.CHECK, "ScheduleProbe", true, 60),
				schedule("mixed-prefill", MIXED_TITLE, ProbeKind.CHECK, "MixedPrefillProbe", false, 55),
				schedule("mixed-prefill", MIXED_TITLE, ProbeKind.CHECK, "MixedPrefillProbe", true, 25),
				schedule("prefill-batch", PREFILL_BATCH_TITLE, ProbeKind.READING, "PrefillBatchProbe", false, 260),
				schedule("prefill-batch", PREFILL_BATCH_TITLE, ProbeKind.READING, "PrefillBatchProbe", true, 70),
				new Probe("lora.tinyllama", LORA_TITLE + ", CPU", ProbeKind.CHECK, CPU, Requirements.cpu(10L << 30),
						List.of(tiny.id()), 775, 2, "LoraProbe", Map.of("serverHeap", "4g"), 5400, "4g"),
				new Probe("gpu.lora.tinyllama", LORA_TITLE + ", GPU", ProbeKind.CHECK, GPU, Requirements.gpu(10L << 30),
						List.of(tiny.id()), 70, 2, "LoraProbe", Map.of("gpu", "true", "serverHeap", "4g"), 3600, "4g"),
				new Probe("vision.moondream2", VISION_TITLE + ", CPU at two thread counts", ProbeKind.CHECK, CPU,
						Requirements.cpu(10L << 30), List.of("moondream2"), 1080, 2, "VisionProbe",
						Map.of("serverHeap", CatalogModel.byId("moondream2").heap()), 5400, "512m"),
				new Probe("gpu.vision.moondream2", VISION_TITLE + ", GPU", ProbeKind.CHECK, GPU, Requirements.gpu(10L << 30),
						List.of("moondream2"), 430, 2, "VisionProbe",
						Map.of("gpu", "true", "serverHeap", CatalogModel.byId("moondream2").heap()), 3600, "512m"),
				new Probe("launcher.tinyllama", "The juno launcher: local mode with --api-port answers and exits cleanly; "
						+ "the LoRA grammar warning", ProbeKind.CHECK, QUICK_CPU, Requirements.cpu(6L << 30),
						List.of(tiny.id()), 15, 1, "LauncherProbe", Map.of("serverHeap", "4g"), 1800, "512m"),
				new Probe("cpu.unit-tests", "Unit tests outside the GPU groups, the unit-test modules", ProbeKind.CHECK,
						CPU, Requirements.cpu(8L << 30), List.of(), 2200, 1, "UnitTestsProbe",
						Map.of("excludedGroups", "gpu,rocm", "modules", UNIT_TEST_MODULES), 6600, "512m"),
				new Probe("gpu.unit-tests", "Unit tests tagged gpu, node module and what it needs", ProbeKind.CHECK, GPU,
						Requirements.cuda(4L << 30), List.of(), 1450, 1, "UnitTestsProbe",
						Map.of("groups", "gpu", "modules", "node"), 5400, "512m"),
				new Probe("rocm.unit-tests", "Unit tests tagged rocm, node module and what it needs", ProbeKind.CHECK, GPU,
						Requirements.rocm(4L << 30), List.of(), 1450, 1, "UnitTestsProbe",
						Map.of("groups", "rocm", "modules", "node"), 5400, "512m"));
	}

	/**
	 * Live checks 1 to 10 on one more model family. On the CPU, checks 9 and 10 hold the weights and a dense KV
	 * sized to the model's whole context on the heap, hence the heaps; {@code ramGiB} covers the larger of
	 * that heap and the three forked nodes' heaps.
	 */
	private static Probe live(String modelId, String name, boolean cuda, String heap, int ramGiB, int cost) {
		String backend = cuda ? "cuda" : "cpu";
		String prefix = cuda ? "gpu" : "cpu";
		return new Probe(prefix + ".live." + modelId, "Live checks 1 to 10 on " + name + ", every check and node on "
				+ (cuda ? "CUDA" : "the CPU"), ProbeKind.CHECK, cuda ? GPU : CPU,
				cuda ? Requirements.cuda((long) ramGiB << 30) : Requirements.cpu((long) ramGiB << 30),
				List.of(modelId), cost, 2, "LiveChecksProbe", Map.of("backend", backend), 7200, heap);
	}

	/** The CPU hot-path smoke on one model, at 4 threads and at every processor; heaps as the smoke sets them. */
	private static Probe hotPath(String modelId, Set<String> suites, String heap, int ramGiB, int cost, int priority) {
		return new Probe("cpu.hot-path." + modelId, "CPU decode and a 512-token prompt at two thread counts, "
				+ CatalogModel.byId(modelId).title(), ProbeKind.CHECK, suites, Requirements.cpu((long) ramGiB << 30),
				List.of(modelId), cost, priority, "CpuHotPathProbe", Map.of("serverHeap", heap), 3600, "512m");
	}

	/** Mean NLL over the quality text on one model, integer and float-activation kernels, CPU. */
	private static Probe quality(String modelId, Set<String> suites, int cost, int priority) {
		CatalogModel m = CatalogModel.byId(modelId);
		return new Probe("quality." + modelId, "Mean NLL over 2048 tokens, integer and float-activation kernels, "
				+ m.title(), ProbeKind.CHECK, suites, Requirements.cpu(8L << 30), List.of(modelId), cost, priority,
				"QualityProbe", Map.of("childHeap", m.heap()), Math.max(7200, 3 * cost), "512m");
	}

	/** The throughput harness's two lanes, prefill at {@code nPrompt} and generation, on one model and backend. */
	private static Probe throughput(String modelId, int nPrompt, boolean gpu, Set<String> suites, int ramGiB, int cost,
			int priority) {
		return throughput(modelId, nPrompt, gpu, suites, ramGiB, cost, priority, CatalogModel.byId(modelId).heap());
	}

	/** With a heap other than the model's, where the prompt's KV outgrows it. */
	private static Probe throughput(String modelId, int nPrompt, boolean gpu, Set<String> suites, int ramGiB, int cost,
			int priority, String heap) {
		CatalogModel m = CatalogModel.byId(modelId);
		return new Probe((gpu ? "gpu" : "cpu") + ".throughput." + modelId + "." + nPrompt,
				(gpu ? "GPU" : "CPU") + " throughput, " + m.title() + ", " + nPrompt + "-token prompt", ProbeKind.READING,
				suites, gpu ? Requirements.gpu((long) ramGiB << 30) : Requirements.cpu((long) ramGiB << 30),
				List.of(modelId), cost, priority, "ThroughputProbe",
				Map.of("nPrompt", String.valueOf(nPrompt), "gpu", String.valueOf(gpu)), Math.max(5400, 3 * cost), heap);
	}

	private static final String PARALLEL_TITLE = "--parallel 1 against 8 over 8 concurrent clients";
	private static final String SCHEDULE_TITLE = "Static against continuous schedule: blocking, streamed, prefix-cache "
			+ "session";
	private static final String MIXED_TITLE = "Mixed chunked prefill against prefill at admission, continuous schedule";
	private static final String PREFILL_BATCH_TITLE = "Prefill throughput at --prefill-batch 1 and 32";
	private static final String VISION_TITLE = "Image to text, moondream2: latency, decode t/s, the one-sentence "
			+ "caption";
	private static final String LORA_TITLE = "LoRA name recall: train to the loss target, --lora-play, merge and "
			+ "reload, TinyLlama";

	/** One of the schedule scripts' workloads on TinyLlama, through the launcher; GPU entries in the GPU suite. */
	private static Probe schedule(String name, String title, ProbeKind kind, String mainClass, boolean gpu, int cost) {
		return new Probe((gpu ? "gpu." : "") + name + ".tinyllama", title + ", TinyLlama, " + (gpu ? "GPU" : "CPU"), kind,
				gpu ? GPU : CPU, gpu ? Requirements.gpu(8L << 30) : Requirements.cpu(8L << 30), List.of("tinyllama-1.1b"),
				cost, 2, mainClass, gpu ? Map.of("gpu", "true", "serverHeap", "4g") : Map.of("serverHeap", "4g"), 5400,
				"512m");
	}

	static Probe byId(String id) {
		for (Probe p : all())
			if (p.id().equals(id))
				return p;
		throw new IllegalArgumentException("no probe " + id);
	}
}
