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

import java.io.IOException;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.Flow;

import cab.ml.juno.coordinator.GenerationLoop;
import cab.ml.juno.coordinator.GenerationResult;
import cab.ml.juno.coordinator.InferenceRequest;
import cab.ml.juno.coordinator.PublisherTokenConsumer;
import cab.ml.juno.coordinator.RequestPriority;
import cab.ml.juno.coordinator.BatchConfig;
import cab.ml.juno.coordinator.InferenceApiServer;
import cab.ml.juno.coordinator.RequestScheduler;
import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.lora.LoraAdapterSet;
import cab.ml.juno.node.CudaAvailability;
import cab.ml.juno.node.CudaMatVec;
import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.ForwardPassHandlerLoader;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.GpuContext;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.node.LocalInferencePipeline;
import cab.ml.juno.node.MatVec;
import cab.ml.juno.node.ShardContext;
import cab.ml.juno.registry.ModelDescriptor;
import cab.ml.juno.registry.ModelRegistry;
import cab.ml.juno.registry.ModelStatus;
import cab.ml.juno.registry.NodeDescriptor;
import cab.ml.juno.registry.NodeStatus;
import cab.ml.juno.registry.QuantizationType;
import cab.ml.juno.registry.ShardMap;
import cab.ml.juno.registry.ShardPlanner;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatTemplateFormatter;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.GgufTokenizer;
import cab.ml.juno.tokenizer.Tokenizer;

/**
 * Facade for single-process Juno inference: loads a GGUF, builds a local pipeline,
 * and exposes chat, embedding, and scheduler access.
 */
public final class JunoPlayer implements AutoCloseable {

	private final String modelId;
	private final SamplingParams samplingParams;
	private final String byteOrder;
	private final Tokenizer tokenizer;
	private final LocalInferencePipeline pipeline;
	private final GenerationLoop loop;
	private final RequestScheduler scheduler;
	private final ModelRegistry modelRegistry;
	private final List<ForwardPassHandler> handlers;
	private final GpuContext gpuContext;

	private JunoPlayer(String modelId, SamplingParams samplingParams, String byteOrder, Tokenizer tokenizer,
			LocalInferencePipeline pipeline, GenerationLoop loop, RequestScheduler scheduler, ModelRegistry registry,
			List<ForwardPassHandler> handlers, GpuContext gpuContext) {
		this.modelId = modelId;
		this.samplingParams = samplingParams;
		this.byteOrder = byteOrder;
		this.tokenizer = tokenizer;
		this.pipeline = pipeline;
		this.loop = loop;
		this.scheduler = scheduler;
		this.modelRegistry = registry;
		this.handlers = handlers;
		this.gpuContext = gpuContext;
	}

	public static Builder builder(Path modelPath) {
		return new Builder(modelPath);
	}

	public GenerationResult chat(List<ChatMessage> messages) {
		InferenceRequest req = InferenceRequest.of(modelId, messages, samplingParams, RequestPriority.NORMAL);
		return scheduler.submitAndWait(req);
	}

	/**
	 * Stream decoded text pieces as a {@link Flow.Publisher}. Subscribe before any
	 * heavy callers dispatch work; the publisher completes when generation finishes.
	 */
	public Flow.Publisher<String> streamPublisher(List<ChatMessage> messages) {
		InferenceRequest req = InferenceRequest.of(modelId, messages, samplingParams, RequestPriority.NORMAL);
		PublisherTokenConsumer bridge = new PublisherTokenConsumer();
		scheduler.submit(req, bridge).whenComplete((res, err) -> bridge.finish());
		return bridge.publisher();
	}

	/**
	 * Embedding of the formatted chat prompt: RMS-normalized last-token hidden state
	 * before the LM head (hidden size {@link LlamaConfig#hiddenDim()}).
	 */
	public float[] embed(List<ChatMessage> messages) {
		ChatTemplateFormatter formatter = ChatTemplateFormatter.forModelType(modelId);
		String prompt = formatter.format(messages);
		int[] ids = tokenizer.encode(prompt);
		return pipeline.embedLastToken(UUID.randomUUID().toString(), ids);
	}

	public RequestScheduler scheduler() {
		return scheduler;
	}

	public GenerationLoop loop() {
		return loop;
	}

	public LocalInferencePipeline pipeline() {
		return pipeline;
	}

	public Tokenizer tokenizer() {
		return tokenizer;
	}

	public String modelId() {
		return modelId;
	}

	public ModelRegistry modelRegistry() {
		return modelRegistry;
	}

	public SamplingParams samplingParams() {
		return samplingParams;
	}

	public InferenceApiServer startApiServer(int port) {
		InferenceApiServer server = new InferenceApiServer(scheduler, modelRegistry, byteOrder);
		server.start(port);
		return server;
	}

	@Override
	public void close() {
		for (ForwardPassHandler h : handlers)
			h.releaseGpuResources();
		if (gpuContext != null)
			gpuContext.close();
	}

	public static final class Builder {

		private final Path modelPath;
		private int nodeCount = 3;
		private Path loraPlayPath;
		private boolean useGpu = true;
		private SamplingParams samplingParams = SamplingParams.defaults();
		private String byteOrder = "BE";

		private Builder(Path modelPath) {
			this.modelPath = modelPath;
		}

		public Builder nodeCount(int nodeCount) {
			this.nodeCount = Math.max(1, nodeCount);
			return this;
		}

		public Builder loraPlayPath(Path path) {
			this.loraPlayPath = path;
			return this;
		}

		public Builder useGpu(boolean useGpu) {
			this.useGpu = useGpu;
			return this;
		}

		public Builder samplingParams(SamplingParams p) {
			this.samplingParams = p;
			return this;
		}

		public Builder byteOrder(String bo) {
			this.byteOrder = "LE".equalsIgnoreCase(bo) ? "LE" : "BE";
			return this;
		}

		public JunoPlayer build() throws IOException {
			System.setProperty("juno.byteOrder", byteOrder);

			LlamaConfig config;
			Tokenizer tokenizer;
			try (GgufReader reader = GgufReader.open(modelPath)) {
				config = LlamaConfig.from(reader);
				tokenizer = GgufTokenizer.load(reader);
			}

			long vramPerLayerBytes = estimateVramPerLayer(config.hiddenDim());
			long nodeVramBytes = config.numLayers() * vramPerLayerBytes * 2;

			List<NodeDescriptor> nodes = new ArrayList<>();
			for (int i = 0; i < nodeCount; i++) {
				nodes.add(new NodeDescriptor("node-" + i, "localhost", 9092 + i, nodeVramBytes, nodeVramBytes,
						NodeStatus.READY, 1.0, Instant.now(), Instant.now()));
			}

			ShardMap shardMap = ShardPlanner.create().plan("model", config.numLayers(), vramPerLayerBytes, nodes);

			LoraAdapterSet playAdapters = null;
			if (loraPlayPath != null)
				playAdapters = LoraAdapterSet.load(loraPlayPath);

			List<ForwardPassHandler> handlers = new ArrayList<>();
			GpuContext gpuCtx = prepareGpuContext(useGpu);
			MatVec sharedBackend = (gpuCtx != null) ? new CudaMatVec(gpuCtx) : ForwardPassHandlerLoader.selectBackend();
			for (var assignment : shardMap.assignments()) {
				var context = ShardContext.from(assignment, config.vocabSize(), config.hiddenDim(), config.numHeads());
				handlers.add(ForwardPassHandlerLoader.load(modelPath, context, sharedBackend, playAdapters));
			}

			var pipeline = LocalInferencePipeline.from(shardMap, new ArrayList<>(handlers), config.vocabSize(),
					config.hiddenDim(), config.numHeads());
			var kvCache = new KVCacheManager(new GpuKVCache(512L * 1024 * 1024), new CpuKVCache(4096));
			var loop = new GenerationLoop(tokenizer, Sampler.create(), pipeline, kvCache);
			var scheduler = new RequestScheduler(1000, loop, BatchConfig.disabled());

			String filename = modelPath.getFileName().toString();
			String inferenceModelId = ChatModelType.fromPath(modelPath.toString());

			ModelRegistry registry = new ModelRegistry(ShardPlanner.create());
			long vramPerLayer = 4L * config.hiddenDim() * config.hiddenDim() * 2;
			QuantizationType quant = LlamaConfig.fromFilename(filename);
			ModelDescriptor descriptor = new ModelDescriptor(filename, config.architecture(), config.numLayers(),
					config.hiddenDim(), config.vocabSize(), config.numHeads(), vramPerLayer, quant,
					modelPath.toAbsolutePath().toString(), ModelStatus.LOADED, Instant.now());
			registry.putLoaded(descriptor);

			return new JunoPlayer(inferenceModelId, samplingParams, byteOrder, tokenizer, pipeline, loop, scheduler,
					registry, List.copyOf(handlers), gpuCtx);
		}

		private static long estimateVramPerLayer(int hiddenDim) {
			long params = 4L * hiddenDim * hiddenDim;
			return (long) (params * 2.0);
		}

		private static GpuContext prepareGpuContext(boolean useGpu) {
			if (!useGpu || !CudaAvailability.isAvailable())
				return null;
			int dev = Math.max(0, Integer.getInteger("juno.cuda.device", 0));
			if (dev >= CudaAvailability.deviceCount())
				return null;
			return GpuContext.shared(dev);
		}
	}
}
