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
package cab.ml.juno.coordinator;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.node.InferencePipeline;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.tokenizer.ChatTemplateFormatter;
import cab.ml.juno.tokenizer.Tokenizer;

/**
 * Iteration-level running set: overlapping decode and chunked prefill share
 * engine steps. Fairness prefers decode when the step slot budget is full.
 * Stream and non-stream members are first-class.
 */
final class ContinuousBatchEngine {

	private static final Logger log = Logger.getLogger(ContinuousBatchEngine.class.getName());
	private static final String LORA_PLAY_PROPERTY = "juno.lora.play.path";
	/** When {@code false}, admit runs full prefill (pre-mix baseline for bake-off). Default {@code true}. */
	static final String MIXED_PREFILL_PROPERTY = "juno.continuous.mixedPrefill";

	private final GenerationLoop loop;
	private final int maxRunning;
	private final long admitWindowMs;
	private final boolean mixedPrefill;
	private final PriorityBlockingQueue<Pending> waiting = new PriorityBlockingQueue<>();
	private final List<Slot> running = new ArrayList<>();

	private volatile boolean active = true;
	private int admittedSinceLastStep;

	ContinuousBatchEngine(GenerationLoop loop, int maxRunning, long admitWindowMs) {
		if (loop == null)
			throw new IllegalArgumentException("loop must not be null");
		if (maxRunning < 1)
			throw new IllegalArgumentException("maxRunning must be >= 1");
		if (admitWindowMs < 0)
			throw new IllegalArgumentException("admitWindowMs must be >= 0");
		this.loop = loop;
		this.maxRunning = maxRunning;
		this.admitWindowMs = admitWindowMs;
		this.mixedPrefill = mixedPrefillEnabled();
		if (!mixedPrefill)
			log.info("continuous mixed prefill disabled — admit-time full prefill (bake-off baseline)");
	}

	static boolean mixedPrefillEnabled() {
		String raw = System.getProperty(MIXED_PREFILL_PROPERTY, "true");
		return !"false".equalsIgnoreCase(raw.strip()) && !"0".equals(raw.strip());
	}

	void start() {
		Thread.ofVirtual().name("continuous-engine").start(this::runLoop);
	}

	void shutdown() {
		active = false;
	}

	void submit(InferenceRequest request, TokenConsumer consumer, CompletableFuture<GenerationResult> future) {
		waiting.offer(new Pending(request, consumer, future));
	}

	private void runLoop() {
		while (active) {
			try {
				if (running.isEmpty()) {
					long wait = Math.max(admitWindowMs, 50L);
					Pending first = waiting.poll(wait, TimeUnit.MILLISECONDS);
					if (first == null)
						continue;
					admit(first);
				}
				drainAdmit();
				if (running.isEmpty())
					continue;
				engineStep();
				coalesceAdmit();
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
				break;
			} catch (Exception e) {
				log.warning("Continuous engine error: " + e.getMessage());
				failAllRunning(e);
			}
		}
		log.fine("Continuous engine loop stopped");
	}

	private void drainAdmit() {
		while (running.size() < maxRunning) {
			Pending p = waiting.poll();
			if (p == null)
				break;
			admit(p);
		}
	}

	private void coalesceAdmit() throws InterruptedException {
		if (running.size() >= maxRunning || admitWindowMs <= 0)
			return;
		Pending p = waiting.poll(admitWindowMs, TimeUnit.MILLISECONDS);
		if (p != null) {
			admit(p);
			drainAdmit();
		}
	}

	private void admit(Pending pending) {
		try {
			Slot slot = createSlot(pending);
			running.add(slot);
			admittedSinceLastStep++;
		} catch (Exception e) {
			log.warning("Admit failed for " + pending.request().requestId() + ": " + e.getMessage());
			pending.future().completeExceptionally(e);
		}
	}

	/**
	 * Admit without blocking on full prefill when mixed mode is on — remaining
	 * prompt work runs as ubatch chunks mixed into later engine steps. When
	 * {@link #MIXED_PREFILL_PROPERTY} is {@code false}, runs admit-time full
	 * prefill (bake-off baseline).
	 */
	private Slot createSlot(Pending pending) {
		InferenceRequest request = pending.request();
		Tokenizer tokenizer = loop.tokenizer();
		KVCacheManager kvCache = loop.kvCache();
		InferencePipeline pipeline = loop.pipeline();

		ChatTemplateFormatter formatter = ChatTemplateFormatter.forModelType(request.modelId());
		String prompt = formatter.format(request.messages());
		int[] promptIds = tokenizer.encode(prompt);

		final String kvKey = request.kvCacheKey();
		final boolean hasSession = request.sessionId() != null;
		boolean loraPlay = System.getProperty(LORA_PLAY_PROPERTY) != null
				&& !System.getProperty(LORA_PLAY_PROPERTY).isBlank();

		int startPos = 0;
		var prefixMatch = kvCache.findLongestPrefix(promptIds);
		boolean hadCacheHit = false;
		if (hasSession && !loraPlay && prefixMatch.isHit()) {
			startPos = prefixMatch.matchedTokens();
			hadCacheHit = true;
			log.info("Prefix cache hit: " + startPos + "/" + promptIds.length + " tokens cached (session=" + kvKey
					+ ")");
		}

		pending.consumer().onPrefillStart(promptIds.length);
		ContinuousPrefillState prefill;
		if (!mixedPrefill) {
			int prefillSteps = promptIds.length - 1 - startPos;
			if (prefillSteps > 0) {
				PrefillChunker.run(pipeline, loop.prefillMode(), loop.prefillBatchSize(), kvKey, promptIds, startPos);
			}
			pending.consumer().onPrefillComplete();
			prefill = ContinuousPrefillState.start(promptIds, promptIds.length > 0 ? promptIds.length - 1 : 0);
		} else {
			prefill = ContinuousPrefillState.start(promptIds, startPos);
		}

		int decodeBase = promptIds.length > 0 ? promptIds.length - 1 : 0;
		Slot slot = new Slot(request, pending.consumer(), pending.future(), Instant.now(), kvKey, hasSession,
				promptIds.clone(), promptIds.length, decodeBase, hadCacheHit, prefill);
		slot.stream = tokenizer.openStreamContext();
		if (prefill.isComplete()) {
			slot.phase = Phase.DECODE;
			if (mixedPrefill)
				pending.consumer().onPrefillComplete();
		}
		return slot;
	}

	private void engineStep() {
		ContinuousMixedStepPolicy.Plan<Slot> plan = ContinuousMixedStepPolicy.plan(running, maxRunning,
				loop.prefillBatchSize(), Slot::isDecode, s -> s.prefill.remainingTokens());

		int admitted = admittedSinceLastStep;
		admittedSinceLastStep = 0;

		int prefillChunks = runPrefillChunks(plan.prefill());
		List<Slot> justFinished = runDecode(plan.decode());
		retireFinished(justFinished);

		ContinuousStepEvent ev = new ContinuousStepEvent();
		ev.runningSetSize = running.size() + justFinished.size();
		ev.decodeBatchSize = plan.decode().size();
		ev.prefillChunks = prefillChunks;
		ev.prefillTokens = plan.prefill().stream().mapToInt(ContinuousMixedStepPolicy.PrefillWork::tokenBudget).sum();
		ev.admitted = admitted;
		ev.retired = justFinished.size();
		ev.commit();
	}

	private int runPrefillChunks(List<ContinuousMixedStepPolicy.PrefillWork<Slot>> works) {
		if (works.isEmpty())
			return 0;
		InferencePipeline pipeline = loop.pipeline();
		PrefillMode mode = loop.prefillMode();
		int done = 0;
		for (ContinuousMixedStepPolicy.PrefillWork<Slot> work : works) {
			Slot s = work.member();
			ContinuousPrefillState.Chunk chunk = s.prefill.nextChunk(work.tokenBudget());
			if (chunk.isEmpty())
				continue;
			runOneChunk(pipeline, mode, s, chunk);
			s.prefill.advance(chunk.tokenCount());
			done++;
			if (s.prefill.isComplete()) {
				s.phase = Phase.DECODE;
				s.consumer.onPrefillComplete();
			}
		}
		return done;
	}

	private static void runOneChunk(InferencePipeline pipeline, PrefillMode mode, Slot slot,
			ContinuousPrefillState.Chunk chunk) {
		if (mode == PrefillMode.SINGLE) {
			for (int i = 0; i < chunk.tokenCount(); i++) {
				int pos = chunk.startPos() + i;
				int[] slice = Arrays.copyOfRange(slot.allTokens, 0, pos + 1);
				pipeline.forward(slot.kvKey, slice, pos);
			}
			return;
		}
		pipeline.prefillBatch(slot.kvKey, chunk.tokens(), chunk.startPos());
	}

	private List<Slot> runDecode(List<Slot> batch) {
		if (batch.isEmpty())
			return List.of();

		List<Slot> active = new ArrayList<>(batch.size());
		List<String> ids = new ArrayList<>(batch.size());
		List<int[]> toks = new ArrayList<>(batch.size());
		List<Integer> pos = new ArrayList<>(batch.size());
		for (Slot s : batch) {
			if (s.generated.size() >= s.request.samplingParams().maxTokens())
				continue;
			active.add(s);
			ids.add(s.kvKey);
			toks.add(s.allTokens);
			pos.add(s.decodeBase + s.generated.size());
		}
		if (active.isEmpty())
			return List.of();

		float[][] logitsBatch = loop.pipeline().forwardBatch(ids, toks, pos);
		Sampler sampler = loop.sampler();
		Tokenizer tokenizer = loop.tokenizer();

		List<Slot> justFinished = new ArrayList<>();
		for (int j = 0; j < active.size(); j++) {
			Slot s = active.get(j);
			float[] logits = logitsBatch[j];
			int[] historyArr = s.generated.stream().mapToInt(Integer::intValue).toArray();
			int nextToken = sampler.sample(logits, s.request.samplingParams(), historyArr);

			if (nextToken == tokenizer.eosTokenId()) {
				s.eosFilter.discardHeld();
				s.reason = GenerationResult.StopReason.EOS_TOKEN;
				justFinished.add(s);
			} else if (sampler.isStopToken(nextToken, s.request.samplingParams())) {
				s.eosFilter.discardHeld();
				s.reason = GenerationResult.StopReason.STOP_TOKEN;
				justFinished.add(s);
			} else {
				String piece = s.stream.append(nextToken);
				EosOutputFilter.Outcome outcome = s.eosFilter.accept(piece);
				if (!outcome.emit().isEmpty()) {
					s.consumer.onToken(outcome.emit(), nextToken, s.generated.size());
					TokenProducedEvent tpe = new TokenProducedEvent();
					tpe.requestId = s.kvKey;
					tpe.position = s.generated.size();
					tpe.commit();
				}
				if (outcome.stop()) {
					s.reason = GenerationResult.StopReason.EOS_TOKEN;
					justFinished.add(s);
				} else {
					s.generated.add(nextToken);
					s.allTokens = GenerationLoop.appendToken(s.allTokens, nextToken);
					if (s.generated.size() >= s.request.samplingParams().maxTokens()) {
						s.reason = GenerationResult.StopReason.MAX_TOKENS;
						justFinished.add(s);
					}
				}
			}
		}
		return justFinished;
	}

	private void retireFinished(List<Slot> finished) {
		if (finished.isEmpty())
			return;
		KVCacheManager kvCache = loop.kvCache();
		InferencePipeline pipeline = loop.pipeline();
		for (Slot s : finished) {
			running.remove(s);
			completeSlot(s, kvCache, pipeline);
		}
	}

	private void completeSlot(Slot s, KVCacheManager kvCache, InferencePipeline pipeline) {
		try {
			EosOutputFilter.Outcome flushed = s.eosFilter.finish(s.stream.flush());
			if (!flushed.emit().isEmpty())
				s.consumer.onToken(flushed.emit(), -1, s.generated.size());
			if (flushed.stop())
				s.reason = GenerationResult.StopReason.EOS_TOKEN;

			if (s.promptLen > 0 && !s.hadCacheHit)
				kvCache.cachePrefix(s.allTokens, s.promptLen, s.kvKey + ":prefix");

			if (s.hasSession) {
				kvCache.cachePrefix(java.util.Arrays.copyOf(s.allTokens, s.promptLen), s.promptLen, s.kvKey);
			} else {
				kvCache.evict(s.kvKey);
				pipeline.evict(s.kvKey);
			}

			s.future.complete(new GenerationResult(s.kvKey, s.eosFilter.text(), s.generated, s.promptLen,
					s.generated.size(), s.reason, Instant.now(), Duration.between(s.start, Instant.now())));
		} catch (Exception e) {
			s.future.completeExceptionally(e);
		}
	}

	private void failAllRunning(Exception e) {
		List<Slot> copy = new ArrayList<>(running);
		running.clear();
		for (Slot s : copy)
			s.future.completeExceptionally(e);
	}

	private record Pending(InferenceRequest request, TokenConsumer consumer,
			CompletableFuture<GenerationResult> future) implements Comparable<Pending> {
		@Override
		public int compareTo(Pending other) {
			return request.compareTo(other.request);
		}
	}

	private enum Phase {
		PREFILL, DECODE
	}

	private static final class Slot {
		final InferenceRequest request;
		final TokenConsumer consumer;
		final CompletableFuture<GenerationResult> future;
		final Instant start;
		final String kvKey;
		final boolean hasSession;
		int[] allTokens;
		final int promptLen;
		final int decodeBase;
		final boolean hadCacheHit;
		final ContinuousPrefillState prefill;
		final List<Integer> generated = new ArrayList<>();
		final EosOutputFilter eosFilter = new EosOutputFilter();
		Tokenizer.StreamContext stream;
		GenerationResult.StopReason reason = GenerationResult.StopReason.MAX_TOKENS;
		Phase phase;

		Slot(InferenceRequest request, TokenConsumer consumer, CompletableFuture<GenerationResult> future,
				Instant start, String kvKey, boolean hasSession, int[] allTokens, int promptLen, int decodeBase,
				boolean hadCacheHit, ContinuousPrefillState prefill) {
			this.request = request;
			this.consumer = consumer;
			this.future = future;
			this.start = start;
			this.kvKey = kvKey;
			this.hasSession = hasSession;
			this.allTokens = allTokens;
			this.promptLen = promptLen;
			this.decodeBase = decodeBase;
			this.hadCacheHit = hadCacheHit;
			this.prefill = prefill;
			this.phase = prefill.isComplete() ? Phase.DECODE : Phase.PREFILL;
		}

		boolean isDecode() {
			return phase == Phase.DECODE;
		}
	}
}
