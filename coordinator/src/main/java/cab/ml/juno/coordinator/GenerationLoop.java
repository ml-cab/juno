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
import java.util.Random;
import java.util.logging.Logger;

import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.node.InferencePipeline;
import cab.ml.juno.sampler.GrammarSession;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatTemplateFormatter;
import cab.ml.juno.tokenizer.Tokenizer;

/**
 * Core autoregressive generation loop.
 *
 * Implements the 8-step loop from the architecture doc: 1. encode prompt
 * (chatTemplate + tokenizer) 2. check prefix cache 3. forward pass (full
 * prefill or incremental from cache hit) 4. sample next token 5. check EOS /
 * stop tokens 6. decode token piece 7. stream piece to client via TokenConsumer
 * 8. repeat
 *
 * Stateless — one shared instance, called per request on a Virtual Thread. Each
 * call is independent; all state lives on the stack.
 */
public final class GenerationLoop {

	private static final Logger log = Logger.getLogger(GenerationLoop.class.getName());

	private final Tokenizer tokenizer;
	private final Sampler sampler;
	private final InferencePipeline pipeline;
	private final KVCacheManager kvCache;
	private final PrefillMode prefillMode;
	private final int prefillBatchSize;

	/**
	 * Construct a generation loop with the default prefill mode ({@link PrefillMode#BATCHED})
	 * and default chunk size ({@link PrefillBatchOptions#DEFAULT_CHUNK_SIZE}).
	 */
	public GenerationLoop(Tokenizer tokenizer, Sampler sampler, InferencePipeline pipeline, KVCacheManager kvCache) {
		this(tokenizer, sampler, pipeline, kvCache, PrefillMode.BATCHED, PrefillBatchOptions.DEFAULT_CHUNK_SIZE);
	}

	/**
	 * Construct a generation loop with an explicit prefill mode and default chunk size.
	 *
	 * @param prefillMode {@link PrefillMode#BATCHED} (default) for windowed GEMM
	 *                    prefill; {@link PrefillMode#SINGLE} for the original
	 *                    sequential one-token loop (escape hatch / bisection).
	 */
	public GenerationLoop(Tokenizer tokenizer, Sampler sampler, InferencePipeline pipeline, KVCacheManager kvCache,
			PrefillMode prefillMode) {
		this(tokenizer, sampler, pipeline, kvCache, prefillMode, PrefillBatchOptions.DEFAULT_CHUNK_SIZE);
	}

	/**
	 * Full constructor with prefill mode and microbatch chunk size.
	 *
	 * @param prefillBatchSize max tokens per {@code prefillBatch} call when mode is
	 *                         {@link PrefillMode#BATCHED}; ignored for {@link PrefillMode#SINGLE}
	 */
	public GenerationLoop(Tokenizer tokenizer, Sampler sampler, InferencePipeline pipeline, KVCacheManager kvCache,
			PrefillMode prefillMode, int prefillBatchSize) {
		this.tokenizer = tokenizer;
		this.sampler = sampler;
		this.pipeline = pipeline;
		this.kvCache = kvCache;
		this.prefillMode = prefillMode;
		this.prefillBatchSize = prefillBatchSize;
	}

	Tokenizer tokenizer() {
		return tokenizer;
	}

	Sampler sampler() {
		return sampler;
	}

	InferencePipeline pipeline() {
		return pipeline;
	}

	KVCacheManager kvCache() {
		return kvCache;
	}

	PrefillMode prefillMode() {
		return prefillMode;
	}

	int prefillBatchSize() {
		return prefillBatchSize;
	}

	/**
	 * Run batched generation for N requests simultaneously.
	 *
	 * One forwardBatch() call per decode step serves all active requests — the GPU
	 * sees a full batch matrix instead of N scalar passes.
	 *
	 * Algorithm (static batching): 1. Encode all prompts and resolve prefix-cache
	 * startPos per request. 2. Each step: collect still-active requests, call
	 * forwardBatch() once, sample independently per request, stream tokens, mark
	 * finished. 3. Loop until every request has hit EOS or its own maxTokens.
	 *
	 * Requests finish independently — a short maxTokens request exits early without
	 * stalling others. Streaming consumers receive tokens in real time, step by
	 * step, exactly as in single-request generation.
	 *
	 * @param entries one entry per request (request + consumer pair)
	 * @return one GenerationResult per entry, in the same order
	 */
	@SuppressWarnings("unchecked")
	public List<GenerationResult> generateBatch(List<BatchEntry> entries) {
		if (entries.isEmpty())
			return List.of();
		if (entries.size() == 1) {
			// Fast path — skip batch overhead for a single entry
			BatchEntry e = entries.get(0);
			return List.of(generate(e.request(), e.consumer()));
		}

		int n = entries.size();

		// ── Per-request state ─────────────────────────────────────────────────
		String[] requestIds = new String[n];
		int[][] allTokens = new int[n][];
		int[] promptLens = new int[n]; // length of original prompt (before generation)
		int[] startPos = new int[n]; // KV cache offset per request
		int[] maxTokens = new int[n];
		List<Integer>[] generated = new List[n];
		EosOutputFilter[] eosFilters = new EosOutputFilter[n];
		StopSequenceFilter[] stopFilters = new StopSequenceFilter[n];
		SamplingParams[] params = new SamplingParams[n];
		Random[] rngs = new Random[n];
		GrammarSession[] grammars = new GrammarSession[n];
		GenerationResult.StopReason[] reasons = new GenerationResult.StopReason[n];
		boolean[] active = new boolean[n];
		Instant[] starts = new Instant[n];

		// ── Step 1: encode all prompts ────────────────────────────────────────
		for (int i = 0; i < n; i++) {
			InferenceRequest req = entries.get(i).request();
			requestIds[i] = req.requestId();
			starts[i] = Instant.now();

			// modelId is set by the caller (e.g. ConsoleMain via ChatModelType.fromPath)
			// to the canonical type key ("phi3", "tinyllama", "llama3", etc.).
			// ChatTemplateFormatter.forModelType() handles the full lookup including phi3;
			// the previous inline ternary chain omitted phi3 → fell through to ChatML →
			// model saw foreign tokens and generated garbage.
			ChatTemplateFormatter formatter = ChatTemplateFormatter.forModelType(req.modelId());
			String prompt = formatter.format(req.messages());
			int[] promptIds = tokenizer.encode(prompt);

			var prefixMatch = kvCache.findLongestPrefix(promptIds);
			startPos[i] = prefixMatch.isHit() ? prefixMatch.matchedTokens() : 0;

			allTokens[i] = promptIds.clone();
			promptLens[i] = promptIds.length;
			params[i] = resolveSamplingParams(req.samplingParams());
			maxTokens[i] = params[i].maxTokens();
			rngs[i] = params[i].seed() != null ? new Random(params[i].seed()) : null;
			grammars[i] = GrammarBinding.open(tokenizer, params[i], requestIds[i]);
			generated[i] = new ArrayList<>();
			eosFilters[i] = new EosOutputFilter();
			stopFilters[i] = new StopSequenceFilter(params[i].stopStrings());
			reasons[i] = GenerationResult.StopReason.MAX_TOKENS;
			active[i] = true;
		}

		// ── Step 1b: Prefill — populate KV cache for all uncached prompt tokens ─
		// Each request gets its own prefill: positions startPos[i]..promptLen[i]-2
		// so the KV cache is warm before the decode loop starts.
		boolean[] hadCacheHit = new boolean[n]; // remember original hit status for later
		for (int i = 0; i < n; i++) {
			hadCacheHit[i] = (startPos[i] > 0);
			int[] promptIds = Arrays.copyOfRange(allTokens[i], 0, promptLens[i]);
			int windowSize = promptLens[i] - 1 - startPos[i];
			if (windowSize > 0) {
				PrefillChunker.run(pipeline, prefillMode, prefillBatchSize, requestIds[i], promptIds, startPos[i]);
			}
			// Decode step 0 covers position promptLen-1 (last prompt token)
			if (promptLens[i] > 0) {
				startPos[i] = promptLens[i] - 1;
			}
		}

		// ── Steps 2–N: batched decode loop ────────────────────────────────────
		Tokenizer.StreamContext[] streams = new Tokenizer.StreamContext[n];
		for (int i = 0; i < n; i++)
			streams[i] = tokenizer.openStreamContext();

		int globalMaxTokens = 0;
		for (int mt : maxTokens)
			globalMaxTokens = Math.max(globalMaxTokens, mt);

		for (int step = 0; step < globalMaxTokens; step++) {

			// Collect active requests for this step
			List<String> batchIds = new ArrayList<>(n);
			List<int[]> batchToks = new ArrayList<>(n);
			List<Integer> batchPos = new ArrayList<>(n);
			List<Integer> batchIdx = new ArrayList<>(n); // original index

			for (int i = 0; i < n; i++) {
				if (!active[i])
					continue;
				if (generated[i].size() >= maxTokens[i]) {
					active[i] = false;
					continue;
				}
				batchIds.add(requestIds[i]);
				batchToks.add(allTokens[i]);
				batchPos.add(startPos[i] + generated[i].size());
				batchIdx.add(i);
			}

			if (batchIds.isEmpty())
				break;

			// One forwardBatch call — the key GPU efficiency gain
			float[][] logitsBatch = pipeline.forwardBatch(batchIds, batchToks, batchPos);

			// Sample + stream for each result independently
			for (int j = 0; j < batchIdx.size(); j++) {
				int i = batchIdx.get(j);
				float[] logits = logitsBatch[j];

				int[] historyArr = generated[i].stream().mapToInt(Integer::intValue).toArray();
				int nextToken = sampler.sample(logits, params[i], historyArr, rngs[i], grammars[i]);

				if (nextToken == tokenizer.eosTokenId()) {
					eosFilters[i].discardHeld();
					stopFilters[i].discardHeld();
					reasons[i] = GenerationResult.StopReason.EOS_TOKEN;
					active[i] = false;
				} else if (sampler.isStopToken(nextToken, params[i])) {
					eosFilters[i].discardHeld();
					stopFilters[i].discardHeld();
					reasons[i] = GenerationResult.StopReason.STOP_TOKEN;
					active[i] = false;
				} else {
					String piece = streams[i].append(nextToken);
					EosOutputFilter.Outcome eosOut = eosFilters[i].accept(piece);
					StopSequenceFilter.Outcome stopOut = stopFilters[i].accept(eosOut.emit());
					if (!stopOut.emit().isEmpty()) {
						entries.get(i).consumer().onToken(stopOut.emit(), nextToken, generated[i].size());
						TokenProducedEvent tpe = new TokenProducedEvent();
						tpe.requestId = requestIds[i];
						tpe.position = generated[i].size();
						tpe.commit();
					}
					if (eosOut.stop()) {
						reasons[i] = GenerationResult.StopReason.EOS_TOKEN;
						active[i] = false;
					} else if (stopOut.stop()) {
						reasons[i] = GenerationResult.StopReason.STOP_TOKEN;
						active[i] = false;
					} else {
						generated[i].add(nextToken);
						allTokens[i] = GenerationLoop.appendToken(allTokens[i], nextToken);
					}
				}
			}
		}

		// ── Build results + cleanup ───────────────────────────────────────────
		List<GenerationResult> results = new ArrayList<>(n);
		for (int i = 0; i < n; i++) {
			EosOutputFilter.Outcome flushedEos = eosFilters[i].finish(streams[i].flush());
			StopSequenceFilter.Outcome flushedStop = stopFilters[i].finish(flushedEos.emit());
			if (!flushedStop.emit().isEmpty()) {
				entries.get(i).consumer().onToken(flushedStop.emit(), -1, generated[i].size());
			}
			if (flushedEos.stop())
				reasons[i] = GenerationResult.StopReason.EOS_TOKEN;
			else if (flushedStop.stop())
				reasons[i] = GenerationResult.StopReason.STOP_TOKEN;

			// Cache prompt prefix for future requests
			if (!hadCacheHit[i] && promptLens[i] > 0) {
				int[] promptOnly = new int[promptLens[i]];
				System.arraycopy(allTokens[i], 0, promptOnly, 0, promptLens[i]);
				kvCache.cachePrefix(promptOnly, promptOnly.length, requestIds[i] + ":prefix");
			}
			kvCache.evict(requestIds[i]);
			pipeline.evict(requestIds[i]);

			results.add(new GenerationResult(requestIds[i], stopFilters[i].text(), generated[i], promptLens[i],
					generated[i].size(), reasons[i], Instant.now(), Duration.between(starts[i], Instant.now())));
		}
		return results;
	}

	/**
	 * Run generation for a single request.
	 *
	 * <h3>Session-aware KV cache</h3> When {@code request.sessionId()} is present
	 * the loop uses the sessionId as the KV key for both the underlying pipeline
	 * and the prefix trie. This means:
	 * <ul>
	 * <li>On turn 1 the full prompt is prefilled and the resulting KV blocks are
	 * stored under the sessionId.</li>
	 * <li>On turn N {@link KVCacheManager#findLongestPrefix} returns the token
	 * count already processed in earlier turns. Prefill starts from that offset, so
	 * no token is ever processed twice.</li>
	 * <li>KV blocks are NOT evicted at the end of each turn — they survive until
	 * the caller explicitly calls {@link #evictSession(String)}.</li>
	 * </ul>
	 *
	 * Stateless requests (no sessionId) behave exactly as before: always prefill
	 * from position 0 and evict immediately on completion.
	 *
	 * @param request  the inference request
	 * @param consumer receives each token piece as it is generated
	 * @return final GenerationResult with full text + stats
	 */
	public GenerationResult generate(InferenceRequest request, TokenConsumer consumer) {
		Instant start = Instant.now();

		// ── Step 1: Encode prompt ─────────────────────────────────────────────
		// modelId is the canonical type key set by the caller ("phi3", "tinyllama", …).
		ChatTemplateFormatter formatter = ChatTemplateFormatter.forModelType(request.modelId());
		String prompt = formatter.format(request.messages());
		int[] promptIds = tokenizer.encode(prompt);

		// ── Step 2: Determine prefill start position ──────────────────────────
		// For session requests: consult the prefix cache. The session key is stable
		// across turns, so a hit means those tokens were already processed and their
		// KV blocks still live in the pipeline under the session key. Start the
		// prefill from the matched offset to skip all previously-seen tokens.
		//
		// For stateless requests: always start at 0. There is no stable key so no
		// cache entry was ever written, and the pipeline has no KV blocks to reuse.
		final String kvKey = request.kvCacheKey();
		final boolean hasSession = request.sessionId() != null;

		int startPos = 0;
		if (hasSession) {
			var prefixMatch = kvCache.findLongestPrefix(promptIds);
			if (prefixMatch.isHit()) {
				startPos = prefixMatch.matchedTokens();
				log.info("Prefix cache hit: " + startPos + "/" + promptIds.length + " tokens cached (session=" + kvKey
						+ ")");
			}
		}

		// Build working token array (prompt IDs only at first)
		int[] allTokens = promptIds.clone();
		List<Integer> generatedIds = new ArrayList<>();
		EosOutputFilter eosFilter = new EosOutputFilter();
		SamplingParams params = resolveSamplingParams(request.samplingParams());
		StopSequenceFilter stopFilter = new StopSequenceFilter(params.stopStrings());
		Random rng = params.seed() != null ? new Random(params.seed()) : null;
		GrammarSession grammar = GrammarBinding.open(tokenizer, params, kvKey);
		GenerationResult.StopReason stopReason = GenerationResult.StopReason.MAX_TOKENS;

		// ── Step 2b: Prefill — populate KV cache for uncached prompt tokens ──
		// Walk positions startPos..promptLen-2, storing KV at each position under
		// kvKey. The last prompt token (position promptLen-1) is left for step 0 of
		// the decode loop so its logits drive the first sampled token.
		int prefillSteps = promptIds.length - 1 - startPos;
		if (prefillSteps > 0) {
			log.info("Prefill: " + prefillSteps + " steps for prompt of " + promptIds.length + " tokens (kvKey=" + kvKey
					+ "  mode=" + prefillMode + ")");
			consumer.onPrefillStart(promptIds.length);
			long prefillCallStart = System.nanoTime();
			PrefillChunker.run(pipeline, prefillMode, prefillBatchSize, kvKey, promptIds, startPos);
			log.info("Prefill: chunker RETURNED kvKey=" + kvKey + " mode=" + prefillMode + " chunkSize="
					+ prefillBatchSize + " elapsedMs=" + (System.nanoTime() - prefillCallStart) / 1_000_000.0);
			consumer.onPrefillComplete();
			log.info("Prefill complete. Decode starts at position " + (promptIds.length - 1) + " kvKey=" + kvKey);
		}
		// Advance startPos so the decode loop runs at the correct sequence positions:
		// step 0 → position promptLen-1 (last prompt token, yields first-token logits)
		// step 1 → position promptLen (first generated token)
		// ...
		if (promptIds.length > 0) {
			startPos = promptIds.length - 1;
		}

		// ── Steps 3–8: Autoregressive decode loop ─────────────────────────────
		int maxTokens = params.maxTokens();
		Tokenizer.StreamContext stream = tokenizer.openStreamContext();
		log.info("Decode: starting loop kvKey=" + kvKey + " maxTokens=" + maxTokens + " startPos=" + startPos
				+ " grammar=" + (grammar != null));

		for (int step = 0; step < maxTokens; step++) {

			long stepStart = System.nanoTime();
			// Step 3: Forward pass — always under kvKey so the pipeline reuses its
			// internal KV matrices for this session.
			float[] logits = pipeline.forward(kvKey, allTokens, startPos + step);
			double forwardMs = (System.nanoTime() - stepStart) / 1_000_000.0;

			// Step 4: Sample next token
			int[] historyArr = generatedIds.stream().mapToInt(Integer::intValue).toArray();
			int nextToken = sampler.sample(logits, params, historyArr, rng, grammar);

			// Step 5: Check stop conditions by token ID
			if (nextToken == tokenizer.eosTokenId()) {
				eosFilter.discardHeld();
				stopFilter.discardHeld();
				stopReason = GenerationResult.StopReason.EOS_TOKEN;
				log.info("Decode: step " + step + " EOS_TOKEN kvKey=" + kvKey + " forwardMs=" + forwardMs);
				break;
			}
			if (sampler.isStopToken(nextToken, params)) {
				eosFilter.discardHeld();
				stopFilter.discardHeld();
				stopReason = GenerationResult.StopReason.STOP_TOKEN;
				log.info("Decode: step " + step + " STOP_TOKEN kvKey=" + kvKey + " forwardMs=" + forwardMs);
				break;
			}

			// Step 6–7: Decode and stream through EosOutputFilter then stop sequences.
			String piece = stream.append(nextToken);
			EosOutputFilter.Outcome eosOut = eosFilter.accept(piece);
			StopSequenceFilter.Outcome stopOut = stopFilter.accept(eosOut.emit());
			if (!stopOut.emit().isEmpty()) {
				consumer.onToken(stopOut.emit(), nextToken, step);
				TokenProducedEvent tpe = new TokenProducedEvent();
				tpe.requestId = kvKey;
				tpe.position = step;
				tpe.commit();
			}
			if (eosOut.stop()) {
				stopReason = GenerationResult.StopReason.EOS_TOKEN;
				log.info("Decode: step " + step + " EOS_MARKER kvKey=" + kvKey + " forwardMs=" + forwardMs);
				break;
			}
			if (stopOut.stop()) {
				stopReason = GenerationResult.StopReason.STOP_TOKEN;
				log.info("Decode: step " + step + " STOP_SEQUENCE kvKey=" + kvKey + " forwardMs=" + forwardMs);
				break;
			}

			generatedIds.add(nextToken);
			allTokens = GenerationLoop.appendToken(allTokens, nextToken);
		}
		log.info("Decode: loop EXITED kvKey=" + kvKey + " tokensGenerated=" + generatedIds.size() + " stopReason="
				+ stopReason);

		// ── Post-generation: cache or evict ───────────────────────────────────
		if (hasSession) {
			// Cache the current formatted prompt token sequence (NOT the generated
			// tokens). The next turn's formatted prompt begins with ALL of the current
			// turn's prompt tokens (the conversation grows monotonically), so
			// findLongestPrefix on turn N+1 will match exactly promptIds.length tokens
			// and skip re-processing them.
			//
			// Caching allTokens would be wrong: generated token IDs do not appear in
			// the next turn's formatted prompt (the assistant text is re-encoded from
			// its string representation at turn N+1, which may produce different IDs due
			// to SimpleTokenizer round-trip behaviour and special-token boundaries). The
			// trie leaf would be unreachable because the paths diverge before reaching
			// it, and findLongestPrefix would return no hit.
			kvCache.cachePrefix(promptIds, promptIds.length, kvKey);
			// Do NOT evict — the pipeline's KV blocks under sessionId must survive
			// until the session ends. Caller is responsible for calling evictSession().
		} else {
			// Stateless request — clean up the pipeline KV immediately.
			// No cachePrefix call: there is no stable key for a future request to match.
			// kvCache.evict() only clears KVCacheManager's cross-node/session-restore
			// tier; pipeline.evict() releases each handler's own in-process working
			// KV arrays for this requestId — without it they leak for the life of
			// the process (see ForwardPassHandler.evict javadoc).
			kvCache.evict(kvKey);
			pipeline.evict(kvKey);
		}

		EosOutputFilter.Outcome flushedEos = eosFilter.finish(stream.flush());
		StopSequenceFilter.Outcome flushedStop = stopFilter.finish(flushedEos.emit());
		if (!flushedStop.emit().isEmpty())
			consumer.onToken(flushedStop.emit(), -1, generatedIds.size());
		if (flushedEos.stop())
			stopReason = GenerationResult.StopReason.EOS_TOKEN;
		else if (flushedStop.stop())
			stopReason = GenerationResult.StopReason.STOP_TOKEN;

		GenerationResult result = new GenerationResult(kvKey, stopFilter.text(), generatedIds, promptIds.length,
				generatedIds.size(), stopReason, Instant.now(), Duration.between(start, Instant.now()));
		log.info("generate() RETURNING kvKey=" + kvKey + " stopReason=" + stopReason + " tokensGenerated="
				+ generatedIds.size() + " totalDurationMs=" + result.latency().toMillis() + " textLength="
				+ result.text().length());
		return result;
	}

	SamplingParams resolveSamplingParams(SamplingParams params) {
		int[] merged = OpenAiAdapter.stopTokenIdsFromStrings(tokenizer, params.stopStrings(), params.stopTokenIds());
		return params.withStopTokenIds(merged);
	}

	/** Collapses newlines/control chars so one log line per decode step stays one line. */
	private static String escapeForLog(String s) {
		return s.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t");
	}

	/**
	 * Release all KV resources held for a conversation session.
	 *
	 * Evicts KV blocks from both GPU and CPU cache tiers and removes the
	 * prefix-trie entry so a later session that begins with the same tokens does
	 * not get a stale hit pointing at freed KV blocks.
	 *
	 * Call this when the user ends a multi-turn session — e.g. when the REPL
	 * receives "exit", or when a REST session times out.
	 *
	 * @param sessionId the sessionId that was passed to
	 *                  {@link InferenceRequest#ofSession}
	 */
	public void evictSession(String sessionId) {
		kvCache.evict(sessionId);
		kvCache.invalidatePrefix(sessionId);
		pipeline.evict(sessionId);
	}

	// ── Helpers ───────────────────────────────────────────────────────────────

	static int[] appendToken(int[] tokens, int newToken) {
		int[] next = new int[tokens.length + 1];
		System.arraycopy(tokens, 0, next, 0, tokens.length);
		next[tokens.length] = newToken;
		return next;
	}
}