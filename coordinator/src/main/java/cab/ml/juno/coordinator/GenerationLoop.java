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
import java.util.logging.Logger;

import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.node.InferencePipeline;
import cab.ml.juno.sampler.Sampler;
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

	public GenerationLoop(Tokenizer tokenizer, Sampler sampler, InferencePipeline pipeline, KVCacheManager kvCache) {
		this.tokenizer = tokenizer;
		this.sampler = sampler;
		this.pipeline = pipeline;
		this.kvCache = kvCache;
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
			maxTokens[i] = req.samplingParams().maxTokens();
			generated[i] = new ArrayList<>();
			eosFilters[i] = new EosOutputFilter();
			reasons[i] = GenerationResult.StopReason.MAX_TOKENS;
			active[i] = true;
		}

		// ── Step 1b: Prefill — populate KV cache for all uncached prompt tokens ─
		// Each request gets its own prefill: walk positions startPos[i]..promptLen[i]-2
		// so the KV cache is warm before the decode loop starts.
		boolean[] hadCacheHit = new boolean[n]; // remember original hit status for later
		for (int i = 0; i < n; i++) {
			hadCacheHit[i] = (startPos[i] > 0);
			int[] promptIds = Arrays.copyOfRange(allTokens[i], 0, promptLens[i]);
			for (int p = startPos[i]; p < promptLens[i] - 1; p++) {
				int[] prefillSlice = Arrays.copyOfRange(promptIds, 0, p + 1);
				pipeline.forward(requestIds[i], prefillSlice, p); // KV stored; logits discarded
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
				InferenceRequest req = entries.get(i).request();
				float[] logits = logitsBatch[j];

				int[] historyArr = generated[i].stream().mapToInt(Integer::intValue).toArray();
				int nextToken = sampler.sample(logits, req.samplingParams(), historyArr);

				if (nextToken == tokenizer.eosTokenId()) {
					eosFilters[i].discardHeld();
					reasons[i] = GenerationResult.StopReason.EOS_TOKEN;
					active[i] = false;
				} else if (sampler.isStopToken(nextToken, req.samplingParams())) {
					eosFilters[i].discardHeld();
					reasons[i] = GenerationResult.StopReason.STOP_TOKEN;
					active[i] = false;
				} else {
					String piece = streams[i].append(nextToken);
					EosOutputFilter.Outcome outcome = eosFilters[i].accept(piece);
					if (!outcome.emit().isEmpty()) {
						entries.get(i).consumer().onToken(outcome.emit(), nextToken, generated[i].size());
						TokenProducedEvent tpe = new TokenProducedEvent();
						tpe.requestId = requestIds[i];
						tpe.position = generated[i].size();
						tpe.commit();
					}
					if (outcome.stop()) {
						reasons[i] = GenerationResult.StopReason.EOS_TOKEN;
						active[i] = false;
					} else {
						generated[i].add(nextToken);
						allTokens[i] = appendToken(allTokens[i], nextToken);
					}
				}
			}
		}

		// ── Build results + cleanup ───────────────────────────────────────────
		List<GenerationResult> results = new ArrayList<>(n);
		for (int i = 0; i < n; i++) {
			EosOutputFilter.Outcome flushed = eosFilters[i].finish(streams[i].flush());
			if (!flushed.emit().isEmpty()) {
				entries.get(i).consumer().onToken(flushed.emit(), -1, generated[i].size());
			}
			if (flushed.stop())
				reasons[i] = GenerationResult.StopReason.EOS_TOKEN;

			// Cache prompt prefix for future requests
			if (!hadCacheHit[i] && promptLens[i] > 0) {
				int[] promptOnly = new int[promptLens[i]];
				System.arraycopy(allTokens[i], 0, promptOnly, 0, promptLens[i]);
				kvCache.cachePrefix(promptOnly, promptOnly.length, requestIds[i] + ":prefix");
			}
			kvCache.evict(requestIds[i]);

			results.add(new GenerationResult(requestIds[i], eosFilters[i].text(), generated[i], promptLens[i],
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
		GenerationResult.StopReason stopReason = GenerationResult.StopReason.MAX_TOKENS;

		// ── Step 2b: Prefill — populate KV cache for uncached prompt tokens ──
		// Walk positions startPos..promptLen-2, storing KV at each position under
		// kvKey. The last prompt token (position promptLen-1) is left for step 0 of
		// the decode loop so its logits drive the first sampled token.
		int prefillSteps = promptIds.length - 1 - startPos;
		if (prefillSteps > 0) {
			log.info("Prefill: " + prefillSteps + " steps for prompt of " + promptIds.length + " tokens (kvKey=" + kvKey
					+ ")");
			consumer.onPrefillStart(promptIds.length);
			for (int p = startPos; p < promptIds.length - 1; p++) {
				int[] prefillSlice = Arrays.copyOfRange(promptIds, 0, p + 1);
				pipeline.forward(kvKey, prefillSlice, p); // KV stored under kvKey; logits discarded
			}
			consumer.onPrefillComplete();
			log.info("Prefill complete. Decode starts at position " + (promptIds.length - 1));
		}
		// Advance startPos so the decode loop runs at the correct sequence positions:
		// step 0 → position promptLen-1 (last prompt token, yields first-token logits)
		// step 1 → position promptLen (first generated token)
		// ...
		if (promptIds.length > 0) {
			startPos = promptIds.length - 1;
		}

		// ── Steps 3–8: Autoregressive decode loop ─────────────────────────────
		int maxTokens = request.samplingParams().maxTokens();
		Tokenizer.StreamContext stream = tokenizer.openStreamContext();

		for (int step = 0; step < maxTokens; step++) {

			// Step 3: Forward pass — always under kvKey so the pipeline reuses its
			// internal KV matrices for this session.
			float[] logits = pipeline.forward(kvKey, allTokens, startPos + step);

			// Step 4: Sample next token
			int[] historyArr = generatedIds.stream().mapToInt(Integer::intValue).toArray();
			int nextToken = sampler.sample(logits, request.samplingParams(), historyArr);

			// Step 5: Check stop conditions by token ID
			if (nextToken == tokenizer.eosTokenId()) {
				eosFilter.discardHeld();
				stopReason = GenerationResult.StopReason.EOS_TOKEN;
				break;
			}
			if (sampler.isStopToken(nextToken, request.samplingParams())) {
				eosFilter.discardHeld();
				stopReason = GenerationResult.StopReason.STOP_TOKEN;
				break;
			}

			// Step 6–7: Decode and stream through EosOutputFilter.
			// Holds back partial turn-end markers (e.g. "</"+"s"+">") and strips
			// complete markers for every supported chat template so /train-qa
			// completions never leak "</s>", "<|end|>", "<|im_end|>", etc.
			String piece = stream.append(nextToken);
			EosOutputFilter.Outcome outcome = eosFilter.accept(piece);
			if (!outcome.emit().isEmpty()) {
				consumer.onToken(outcome.emit(), nextToken, step);
				TokenProducedEvent tpe = new TokenProducedEvent();
				tpe.requestId = kvKey;
				tpe.position = step;
				tpe.commit();
			}
			if (outcome.stop()) {
				stopReason = GenerationResult.StopReason.EOS_TOKEN;
				break;
			}

			generatedIds.add(nextToken);
			allTokens = appendToken(allTokens, nextToken);
		}

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
			kvCache.evict(kvKey);
		}

		EosOutputFilter.Outcome flushed = eosFilter.finish(stream.flush());
		if (!flushed.emit().isEmpty())
			consumer.onToken(flushed.emit(), -1, generatedIds.size());
		if (flushed.stop())
			stopReason = GenerationResult.StopReason.EOS_TOKEN;

		return new GenerationResult(kvKey, eosFilter.text(), generatedIds, promptIds.length, generatedIds.size(),
				stopReason, Instant.now(), Duration.between(start, Instant.now()));
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
	}

	// ── Helpers ───────────────────────────────────────────────────────────────

	private int[] appendToken(int[] tokens, int newToken) {
		int[] next = new int[tokens.length + 1];
		System.arraycopy(tokens, 0, next, 0, tokens.length);
		next[tokens.length] = newToken;
		return next;
	}
}