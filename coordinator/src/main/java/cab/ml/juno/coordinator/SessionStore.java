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

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ThreadLocalRandom;
import java.util.regex.Pattern;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import cab.ml.juno.kvcache.CacheTypeOptions;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.kvcache.KvSessionFileException;
import cab.ml.juno.kvcache.KvSessionFileException.Reason;
import cab.ml.juno.kvcache.KvSessionStamp;
import cab.ml.juno.kvcache.SessionDirOptions;
import cab.ml.juno.lora.LoraPlaySpec;
import cab.ml.juno.node.SessionKvPart;

/**
 * Explicit session save, restore and delete ({@code POST /v1/sessions/{id}/save}, {@code .../restore},
 * {@code DELETE /v1/sessions/{id}}): the only code that writes session KV to disk, and only when a client asks.
 *
 * <p>A save writes, under the server's session directory, one file per stage holding that stage's layers
 * ({@link cab.ml.juno.kvcache.KvSessionFile}; on a cluster each node writes under its own directory) and then the
 * manifest {@code <id>.session}: format version, model and adapter identity, cache types, the tokens whose positions
 * the rows hold (the session's prefix entry), and each stage's file and size. A restore checks the manifest against
 * this server before it touches anything, restores every stage, and makes the tokens the session's prefix entry, so the
 * next chat turn, which sends the whole conversation as usual, prefills only its new tokens. All three hold the session
 * ({@link RequestScheduler#withSessionHeld}): a session with a request in flight answers 409.
 */
public final class SessionStore {

	public static final int MANIFEST_VERSION = 1;
	public static final int MAX_ID_LENGTH = 128;

	private static final Pattern ID = Pattern.compile("[A-Za-z0-9._-]+");
	private static final String ALL_STAGES = "all";
	private static final ObjectMapper JSON = new ObjectMapper();
	private static final int IDENTITY_HEAD_BYTES = 1 << 20;

	/**
	 * What a saved session belongs to: the model file (name, size and a SHA-256 of its first MiB, which holds its
	 * header and metadata) and the LoRA adapters played over it (each file the same way, with its scale; empty
	 * without).
	 */
	public record Identity(String model, String adapter) {
		public Identity {
			if (model == null || model.isBlank())
				throw new IllegalArgumentException("model identity must not be blank");
			if (adapter == null)
				adapter = "";
		}

		/** Identity of {@code modelFile} with the adapters of a {@code --lora-play} specification (null for none). */
		public static Identity of(Path modelFile, String loraPlaySpec) throws IOException {
			StringBuilder adapter = new StringBuilder();
			if (loraPlaySpec != null && !loraPlaySpec.isBlank())
				for (LoraPlaySpec.Entry e : LoraPlaySpec.parse(loraPlaySpec)) {
					if (adapter.length() > 0)
						adapter.append("; ");
					adapter.append(fileIdentity(e.path())).append(" at scale ").append(e.scale());
				}
			return new Identity(fileIdentity(modelFile), adapter.toString());
		}

		static String fileIdentity(Path file) throws IOException {
			long size = Files.size(file);
			MessageDigest sha;
			try {
				sha = MessageDigest.getInstance("SHA-256");
			} catch (NoSuchAlgorithmException e) {
				throw new IllegalStateException(e);
			}
			try (InputStream in = Files.newInputStream(file)) {
				sha.update(in.readNBytes((int) Math.min(size, IDENTITY_HEAD_BYTES)));
			}
			return file.getFileName() + ", " + size + " bytes, first MiB sha256 " + HexFormat.of().formatHex(sha.digest());
		}
	}

	/** What a call answers: the session, its token count, the bytes written or read, and the time taken. */
	public record Outcome(String sessionId, int tokens, long bytes, long elapsedMs) {
	}

	private final RequestScheduler scheduler;
	private final Path dir;
	private final Identity identity;

	/**
	 * @param dir      the session directory, or null: save and restore are then refused, delete still works
	 * @param identity what saved sessions belong to; required with a directory
	 */
	public SessionStore(RequestScheduler scheduler, Path dir, Identity identity) {
		if (scheduler == null)
			throw new IllegalArgumentException("scheduler must not be null");
		if (dir != null && identity == null)
			throw new IllegalArgumentException("a session directory needs the model's identity");
		this.scheduler = scheduler;
		this.dir = dir;
		this.identity = identity;
	}

	/** A store with no directory: save and restore answer that {@code --session-dir} is needed. */
	public static SessionStore withoutDirectory(RequestScheduler scheduler) {
		return new SessionStore(scheduler, null, null);
	}

	/**
	 * The store for a server started with {@link SessionDirOptions}' directory, if one is set, for {@code modelFile}
	 * and the adapters of {@code loraPlaySpec} (null for none). A configured path that exists and is not a directory
	 * is refused at startup.
	 */
	public static SessionStore configure(RequestScheduler scheduler, Path modelFile, String loraPlaySpec)
			throws IOException {
		Optional<Path> d = SessionDirOptions.fromEnv();
		if (d.isEmpty())
			return withoutDirectory(scheduler);
		if (Files.exists(d.get()) && !Files.isDirectory(d.get()))
			throw new IllegalArgumentException("--session-dir " + d.get() + " is not a directory");
		return new SessionStore(scheduler, d.get(), Identity.of(modelFile, loraPlaySpec));
	}

	public Optional<Path> directory() {
		return Optional.ofNullable(dir);
	}

	/** @throws IllegalArgumentException unless {@code id} is 1 to 128 letters, digits, '-', '_' or '.' */
	public static void requireValidId(String id) {
		if (id == null || id.isEmpty() || id.length() > MAX_ID_LENGTH || !ID.matcher(id).matches())
			throw new IllegalArgumentException("a session id is 1 to " + MAX_ID_LENGTH
					+ " letters, digits, '-', '_' or '.'");
	}

	/** Saves the session's KV: positions {@code [0, n)} of its prefix entry's {@code n} tokens, on every stage. */
	public Outcome save(String sessionId) throws IOException {
		requireValidId(sessionId);
		Path d = requireDir();
		GenerationLoop loop = scheduler.generationLoop();
		return scheduler.withSessionHeld(sessionId, () -> {
			long t0 = System.nanoTime();
			int[] tokens = loop.kvCache().prefixEntry(sessionId).orElseThrow(() -> new KvSessionFileException(
					Reason.NOT_FOUND, "session " + sessionId + " holds no KV to save"));
			Files.createDirectories(d);
			Path manifest = d.resolve(manifestName(sessionId));
			Files.deleteIfExists(manifest); // a save that fails part way leaves nothing to restore
			KvSessionStamp stamp = new KvSessionStamp(identity.model(), identity.adapter(), ALL_STAGES, tokens);
			List<SessionKvPart> parts = loop.pipeline().saveKv(sessionId, stamp, d);
			long manifestBytes = writeManifest(manifest, sessionId, tokens, parts);
			return new Outcome(sessionId, tokens.length, sum(parts) + manifestBytes, millisSince(t0));
		});
	}

	/**
	 * Restores a saved session on every stage and registers its tokens as the session's prefix entry. A manifest of
	 * another model, adapter, cache type or format is refused before anything changes; a stage that refuses leaves the
	 * session with no KV.
	 */
	public Outcome restore(String sessionId) throws IOException {
		requireValidId(sessionId);
		Path d = requireDir();
		GenerationLoop loop = scheduler.generationLoop();
		return scheduler.withSessionHeld(sessionId, () -> {
			long t0 = System.nanoTime();
			Path manifestFile = d.resolve(manifestName(sessionId));
			Manifest m = readManifest(manifestFile);
			CacheTypeOptions types = CacheTypeOptions.fromEnv();
			expect(sessionId, "model", m.model(), identity.model());
			expect(sessionId, "adapter", m.adapter(), identity.adapter());
			expect(sessionId, "K cache type", m.typeK(), types.typeK().cliName());
			expect(sessionId, "V cache type", m.typeV(), types.typeV().cliName());
			KvSessionStamp stamp = new KvSessionStamp(m.model(), m.adapter(), ALL_STAGES, m.tokens());
			KVCacheManager kv = loop.kvCache();
			kv.evict(sessionId);
			kv.invalidatePrefix(sessionId);
			List<SessionKvPart> parts;
			try {
				parts = loop.pipeline().restoreKv(sessionId, stamp, d);
			} catch (KvSessionFileException e) {
				if (e.reason() != Reason.NOT_FOUND)
					throw e;
				throw new KvSessionFileException(Reason.MISMATCH, "session " + sessionId + " was saved on stages "
						+ m.stages() + ", and a stage here found no file of it: " + e.getMessage()
						+ " (another shard plan, or a node whose session directory does not hold the session)");
			}
			kv.cachePrefix(m.tokens(), m.tokens().length, sessionId);
			return new Outcome(sessionId, m.tokens().length, sum(parts) + Files.size(manifestFile), millisSince(t0));
		});
	}

	/** Frees the session's KV in memory on every stage; its saved files stay. */
	public Outcome delete(String sessionId) throws IOException {
		requireValidId(sessionId);
		GenerationLoop loop = scheduler.generationLoop();
		return scheduler.withSessionHeld(sessionId, () -> {
			long t0 = System.nanoTime();
			int tokens = loop.kvCache().prefixEntry(sessionId).map(t -> t.length).orElse(0);
			loop.evictSession(sessionId);
			return new Outcome(sessionId, tokens, 0, millisSince(t0));
		});
	}

	static String manifestName(String sessionId) {
		return sessionId + ".session";
	}

	private Path requireDir() {
		if (dir == null)
			throw new KvSessionFileException(Reason.UNSUPPORTED, "saving and restoring sessions needs a session"
					+ " directory: start the server with --session-dir PATH (or " + SessionDirOptions.ENV + ")");
		return dir;
	}

	private record Manifest(String model, String adapter, String typeK, String typeV, int[] tokens,
			List<String> stages) {
	}

	private long writeManifest(Path file, String sessionId, int[] tokens, List<SessionKvPart> parts)
			throws IOException {
		CacheTypeOptions types = CacheTypeOptions.fromEnv();
		ObjectNode root = JSON.createObjectNode();
		root.put("format", MANIFEST_VERSION);
		root.put("session_id", sessionId);
		root.put("model", identity.model());
		root.put("adapter", identity.adapter());
		root.put("cache_type_k", types.typeK().cliName());
		root.put("cache_type_v", types.typeV().cliName());
		root.put("saved_at", Instant.now().toString());
		ArrayNode t = root.putArray("tokens");
		for (int token : tokens)
			t.add(token);
		ArrayNode stages = root.putArray("stages");
		for (SessionKvPart p : parts)
			stages.addObject().put("stage", p.stage()).put("file", p.file()).put("bytes", p.bytes());
		Path tmp = file.resolveSibling(file.getFileName() + ".tmp-" + Long.toHexString(ThreadLocalRandom.current().nextLong()));
		try {
			Files.write(tmp, JSON.writerWithDefaultPrettyPrinter().writeValueAsBytes(root));
			try {
				Files.move(tmp, file, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
			} catch (AtomicMoveNotSupportedException e) {
				Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING);
			}
		} finally {
			Files.deleteIfExists(tmp);
		}
		return Files.size(file);
	}

	private static Manifest readManifest(Path file) throws IOException {
		JsonNode root;
		try {
			root = JSON.readTree(Files.readAllBytes(file));
		} catch (NoSuchFileException e) {
			throw new KvSessionFileException(Reason.NOT_FOUND, "no saved session " + file.getFileName());
		} catch (JsonProcessingException e) {
			throw corrupt(file, "not a session manifest");
		}
		if (root == null || !root.isObject() || !root.path("format").isInt())
			throw corrupt(file, "not a session manifest");
		int format = root.get("format").asInt();
		if (format != MANIFEST_VERSION)
			throw new KvSessionFileException(Reason.MISMATCH, file.getFileName() + " is format " + format
					+ "; this build reads format " + MANIFEST_VERSION);
		JsonNode tokens = root.path("tokens");
		JsonNode stages = root.path("stages");
		for (String field : new String[] { "model", "adapter", "cache_type_k", "cache_type_v" })
			if (!root.path(field).isTextual())
				throw corrupt(file, "no " + field);
		if (!tokens.isArray() || tokens.isEmpty() || !stages.isArray())
			throw corrupt(file, "no tokens or stages");
		int[] t = new int[tokens.size()];
		for (int i = 0; i < t.length; i++) {
			if (!tokens.get(i).isInt())
				throw corrupt(file, "token " + i + " is not an integer");
			t[i] = tokens.get(i).asInt();
		}
		List<String> labels = new ArrayList<>();
		for (JsonNode s : stages)
			labels.add(s.path("stage").asText());
		return new Manifest(root.get("model").asText(), root.get("adapter").asText(), root.get("cache_type_k").asText(),
				root.get("cache_type_v").asText(), t, labels);
	}

	private static void expect(String sessionId, String field, String saved, String here) {
		if (!saved.equals(here))
			throw new KvSessionFileException(Reason.MISMATCH, "session " + sessionId + " was saved with " + field
					+ " \"" + saved + "\"; this server has \"" + here + "\"");
	}

	private static KvSessionFileException corrupt(Path file, String what) {
		return new KvSessionFileException(Reason.CORRUPT, file.getFileName() + " is damaged: " + what);
	}

	private static long sum(List<SessionKvPart> parts) {
		long s = 0;
		for (SessionKvPart p : parts)
			s += p.bytes();
		return s;
	}

	private static long millisSince(long t0) {
		return (System.nanoTime() - t0) / 1_000_000;
	}
}
