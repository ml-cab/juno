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
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.logging.Logger;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Resolves a {@code repo[:quant]} spec against the Hugging Face Hub HTTP API
 * and downloads the chosen GGUF asset, with resume support and ETag-based
 * caching — backs the {@code --hf} CLI flag.
 *
 * <p><b>No Python anywhere in this path</b> — this is a plain JDK
 * {@link HttpClient} consumer (and Jackson, already a {@code juno-player}
 * dependency for OpenAI-compatible JSON) talking to the public
 * {@code huggingface.co} REST API; there is no dependency on the Python
 * {@code huggingface_hub} package or the {@code hf} / {@code huggingface-cli}
 * command-line tools.
 *
 * <p><b>Cache location:</b> {@code ~/.cache/juno/models/<org>__<repo>/<file>}
 * by default (override via the constructor) — deliberately <i>not</i> the
 * repo's {@code models/} directory, which CLAUDE.md reserves for local test
 * fixtures that must never be written to or committed by tooling. One
 * directory per repo (org and repo name joined with {@code __}) keeps
 * same-named files from different repos from colliding.
 *
 * <p><b>Quant selection policy:</b> an explicit {@code :quant} suffix must
 * name a {@code .gguf} asset containing that string (case-insensitive);
 * otherwise the first asset containing {@value #PREFERRED_QUANT} wins: else
 * the alphabetically-first {@code .gguf} asset in the repo (deterministic,
 * no silent "biggest file" guess). See {@code docs/howto.md} for the
 * user-facing description of this policy.
 */
public final class HfGgufFetcher {

	private static final Logger log = Logger.getLogger(HfGgufFetcher.class.getName());

	/** Default Hugging Face Hub API base — overridable for tests against a local mock server. */
	public static final String DEFAULT_API_BASE = "https://huggingface.co";

	/** Preferred quantization when the caller does not name one explicitly. */
	public static final String PREFERRED_QUANT = "Q4_K_M";

	private final HttpClient http;
	private final String apiBase;
	private final Path cacheDir;
	private final ObjectMapper json = new ObjectMapper();

	public HfGgufFetcher() {
		// The Hub's file-download endpoint (/resolve/main/...) 302s the actual bytes
		// off to its CDN — JDK's HttpClient.newHttpClient() defaults to
		// Redirect.NEVER, which would otherwise surface that redirect as a download
		// failure instead of transparently following it.
		this(HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build(), DEFAULT_API_BASE,
				defaultCacheDir());
	}

	public HfGgufFetcher(HttpClient http, String apiBase, Path cacheDir) {
		this.http = http;
		this.apiBase = apiBase.endsWith("/") ? apiBase.substring(0, apiBase.length() - 1) : apiBase;
		this.cacheDir = cacheDir;
	}

	/** {@code ~/.cache/juno/models} — see the class doc for why this is not {@code models/}. */
	public static Path defaultCacheDir() {
		String home = System.getProperty("user.home", ".");
		return Path.of(home, ".cache", "juno", "models");
	}

	// ── Spec parsing ─────────────────────────────────────────────────────────

	/** A parsed {@code org/repo[:quant]} spec. */
	public record Spec(String repo, String quant) {

		public static Spec parse(String s) {
			if (s == null || s.isBlank())
				throw new IllegalArgumentException("--hf spec must not be blank");
			String trimmed = s.strip();
			int colon = trimmed.lastIndexOf(':');
			// A colon inside "org/repo" alone (no quant) is not a real-world HF repo id
			// shape, so a lastIndexOf split is unambiguous here.
			if (colon < 0)
				return new Spec(trimmed, null);
			String repo = trimmed.substring(0, colon).strip();
			String quant = trimmed.substring(colon + 1).strip();
			if (repo.isEmpty())
				throw new IllegalArgumentException("--hf spec has an empty repo: " + s);
			return new Spec(repo, quant.isEmpty() ? null : quant);
		}
	}

	/** One candidate GGUF file in a Hugging Face repo. */
	public record Asset(String filename) {
	}

	// ── Hub API ──────────────────────────────────────────────────────────────

	/** List every {@code .gguf} sibling file of {@code repo} via the Hub models API. */
	public List<Asset> listGgufAssets(String repo) throws IOException, InterruptedException {
		String url = apiBase + "/api/models/" + repo;
		HttpRequest req = HttpRequest.newBuilder(URI.create(url)).GET().build();
		HttpResponse<String> res = http.send(req, HttpResponse.BodyHandlers.ofString());
		if (res.statusCode() == 401 || res.statusCode() == 404)
			// The Hub API deliberately returns 401 (not 404) for both a nonexistent repo
			// and a private/gated one it won't disclose to an anonymous caller — so this
			// status alone can't tell them apart. Most real-world hits are a typo'd or
			// renamed repo id, so lead with that rather than a bare status code.
			throw new IOException("Hugging Face repo '" + repo + "' was not found (HTTP " + res.statusCode()
					+ "). Double-check the repo id (org/name) — it may be misspelled, renamed, or a private/gated "
					+ "repo that needs authentication, which --hf does not support yet.");
		if (res.statusCode() != 200)
			throw new IOException("Hugging Face Hub API returned HTTP " + res.statusCode() + " for " + url);
		JsonNode root = json.readTree(res.body());
		JsonNode siblings = root.path("siblings");
		List<Asset> assets = new ArrayList<>();
		if (siblings.isArray()) {
			for (JsonNode s : siblings) {
				String name = s.path("rfilename").asText("");
				if (name.toLowerCase(Locale.ROOT).endsWith(".gguf"))
					assets.add(new Asset(name));
			}
		}
		if (assets.isEmpty())
			throw new IOException("No .gguf assets found in Hugging Face repo: " + repo);
		return assets;
	}

	/**
	 * Choose the asset matching {@code requestedQuant} (case-insensitive
	 * substring), or {@value #PREFERRED_QUANT} when {@code requestedQuant} is
	 * {@code null}, or the alphabetically-first {@code .gguf} asset otherwise.
	 */
	public Asset selectAsset(List<Asset> assets, String requestedQuant) {
		if (assets == null || assets.isEmpty())
			throw new IllegalArgumentException("no GGUF assets to select from");
		if (requestedQuant != null) {
			String needle = requestedQuant.toUpperCase(Locale.ROOT);
			return assets.stream().filter(a -> a.filename().toUpperCase(Locale.ROOT).contains(needle)).findFirst()
					.orElseThrow(() -> new IllegalArgumentException("Requested quant '" + requestedQuant
							+ "' not found. Available: " + assets.stream().map(Asset::filename).sorted().toList()));
		}
		return assets.stream().filter(a -> a.filename().toUpperCase(Locale.ROOT).contains(PREFERRED_QUANT))
				.findFirst()
				.orElseGet(() -> assets.stream().sorted(java.util.Comparator.comparing(Asset::filename)).findFirst()
						.orElseThrow());
	}

	/**
	 * Resolve {@code spec} ({@code org/repo[:quant]}) to a local GGUF path,
	 * downloading (with resume + ETag caching) if not already cached.
	 */
	public Path resolve(String spec) throws IOException, InterruptedException {
		Spec sp = Spec.parse(spec);
		List<Asset> assets = listGgufAssets(sp.repo());
		Asset chosen = selectAsset(assets, sp.quant());
		log.info(() -> "--hf " + spec + " resolved to asset " + chosen.filename() + " in repo " + sp.repo());
		return download(sp.repo(), chosen);
	}

	// ── Download (resume + ETag cache) ──────────────────────────────────────

	Path download(String repo, Asset asset) throws IOException, InterruptedException {
		Path destDir = cacheDir.resolve(sanitizeRepoDirName(repo));
		Files.createDirectories(destDir);
		Path dest = destDir.resolve(asset.filename());
		Path etagFile = destDir.resolve(asset.filename() + ".etag");
		Path partFile = destDir.resolve(asset.filename() + ".part");

		String url = apiBase + "/" + repo + "/resolve/main/" + asset.filename();

		String remoteEtag = headEtag(url);
		if (Files.exists(dest) && remoteEtag != null && remoteEtag.equals(readEtagFile(etagFile))) {
			log.info(() -> "Cache hit (ETag match) for " + dest + " — skipping download");
			return dest;
		}

		long existing = Files.exists(partFile) ? Files.size(partFile) : 0;
		HttpRequest.Builder builder = HttpRequest.newBuilder(URI.create(url));
		if (existing > 0)
			builder.header("Range", "bytes=" + existing + "-");
		HttpResponse<InputStream> res = http.send(builder.GET().build(), HttpResponse.BodyHandlers.ofInputStream());

		boolean resumed = res.statusCode() == 206;
		if (res.statusCode() != 200 && res.statusCode() != 206)
			throw new IOException("Download failed: HTTP " + res.statusCode() + " for " + url);
		if (!resumed)
			existing = 0; // server ignored Range (or nothing to resume) — restart from scratch

		StandardOpenOption[] opts = resumed
				? new StandardOpenOption[] { StandardOpenOption.CREATE, StandardOpenOption.APPEND }
				: new StandardOpenOption[] { StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING };
		long startByte = existing;
		log.info(() -> (resumed ? "Resuming" : "Starting") + " download of " + asset.filename() + " from byte "
				+ startByte);
		try (InputStream in = res.body(); OutputStream out = Files.newOutputStream(partFile, opts)) {
			in.transferTo(out);
		}
		Files.move(partFile, dest, StandardCopyOption.REPLACE_EXISTING);

		String etag = res.headers().firstValue("ETag").orElse(remoteEtag);
		if (etag != null)
			Files.writeString(etagFile, etag);
		log.info(() -> "Downloaded " + dest);
		return dest;
	}

	private String headEtag(String url) throws IOException, InterruptedException {
		try {
			HttpRequest head = HttpRequest.newBuilder(URI.create(url)).method("HEAD", HttpRequest.BodyPublishers.noBody())
					.build();
			HttpResponse<Void> res = http.send(head, HttpResponse.BodyHandlers.discarding());
			if (res.statusCode() != 200)
				return null;
			return res.headers().firstValue("ETag").orElse(null);
		} catch (IOException e) {
			// ETag pre-check is best-effort — a HEAD failure just means "download fresh".
			log.fine(() -> "HEAD request failed for " + url + ": " + e.getMessage());
			return null;
		}
	}

	private static String readEtagFile(Path etagFile) {
		try {
			return Files.exists(etagFile) ? Files.readString(etagFile).strip() : null;
		} catch (IOException e) {
			return null;
		}
	}

	private static String sanitizeRepoDirName(String repo) {
		return repo.replace('/', '_').replace('\\', '_');
	}
}
