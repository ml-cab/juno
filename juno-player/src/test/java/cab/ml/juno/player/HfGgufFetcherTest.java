package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.http.HttpClient;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

/**
 * {@link HfGgufFetcher} tests run entirely against a local
 * {@link HttpServer} mock — no real network access, per
 * {@code docs/infra-plan/PLAN-Infra-Tier7.md} ("no Python dependency
 * anywhere in this path" and "test against a local mock HTTP server").
 */
class HfGgufFetcherTest {

	private HttpServer server;
	private final AtomicInteger fullGetCount = new AtomicInteger();
	private final AtomicInteger headCount = new AtomicInteger();

	private static final String REPO = "acme/tiny-model-GGUF";
	private static final byte[] Q4_CONTENT = "Q4_K_M-GGUF-BYTES-CONTENT".getBytes(StandardCharsets.UTF_8);
	private static final byte[] Q8_CONTENT = "Q8_0-GGUF-BYTES-CONTENT".getBytes(StandardCharsets.UTF_8);
	private static final String Q4_ETAG = "\"etag-q4\"";

	private String baseUrl;

	private void startServer() throws IOException {
		server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
		server.createContext("/api/models/" + REPO, ex -> {
			String body = "{\"siblings\":[" + "{\"rfilename\":\"tiny-model.Q4_K_M.gguf\"},"
					+ "{\"rfilename\":\"tiny-model.Q8_0.gguf\"}," + "{\"rfilename\":\"README.md\"}" + "]}";
			byte[] b = body.getBytes(StandardCharsets.UTF_8);
			ex.getResponseHeaders().add("Content-Type", "application/json");
			ex.sendResponseHeaders(200, b.length);
			try (OutputStream os = ex.getResponseBody()) {
				os.write(b);
			}
		});
		server.createContext("/" + REPO + "/resolve/main/tiny-model.Q4_K_M.gguf", ex -> serveAsset(ex, Q4_CONTENT, Q4_ETAG));
		server.createContext("/" + REPO + "/resolve/main/tiny-model.Q8_0.gguf",
				ex -> serveAsset(ex, Q8_CONTENT, "\"etag-q8\""));
		server.start();
		baseUrl = "http://127.0.0.1:" + server.getAddress().getPort();
	}

	private void serveAsset(HttpExchange ex, byte[] content, String etag) throws IOException {
		ex.getResponseHeaders().add("ETag", etag);
		if ("HEAD".equals(ex.getRequestMethod())) {
			headCount.incrementAndGet();
			ex.sendResponseHeaders(200, -1);
			ex.close();
			return;
		}
		String range = ex.getRequestHeaders().getFirst("Range");
		if (range != null && range.startsWith("bytes=")) {
			int start = Integer.parseInt(range.substring(6, range.length() - 1));
			byte[] slice = java.util.Arrays.copyOfRange(content, Math.min(start, content.length), content.length);
			ex.getResponseHeaders().add("Content-Range",
					"bytes " + start + "-" + (content.length - 1) + "/" + content.length);
			ex.sendResponseHeaders(206, slice.length);
			try (OutputStream os = ex.getResponseBody()) {
				os.write(slice);
			}
			return;
		}
		fullGetCount.incrementAndGet();
		ex.sendResponseHeaders(200, content.length);
		try (OutputStream os = ex.getResponseBody()) {
			os.write(content);
		}
	}

	@AfterEach
	void stopServer() {
		if (server != null)
			server.stop(0);
	}

	private HfGgufFetcher fetcher(Path cacheDir) {
		// Mirrors the redirect-following client HfGgufFetcher() builds in production
		// (see its no-arg constructor) — the real Hub's /resolve/main/... endpoint
		// 302s to its CDN, so a plain HttpClient.newHttpClient() here (default
		// Redirect.NEVER) would let every download test pass while masking that gap.
		return new HfGgufFetcher(HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build(), baseUrl,
				cacheDir);
	}

	// ── Spec parsing ─────────────────────────────────────────────────────────

	@Test
	void spec_parses_repo_without_quant() {
		HfGgufFetcher.Spec sp = HfGgufFetcher.Spec.parse("org/repo");
		assertThat(sp.repo()).isEqualTo("org/repo");
		assertThat(sp.quant()).isNull();
	}

	@Test
	void spec_parses_repo_with_quant() {
		HfGgufFetcher.Spec sp = HfGgufFetcher.Spec.parse("org/repo:Q8_0");
		assertThat(sp.repo()).isEqualTo("org/repo");
		assertThat(sp.quant()).isEqualTo("Q8_0");
	}

	@Test
	void spec_rejects_blank() {
		assertThatThrownBy(() -> HfGgufFetcher.Spec.parse("  ")).isInstanceOf(IllegalArgumentException.class);
	}

	// ── Asset selection ──────────────────────────────────────────────────────

	@Test
	void selectAsset_prefers_q4_k_m_by_default() {
		HfGgufFetcher f = new HfGgufFetcher(HttpClient.newHttpClient(), "http://unused", Path.of("unused"));
		List<HfGgufFetcher.Asset> assets = List.of(new HfGgufFetcher.Asset("m.Q8_0.gguf"),
				new HfGgufFetcher.Asset("m.Q4_K_M.gguf"));
		assertThat(f.selectAsset(assets, null).filename()).isEqualTo("m.Q4_K_M.gguf");
	}

	@Test
	void selectAsset_honors_explicit_quant() {
		HfGgufFetcher f = new HfGgufFetcher(HttpClient.newHttpClient(), "http://unused", Path.of("unused"));
		List<HfGgufFetcher.Asset> assets = List.of(new HfGgufFetcher.Asset("m.Q8_0.gguf"),
				new HfGgufFetcher.Asset("m.Q4_K_M.gguf"));
		assertThat(f.selectAsset(assets, "q8_0").filename()).isEqualTo("m.Q8_0.gguf");
	}

	@Test
	void selectAsset_falls_back_to_first_alphabetical_when_no_preferred_quant_present() {
		HfGgufFetcher f = new HfGgufFetcher(HttpClient.newHttpClient(), "http://unused", Path.of("unused"));
		List<HfGgufFetcher.Asset> assets = List.of(new HfGgufFetcher.Asset("m.IQ4_XS.gguf"),
				new HfGgufFetcher.Asset("m.Q2_K.gguf"));
		assertThat(f.selectAsset(assets, null).filename()).isEqualTo("m.IQ4_XS.gguf");
	}

	@Test
	void selectAsset_throws_when_requested_quant_not_found() {
		HfGgufFetcher f = new HfGgufFetcher(HttpClient.newHttpClient(), "http://unused", Path.of("unused"));
		List<HfGgufFetcher.Asset> assets = List.of(new HfGgufFetcher.Asset("m.Q4_K_M.gguf"));
		assertThatThrownBy(() -> f.selectAsset(assets, "Q9_IMPOSSIBLE")).isInstanceOf(IllegalArgumentException.class);
	}

	// ── End-to-end against the mock server ──────────────────────────────────

	@Test
	void resolve_downloads_the_preferred_quant_when_no_quant_requested(@TempDir Path cacheDir) throws Exception {
		startServer();
		HfGgufFetcher f = fetcher(cacheDir);

		Path result = f.resolve(REPO);

		assertThat(result.getFileName().toString()).isEqualTo("tiny-model.Q4_K_M.gguf");
		assertThat(Files.readAllBytes(result)).isEqualTo(Q4_CONTENT);
		assertThat(fullGetCount.get()).isEqualTo(1);
	}

	@Test
	void resolve_downloads_the_explicitly_requested_quant(@TempDir Path cacheDir) throws Exception {
		startServer();
		HfGgufFetcher f = fetcher(cacheDir);

		Path result = f.resolve(REPO + ":Q8_0");

		assertThat(result.getFileName().toString()).isEqualTo("tiny-model.Q8_0.gguf");
		assertThat(Files.readAllBytes(result)).isEqualTo(Q8_CONTENT);
	}

	@Test
	void listGgufAssets_filters_out_non_gguf_siblings(@TempDir Path cacheDir) throws Exception {
		startServer();
		HfGgufFetcher f = fetcher(cacheDir);

		List<HfGgufFetcher.Asset> assets = f.listGgufAssets(REPO);

		assertThat(assets).extracting(HfGgufFetcher.Asset::filename).containsExactlyInAnyOrder(
				"tiny-model.Q4_K_M.gguf", "tiny-model.Q8_0.gguf");
	}

	@Test
	void second_resolve_is_a_cache_hit_and_does_not_re_download(@TempDir Path cacheDir) throws Exception {
		startServer();
		HfGgufFetcher f = fetcher(cacheDir);

		f.resolve(REPO);
		assertThat(fullGetCount.get()).isEqualTo(1);

		Path result2 = f.resolve(REPO);
		assertThat(fullGetCount.get()).as("ETag cache hit must not re-download the body").isEqualTo(1);
		assertThat(Files.readAllBytes(result2)).isEqualTo(Q4_CONTENT);
	}

	@Test
	void resume_appends_to_a_partial_download_instead_of_restarting(@TempDir Path cacheDir) throws Exception {
		startServer();
		HfGgufFetcher f = fetcher(cacheDir);

		// Simulate a previous interrupted download: first N bytes already on disk
		// as the ".part" file the fetcher's download() looks for.
		Path destDir = cacheDir.resolve(REPO.replace('/', '_'));
		Files.createDirectories(destDir);
		Path partFile = destDir.resolve("tiny-model.Q4_K_M.gguf.part");
		int already = 6;
		Files.write(partFile, java.util.Arrays.copyOfRange(Q4_CONTENT, 0, already));

		Path result = f.download(REPO, new HfGgufFetcher.Asset("tiny-model.Q4_K_M.gguf"));

		assertThat(Files.readAllBytes(result)).isEqualTo(Q4_CONTENT);
		// The mock server only ever saw a ranged request for the remainder, never a
		// second full-content GET — proves resume, not restart-from-scratch.
		assertThat(fullGetCount.get()).isEqualTo(0);
	}

	@Test
	void resolve_follows_a_redirect_from_the_download_endpoint_like_the_real_hub_cdn(@TempDir Path cacheDir)
			throws Exception {
		// The real huggingface.co /resolve/main/<file> endpoint 302s the actual bytes
		// off to its CDN (cdn-lfs.huggingface.co). Reproduce that here so this test
		// fails on a client built with the JDK's default Redirect.NEVER, the way a
		// live "./juno local --hf org/repo" run did before HfGgufFetcher()'s
		// constructor was fixed to opt into Redirect.NORMAL.
		startServer();
		server.createContext("/" + REPO + "/resolve/main/tiny-model.Q4_K_M-redirect.gguf", ex -> {
			ex.getResponseHeaders().add("Location", baseUrl + "/cdn/tiny-model.Q4_K_M-redirect.gguf");
			ex.sendResponseHeaders(302, -1);
			ex.close();
		});
		server.createContext("/cdn/tiny-model.Q4_K_M-redirect.gguf", ex -> serveAsset(ex, Q4_CONTENT, Q4_ETAG));
		HfGgufFetcher f = fetcher(cacheDir);

		Path result = f.download(REPO, new HfGgufFetcher.Asset("tiny-model.Q4_K_M-redirect.gguf"));

		assertThat(Files.readAllBytes(result)).isEqualTo(Q4_CONTENT);
	}

	@Test
	void resolve_reports_clear_error_for_unknown_repo(@TempDir Path cacheDir) throws IOException {
		startServer();
		server.createContext("/api/models/nobody/nothing", ex -> {
			ex.sendResponseHeaders(404, -1);
			ex.close();
		});
		HfGgufFetcher f = fetcher(cacheDir);

		assertThatThrownBy(() -> f.resolve("nobody/nothing")).isInstanceOf(IOException.class)
				.hasMessageContaining("nobody/nothing").hasMessageContaining("not found");
	}

	@Test
	void resolve_reports_clear_error_for_401_the_same_as_404(@TempDir Path cacheDir) throws IOException {
		// The real Hub API returns 401 "Invalid username or password" — not 404 —
		// for both a nonexistent repo and a private/gated one it won't disclose to
		// an anonymous caller, so this must read like the 404 case above rather
		// than a raw, confusing HTTP-401 status dump.
		startServer();
		server.createContext("/api/models/private/gated-or-typo", ex -> {
			ex.sendResponseHeaders(401, -1);
			ex.close();
		});
		HfGgufFetcher f = fetcher(cacheDir);

		assertThatThrownBy(() -> f.resolve("private/gated-or-typo")).isInstanceOf(IOException.class)
				.hasMessageContaining("private/gated-or-typo").hasMessageContaining("not found");
	}

	@Test
	void defaultCacheDir_is_under_user_home_cache_juno_models_not_the_fixtures_dir() {
		Path dir = HfGgufFetcher.defaultCacheDir();
		assertThat(dir.toString()).contains(".cache").contains("juno").contains("models");
		assertThat(dir).isEqualTo(Path.of(System.getProperty("user.home"), ".cache", "juno", "models"));
	}
}
