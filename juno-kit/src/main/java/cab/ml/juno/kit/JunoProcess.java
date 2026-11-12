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


import java.io.IOException;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.MissingNode;

/**
 * Juno's launcher (the class the player jar starts, which {@code juno local} and {@code juno cluster} run) in
 * a child JVM with Juno's runtime flags, as the smoke scripts run it: a REST server held open on a free port,
 * or one console session fed a fixed input. Each runs in a directory of its own, where its log, its JFR
 * recording and its metrics land. A server is stopped by typing {@code exit} at its console, so its shutdown
 * hooks run (and its recording is written) on every OS; it is killed only if it does not stop.
 */
final class JunoProcess implements AutoCloseable {

	static final String LAUNCHER = "cab.ml.juno.player.ConsoleMain";
	static final String LOG = "launcher.log";

	/** How a one-off console session ended. */
	record Result(int exit, String output) {
	}

	/** An HTTP answer: status and body. */
	record Answer(int status, String body) {

		JsonNode json() throws IOException {
			return Json.MAPPER.readTree(body);
		}
	}

	private static final HttpClient HTTP = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build();
	/**
	 * One generation request's limit. A prompt near Phi-3's 4,096-position limit took over 20 minutes on the reference
	 * host's CPU; the probe's own timeout bounds the run.
	 */
	private static final Duration REQUEST_TIMEOUT = Duration.ofHours(2);

	private final Process process;
	private final Path dir;
	private final int port;

	private JunoProcess(Process process, Path dir, int port) {
		this.process = process;
		this.dir = dir;
		this.port = port;
	}

	/** The flags every smoke script gives the launcher for a greedy local run of {@code model}. */
	static List<String> localArgs(Path model, int maxTokens) {
		return new ArrayList<>(List.of("--model-path", model.toAbsolutePath().toString(), "--dtype", "FLOAT16", "--byteOrder", "BE",
				"--max-tokens", String.valueOf(maxTokens), "--temperature", "0", "--top-k", "0", "--top-p", "0",
				"--nodes", "1", "--local"));
	}

	/** One console session: {@code input} typed in, then the console closed; output is stdout and stderr. */
	static Result runOnce(Path dir, String heap, List<String> args, String input, Duration timeout)
			throws IOException, InterruptedException {
		Process p = start(dir, heap, List.of(), args);
		try (OutputStream in = p.getOutputStream()) {
			in.write(input.getBytes(StandardCharsets.UTF_8));
		} catch (IOException ignored) {
			// The launcher may have refused its arguments and exited before reading anything.
		}
		if (!p.waitFor(timeout.toSeconds(), TimeUnit.SECONDS)) {
			kill(p);
			return new Result(-1, KitBootstrap.readText(dir.resolve(LOG)) + "\n[no exit after " + timeout.toSeconds() + " s]");
		}
		return new Result(p.exitValue(), KitBootstrap.readText(dir.resolve(LOG)));
	}

	/** A REST server ({@code --api-port} is added), returned once {@code /v1/cluster/health} answers. */
	static JunoProcess serve(Path dir, String heap, List<String> args, Duration timeout)
			throws IOException, InterruptedException {
		return serve(dir, heap, List.of(), args, timeout);
	}

	/** {@link #serve(Path, String, List, Duration)} with JVM system properties ({@code -Dkey=value}) for the server. */
	static JunoProcess serve(Path dir, String heap, List<String> props, List<String> args, Duration timeout)
			throws IOException, InterruptedException {
		int port = freePort();
		List<String> all = new ArrayList<>(args);
		all.addAll(List.of("--api-port", String.valueOf(port)));
		return awaitApi(new JunoProcess(start(dir, heap, props, all), dir, port), timeout);
	}

	/**
	 * A REST server started by {@code command} as given (a launcher script, not the launcher class), with
	 * {@code --api-port} added; returned once {@code /v1/cluster/health} answers.
	 */
	static JunoProcess serveCommand(Path dir, List<String> command, Duration timeout)
			throws IOException, InterruptedException {
		int port = freePort();
		List<String> all = new ArrayList<>(command);
		all.addAll(List.of("--api-port", String.valueOf(port)));
		Files.createDirectories(dir);
		Path log = dir.resolve(LOG);
		Files.writeString(log, String.join(" ", all) + "\n");
		ProcessBuilder pb = new ProcessBuilder(all).directory(dir.toFile()).redirectErrorStream(true)
				.redirectOutput(ProcessBuilder.Redirect.appendTo(log.toFile()));
		noSharedPrefixes(pb);
		return awaitApi(new JunoProcess(pb.start(), dir, port), timeout);
	}

	private static JunoProcess awaitApi(JunoProcess s, Duration timeout) throws IOException, InterruptedException {
		Path dir = s.dir;
		long deadline = System.nanoTime() + timeout.toNanos();
		while (true) {
			if (!s.process.isAlive())
				throw new IOException("the launcher exited with code " + s.process.exitValue()
						+ " before its API answered; last lines: " + tail(dir.resolve(LOG), 15));
			try {
				if (s.get("/v1/cluster/health").status() == 200)
					return s;
			} catch (IOException notYet) {
				// not listening yet
			}
			if (System.nanoTime() > deadline) {
				s.close();
				throw new IOException("the launcher's API did not answer within " + timeout.toSeconds()
						+ " s; last lines: " + tail(dir.resolve(LOG), 15));
			}
			Thread.sleep(500);
		}
	}

	private static Process start(Path dir, String heap, List<String> props, List<String> args) throws IOException {
		Files.createDirectories(dir);
		List<String> all = new ArrayList<>(List.of("-Djuno.byteOrder=BE"));
		all.addAll(props);
		List<String> cmd = JvmLauncher.command(heap, false, all, LAUNCHER, args);
		Path log = dir.resolve(LOG);
		Files.writeString(log, String.join(" ", cmd) + "\n");
		ProcessBuilder pb = new ProcessBuilder(cmd).directory(dir.toFile()).redirectErrorStream(true)
				.redirectOutput(ProcessBuilder.Redirect.appendTo(log.toFile()));
		noSharedPrefixes(pb);
		return pb.start();
	}

	/**
	 * Probes send the same prompt more than once, and a reading would otherwise time a prefill the server skipped by
	 * starting from the first request's kept prompt; so shared prompt prefixes are off in the kit's servers unless the
	 * tester set {@code JUNO_PREFIX_CACHE_TOKENS} or a probe passes {@code --prefix-cache-tokens}, which wins.
	 */
	private static void noSharedPrefixes(ProcessBuilder pb) {
		pb.environment().putIfAbsent("JUNO_PREFIX_CACHE_TOKENS", "0");
	}

	URI base() {
		return URI.create("http://127.0.0.1:" + port);
	}

	Path dir() {
		return dir;
	}

	Answer get(String path) throws IOException, InterruptedException {
		HttpResponse<String> r = HTTP.send(HttpRequest.newBuilder(base().resolve(path)).timeout(Duration.ofSeconds(30))
				.GET().build(), HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
		return new Answer(r.statusCode(), r.body());
	}

	/** One JSON POST; a request may run as long as a generation on the slowest host takes. */
	Answer post(String path, String json) throws IOException, InterruptedException {
		HttpResponse<String> r = HTTP.send(HttpRequest.newBuilder(base().resolve(path)).timeout(REQUEST_TIMEOUT)
				.header("Content-Type", "application/json")
				.POST(HttpRequest.BodyPublishers.ofString(json, StandardCharsets.UTF_8)).build(),
				HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
		return new Answer(r.statusCode(), r.body());
	}

	/**
	 * One multipart POST as the vision route takes it, and as {@code curl -F} sends it: an {@code image} file part
	 * and a {@code request} part holding the chat request's JSON.
	 */
	Answer postImage(String path, Path image, String requestJson) throws IOException, InterruptedException {
		String boundary = "juno-kit-" + System.nanoTime();
		String name = image.getFileName().toString();
		String lower = name.toLowerCase(java.util.Locale.ROOT);
		String type = lower.endsWith(".png") ? "image/png"
				: lower.endsWith(".jpg") || lower.endsWith(".jpeg") ? "image/jpeg" : "application/octet-stream";
		java.io.ByteArrayOutputStream b = new java.io.ByteArrayOutputStream();
		b.writeBytes(("--" + boundary + "\r\nContent-Disposition: form-data; name=\"image\"; filename=\"" + name
				+ "\"\r\nContent-Type: " + type + "\r\n\r\n").getBytes(StandardCharsets.UTF_8));
		b.writeBytes(Files.readAllBytes(image));
		b.writeBytes(("\r\n--" + boundary + "\r\nContent-Disposition: form-data; name=\"request\"\r\n\r\n" + requestJson
				+ "\r\n--" + boundary + "--\r\n").getBytes(StandardCharsets.UTF_8));
		HttpResponse<String> r = HTTP.send(HttpRequest.newBuilder(base().resolve(path)).timeout(REQUEST_TIMEOUT)
				.header("Content-Type", "multipart/form-data; boundary=" + boundary)
				.POST(HttpRequest.BodyPublishers.ofByteArray(b.toByteArray())).build(),
				HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
		return new Answer(r.statusCode(), r.body());
	}

	/** The first model id {@code /v1/models} lists. */
	String modelId() throws IOException, InterruptedException {
		return get("/v1/models").json().path("data").path(0).path("id").asText("");
	}

	String log() throws IOException {
		return KitBootstrap.readText(dir.resolve(LOG));
	}

	/**
	 * The summary the launcher's {@code --jfr} writes into its directory on exit ({@code target/metrics/metrics.json},
	 * its {@code models[0].metrics}); a missing node before the server stopped or when it wrote none.
	 */
	JsonNode metrics() throws IOException {
		Path f = dir.resolve("target").resolve("metrics").resolve("metrics.json");
		return Files.isRegularFile(f) ? Json.MAPPER.readTree(f.toFile()).path("models").path(0).path("metrics")
				: MissingNode.getInstance();
	}

	/** The JFR recordings the launcher wrote into its directory ({@code --jfr}), after it stopped. */
	List<Path> recordings() throws IOException {
		try (Stream<Path> s = Files.list(dir)) {
			return s.filter(f -> f.getFileName().toString().endsWith(".jfr")).sorted().toList();
		}
	}

	/** The process and every process it started, as now. */
	List<ProcessHandle> tree() {
		List<ProcessHandle> all = new ArrayList<>(List.of(process.toHandle()));
		process.descendants().forEach(all::add);
		return all;
	}

	/**
	 * Types {@code exit} at the console and waits for the process to end by itself: its exit code, or -1 when it
	 * did not within 90 s and had to be stopped.
	 */
	int stop() {
		if (!process.isAlive())
			return process.exitValue();
		try (OutputStream in = process.getOutputStream()) {
			in.write("exit\n".getBytes(StandardCharsets.UTF_8));
		} catch (IOException ignored) {
			// console already closed
		}
		try {
			if (process.waitFor(90, TimeUnit.SECONDS))
				return process.exitValue();
			process.destroy();
			if (process.waitFor(30, TimeUnit.SECONDS))
				return -1;
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}
		kill(process);
		return -1;
	}

	@Override
	public void close() {
		stop();
	}

	/**
	 * Live processes started at or after {@code since} that are forked Juno nodes: matched by command line
	 * where the OS reports one, otherwise (Windows reports only the executable) any new {@code java}.
	 */
	static List<ProcessHandle> nodesStartedSince(Instant since) {
		long self = ProcessHandle.current().pid();
		return ProcessHandle.allProcesses().filter(ProcessHandle::isAlive).filter(h -> h.pid() != self)
				.filter(h -> h.info().startInstant().map(t -> !t.isBefore(since)).orElse(false)).filter(h -> {
					String cl = h.info().commandLine().orElse("");
					if (cl.contains(" "))
						return cl.contains("cab.ml.juno.node.NodeMain");
					String exe = Path.of(h.info().command().orElse("x")).getFileName().toString();
					return exe.equals("java") || exe.equals("java.exe");
				}).toList();
	}

	private static void kill(Process p) {
		p.descendants().forEach(ProcessHandle::destroyForcibly);
		p.destroyForcibly();
	}

	private static int freePort() throws IOException {
		try (ServerSocket s = new ServerSocket(0)) {
			return s.getLocalPort();
		}
	}

	static String tail(Path f, int lines) {
		try {
			List<String> all = KitBootstrap.readText(f).lines().toList();
			return String.join(" | ", all.subList(Math.max(0, all.size() - lines), all.size()));
		} catch (IOException e) {
			return "(no log: " + e.getMessage() + ")";
		}
	}
}
