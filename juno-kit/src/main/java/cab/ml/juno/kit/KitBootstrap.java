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


import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.net.InetAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * The kit's first stage, run by the {@code juno-kit.sh} and {@code juno-kit.bat} launchers straight from source before anything is
 * built: checks the JDK, Maven, disk and network, builds the kit with Maven, and starts it. A failed build
 * still produces a results zip holding the build log and the environment, because a build failure on a
 * tester's system is a result.
 *
 * <p>This file uses JDK classes only and no syntax newer than Java 17 (both held by a unit test), so an
 * older JDK reaches the version message instead of a compile error. It also holds the two pieces the
 * pre-build path needs and the kit reuses, so each has one implementation: {@link Privacy} (scrubbing)
 * and {@link #zip} (packing).
 */
public final class KitBootstrap {

	static final int MIN_JDK = 25;
	static final String RESULTS_DIR = "juno-kit-results";
	static final long MIN_BUILD_DISK_BYTES = 2L << 30;
	static final DateTimeFormatter UTC_STAMP = DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss'Z'", Locale.ROOT);

	/** The kit JVM's own flags; each probe runs in a JVM of its own with Juno's runtime flags. */
	static final List<String> KIT_JVM_FLAGS = List.of("--enable-preview", "--enable-native-access=ALL-UNNAMED",
			"--add-modules", "jdk.incubator.vector", "-Xmx1g");

	private KitBootstrap() {
	}

	public static void main(String[] args) {
		String problem = jdkProblem(Runtime.version().feature());
		if (problem != null) {
			System.err.println(problem);
			System.exit(2);
		}
		Path root = Path.of(System.getProperty("juno.kit.root", ".")).toAbsolutePath().normalize();
		List<String> mvn = findMaven();
		if (mvn == null) {
			System.err.println("juno-kit needs Maven 3.9 or later to build Juno, and found none: no mvn on PATH and"
					+ " no MAVEN_HOME. Install it (https://maven.apache.org/install.html or your package manager)"
					+ " and run the command again.");
			System.exit(2);
		}
		System.exit(run(root, List.of(args), mvn, System.out, true));
	}

	/** {@code null} when {@code feature} is new enough, otherwise what to tell the tester. */
	static String jdkProblem(int feature) {
		if (feature >= MIN_JDK)
			return null;
		return "juno-kit needs JDK " + MIN_JDK + " or later; this is JDK " + feature + ". Install a JDK " + MIN_JDK
				+ " build (for example from https://jdk.java.net or your package manager), put it first on PATH or"
				+ " set JAVA_HOME to it, and run the command again.";
	}

	/** The test seam: no network check. */
	static int run(Path root, List<String> args, List<String> mvn, PrintStream out) {
		return run(root, args, mvn, out, false);
	}

	static int run(Path root, List<String> args, List<String> mvn, PrintStream out, boolean checkNetwork) {
		Path jar = findKitJar(root);
		boolean plan = args.contains("--plan");
		boolean reuseJar = jar != null && (args.contains("--skip-build") || plan || firstIs(args, "resume")
				|| firstIs(args, "pack") || firstIs(args, "ingest"));
		if (plan && jar == null) {
			out.println("--plan needs the kit built, and building is a step --plan does not take. Run the kit"
					+ " without --plan: it builds first (" + buildEstimate() + ") and asks before it runs anything.");
			return 0;
		}
		String utc = ZonedDateTime.now(ZoneOffset.UTC).format(UTC_STAMP);
		Path results = root.resolve(RESULTS_DIR);
		Path buildLog = results.resolve("build-" + utc + ".log");
		if (!reuseJar) {
			try {
				Files.createDirectories(results);
				long free = Files.getFileStore(results).getUsableSpace();
				if (free < MIN_BUILD_DISK_BYTES) {
					out.println("Not enough free disk to build: " + (free >> 20) + " MiB free, "
							+ (MIN_BUILD_DISK_BYTES >> 20) + " MiB needed.");
					return 2;
				}
			} catch (IOException e) {
				out.println("Cannot write " + results + ": " + e.getMessage());
				return 2;
			}
			if (checkNetwork)
				checkNetwork(out);
			out.println("Building Juno and the kit (" + buildEstimate() + "). You may use the computer meanwhile.");
			int rc = build(root, mvn, buildLog, out);
			if (rc != 0) {
				Path zip = packBuildFailure(root, buildLog, utc, out);
				out.println("The build failed (exit code " + rc + "). Please send this file back: "
						+ (zip != null ? zip : buildLog));
				return 1;
			}
			jar = findKitJar(root);
			if (jar == null) {
				out.println("The build succeeded but produced no kit jar under juno-kit/target.");
				return 1;
			}
		}
		return launch(root, jar, args, reuseJar ? null : buildLog);
	}

	private static boolean firstIs(List<String> args, String sub) {
		return !args.isEmpty() && args.get(0).equals(sub);
	}

	private static String buildEstimate() {
		Path m2 = Path.of(System.getProperty("user.home"), ".m2", "repository", "io", "grpc");
		return Files.isDirectory(m2) ? "about 2 to 5 minutes"
				: "about 5 to 15 minutes; Maven downloads Juno's dependencies first";
	}

	static List<String> findMaven() {
		boolean windows = isWindows();
		List<Path> candidates = new ArrayList<>();
		for (String env : List.of("MAVEN_HOME", "M2_HOME")) {
			String home = System.getenv(env);
			if (home != null && !home.isBlank())
				candidates.add(Path.of(home, "bin", windows ? "mvn.cmd" : "mvn"));
		}
		String path = System.getenv("PATH");
		if (path != null)
			for (String dir : path.split(java.io.File.pathSeparator))
				if (!dir.isBlank())
					candidates.add(Path.of(dir, windows ? "mvn.cmd" : "mvn"));
		for (Path c : candidates)
			if (Files.isRegularFile(c))
				return windows ? List.of("cmd.exe", "/c", c.toString()) : List.of(c.toString());
		return null;
	}

	private static void checkNetwork(PrintStream out) {
		HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5))
				.followRedirects(HttpClient.Redirect.NORMAL).build();
		List<String> urls = List.of("https://repo.maven.apache.org/maven2/", "https://huggingface.co/");
		List<CompletableFuture<Integer>> calls = new ArrayList<>();
		for (String u : urls)
			calls.add(http
					.sendAsync(HttpRequest.newBuilder(URI.create(u)).timeout(Duration.ofSeconds(10))
							.method("HEAD", HttpRequest.BodyPublishers.noBody()).build(),
							HttpResponse.BodyHandlers.discarding())
					.thenApply(HttpResponse::statusCode).exceptionally(t -> -1));
		for (int i = 0; i < urls.size(); i++) {
			int code = calls.get(i).join();
			out.println("  " + URI.create(urls.get(i)).getHost() + ": "
					+ (code > 0 && code < 500 ? "reachable" : "NOT reachable"));
		}
	}

	private static int build(Path root, List<String> mvn, Path buildLog, PrintStream out) {
		List<String> cmd = new ArrayList<>(mvn);
		cmd.addAll(List.of("-B", "package", "-DskipTests", "-pl", "juno-kit", "-am"));
		try {
			Process p = new ProcessBuilder(cmd).directory(root.toFile()).redirectErrorStream(true).start();
			p.getOutputStream().close();
			try (BufferedReader in = new BufferedReader(new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8));
					PrintStream log = new PrintStream(Files.newOutputStream(buildLog), true, StandardCharsets.UTF_8)) {
				log.println("$ " + String.join(" ", cmd));
				String line;
				while ((line = in.readLine()) != null) {
					log.println(line);
					if (line.matches("\\[INFO\\] Building .*\\[\\d+/\\d+\\]") || line.startsWith("[ERROR]")
							|| line.contains("BUILD SUCCESS") || line.contains("BUILD FAILURE"))
						out.println("  " + line);
				}
			}
			return p.waitFor();
		} catch (IOException e) {
			try {
				Files.writeString(buildLog, "could not start Maven: " + e + "\n");
			} catch (IOException ignored) {
			}
			return 127;
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			return 130;
		}
	}

	private static int launch(Path root, Path jar, List<String> args, Path buildLog) {
		List<String> cmd = new ArrayList<>();
		cmd.add(javaBinary());
		cmd.addAll(KIT_JVM_FLAGS);
		cmd.add("-Djuno.kit.root=" + root);
		if (buildLog != null)
			cmd.add("-Djuno.kit.buildLog=" + buildLog);
		cmd.addAll(List.of("-jar", jar.toString()));
		cmd.addAll(args);
		try {
			return new ProcessBuilder(cmd).directory(root.toFile()).inheritIO().start().waitFor();
		} catch (IOException e) {
			System.err.println("Could not start the kit: " + e.getMessage());
			return 1;
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			return 130;
		}
	}

	static String javaBinary() {
		return Path.of(System.getProperty("java.home"), "bin", isWindows() ? "java.exe" : "java").toString();
	}

	static Path findKitJar(Path root) {
		Path target = root.resolve("juno-kit").resolve("target");
		if (!Files.isDirectory(target))
			return null;
		try (Stream<Path> s = Files.list(target)) {
			return s.filter(p -> p.getFileName().toString().matches("juno-kit-.*-shaded\\.jar")).findFirst()
					.orElse(null);
		} catch (IOException e) {
			return null;
		}
	}

	private static Path packBuildFailure(Path root, Path buildLog, String utc, PrintStream out) {
		try {
			Path dir = root.resolve(RESULTS_DIR).resolve(utc + "-build-failure");
			Files.createDirectories(dir);
			if (Files.exists(buildLog))
				Files.copy(buildLog, dir.resolve("build.log"));
			Files.writeString(dir.resolve("environment.txt"), environment());
			String[] commit = junoCommit(root);
			Files.writeString(dir.resolve("manifest.json"), buildFailureManifest(kitVersion(root), commit[0],
					Boolean.parseBoolean(commit[1]), utc, System.getProperty("os.name"), System.getProperty("os.arch")));
			Path zip = root.resolve(RESULTS_DIR).resolve(zipName(utc));
			zip(dir, zip, Privacy.fromSystem());
			return zip;
		} catch (IOException e) {
			out.println("Could not pack the build failure: " + e.getMessage());
			return null;
		}
	}

	/** {@code juno-kit-results-<os>-<arch>-<utc>.zip}. */
	static String zipName(String utc) {
		String os = System.getProperty("os.name").toLowerCase(Locale.ROOT);
		String family = os.contains("win") ? "windows" : os.contains("mac") ? "macos" : os.replace(' ', '-');
		return "juno-kit-results-" + family + "-" + System.getProperty("os.arch") + "-" + utc + ".zip";
	}

	private static String environment() {
		StringBuilder sb = new StringBuilder();
		for (String k : List.of("os.name", "os.version", "os.arch", "java.version", "java.vendor", "java.vm.name",
				"java.home"))
			sb.append(k).append('=').append(System.getProperty(k)).append('\n');
		sb.append("availableProcessors=").append(Runtime.getRuntime().availableProcessors()).append('\n');
		return sb.toString();
	}

	/** The manifest of a run that never got past the build; the fields are {@code Manifest}'s. */
	static String buildFailureManifest(String kitVersion, String commit, boolean dirty, String startedUtc, String os,
			String arch) {
		return "{\"schemaVersion\":1,\"kind\":\"build-failure\",\"kitVersion\":" + json(kitVersion)
				+ ",\"junoCommit\":" + json(commit) + ",\"junoTreeDirty\":" + dirty
				+ ",\"kitJarSha256\":null,\"suite\":null,\"probes\":[],\"startedUtc\":" + json(startedUtc)
				+ ",\"finishedUtc\":" + json(startedUtc) + ",\"host\":{\"os\":" + json(os) + ",\"arch\":" + json(arch)
				+ "}}\n";
	}

	private static String json(String s) {
		if (s == null)
			return "null";
		StringBuilder sb = new StringBuilder("\"");
		for (char c : s.toCharArray()) {
			if (c == '"' || c == '\\')
				sb.append('\\').append(c);
			else if (c < 0x20)
				sb.append(String.format(Locale.ROOT, "\\u%04x", (int) c));
			else
				sb.append(c);
		}
		return sb.append('"').toString();
	}

	/** The project version from the root pom, or {@code unknown}. */
	static String kitVersion(Path root) {
		try {
			String pom = Files.readString(root.resolve("pom.xml"));
			Matcher m = Pattern.compile("<artifactId>juno</artifactId>\\s*<version>([^<]+)</version>").matcher(pom);
			return m.find() ? m.group(1).trim() : "unknown";
		} catch (IOException e) {
			return "unknown";
		}
	}

	/**
	 * {commit, dirty}: from git where the checkout has it, else from the folder name of a GitHub commit
	 * archive ({@code <repo>-<40 hex>}), else {@code unknown}.
	 */
	static String[] junoCommit(Path root) {
		String head = runQuietly(root, List.of("git", "rev-parse", "HEAD"));
		if (head != null && head.trim().matches("[0-9a-f]{40}")) {
			String status = runQuietly(root, List.of("git", "status", "--porcelain", "--untracked-files=no"));
			return new String[] { head.trim(), String.valueOf(status != null && !status.isBlank()) };
		}
		Path name = root.getFileName();
		String fromName = name == null ? null : commitFromFolderName(name.toString());
		return new String[] { fromName != null ? fromName : "unknown", "false" };
	}

	static String commitFromFolderName(String folder) {
		Matcher m = Pattern.compile("^[A-Za-z0-9._]+(?:-[A-Za-z0-9._]+)*-([0-9a-f]{40})$").matcher(folder);
		return m.matches() ? m.group(1) : null;
	}

	/** stdout of {@code cmd} run in {@code dir}, or {@code null} when it fails, exits non-zero or takes over 20 s. */
	static String runQuietly(Path dir, List<String> cmd) {
		try {
			Process p = new ProcessBuilder(cmd).directory(dir.toFile()).redirectErrorStream(false).start();
			p.getOutputStream().close();
			byte[] outBytes = p.getInputStream().readAllBytes();
			if (!p.waitFor(20, TimeUnit.SECONDS)) {
				p.destroyForcibly();
				return null;
			}
			return p.exitValue() == 0 ? new String(outBytes, StandardCharsets.UTF_8) : null;
		} catch (IOException e) {
			return null;
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			return null;
		}
	}

	static boolean isWindows() {
		return System.getProperty("os.name").toLowerCase(Locale.ROOT).contains("win");
	}

	/** File types packed into a results zip; everything else (JFR recordings, models, jars) stays out. */
	static final Set<String> PACKED_EXTENSIONS = Set.of("json", "log", "txt", "html", "xml", "md", "csv",
			"properties");

	/**
	 * Zips every text file under {@code dir} whose extension is in {@link #PACKED_EXTENSIONS}, scrubbed by
	 * {@code privacy}, skipping {@code zip} itself.
	 */
	static Path zip(Path dir, Path zip, Privacy privacy) throws IOException {
		Files.createDirectories(zip.toAbsolutePath().getParent());
		List<Path> files;
		try (Stream<Path> s = Files.walk(dir)) {
			files = s.filter(Files::isRegularFile).filter(p -> !p.toAbsolutePath().equals(zip.toAbsolutePath()))
					.filter(p -> PACKED_EXTENSIONS.contains(extension(p))).sorted(Comparator.naturalOrder()).toList();
		}
		try (OutputStream os = Files.newOutputStream(zip); ZipOutputStream z = new ZipOutputStream(os)) {
			for (Path f : files) {
				String name = dir.relativize(f).toString().replace('\\', '/');
				z.putNextEntry(new ZipEntry(name));
				z.write(privacy.scrub(Files.readString(f, StandardCharsets.UTF_8)).getBytes(StandardCharsets.UTF_8));
				z.closeEntry();
			}
		}
		return zip;
	}

	private static String extension(Path p) {
		String n = p.getFileName().toString();
		int dot = n.lastIndexOf('.');
		return dot < 0 ? "" : n.substring(dot + 1).toLowerCase(Locale.ROOT);
	}

	/**
	 * Replaces personal data in results text with placeholders: the home directory (each slash style, and
	 * JSON-escaped) with {@code <home>}, host names with {@code <host>}, the user name with {@code <user>}, and
	 * MAC and IP addresses with {@code <mac>} and {@code <ip>}. Loopback addresses and hardware names stay. A
	 * user name shorter than three characters is replaced only inside the home path, so it cannot rewrite
	 * ordinary words.
	 */
	static final class Privacy {

		private static final Pattern MAC = Pattern.compile(
				"(?<![0-9A-Fa-f:-])[0-9A-Fa-f]{2}([:-])(?:[0-9A-Fa-f]{2}\\1){4}[0-9A-Fa-f]{2}(?![0-9A-Fa-f:-])");
		private static final Pattern IPV6_FULL = Pattern
				.compile("(?<![0-9A-Fa-f:\\w])(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}(?![0-9A-Fa-f:\\w])");
		private static final Pattern IPV6_SHORT = Pattern.compile("(?<![0-9A-Fa-f:\\w])"
				+ "(?:[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4}){0,6})?::(?:[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4}){0,6})?"
				+ "(?![0-9A-Fa-f:\\w])");
		private static final String OCTET = "(?:25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]?\\d)";
		/**
		 * Not after {@code :} or {@code -}, where a dotted quad is a version ({@code jar:4.1.1.4},
		 * {@code build-1.2.3.4}), except net-tools' {@code addr:}.
		 */
		private static final Pattern IPV4 = Pattern
				.compile("(?:(?<![\\w.:-])|(?<=addr:))(?:" + OCTET + "\\.){3}" + OCTET + "(?![\\w.])");
		private static final Pattern IPV4_MAPPED = Pattern.compile(
				"(?i)(?<![0-9A-Fa-f:\\w])(?:0{1,4}:){0,5}::?ffff:(?:" + OCTET + "\\.){3}" + OCTET + "(?![\\w.])");

		private final List<Pattern> homes = new ArrayList<>();
		private final List<Pattern> hosts = new ArrayList<>();
		private final Pattern user;

		Privacy(String userName, String home, List<String> hostNames) {
			Set<String> variants = new LinkedHashSet<>();
			if (home != null && home.length() > 1) {
				variants.add(home);
				variants.add(home.replace('\\', '/'));
				variants.add(home.replace('/', '\\'));
				variants.add(home.replace("\\", "\\\\"));
			}
			variants.stream().sorted(Comparator.comparingInt(String::length).reversed())
					.forEach(v -> homes.add(Pattern.compile(Pattern.quote(v) + "(?![\\w-])", Pattern.CASE_INSENSITIVE)));
			hostNames.stream().filter(h -> h != null && h.length() >= 2 && !h.equalsIgnoreCase("localhost")).distinct()
					.sorted(Comparator.comparingInt(String::length).reversed())
					.forEach(h -> hosts.add(word(h, Pattern.CASE_INSENSITIVE)));
			user = userName != null && userName.length() >= 3 ? word(userName, 0) : null;
		}

		private static Pattern word(String literal, int flags) {
			return Pattern.compile("(?<![\\w-])" + Pattern.quote(literal) + "(?![\\w-])", flags);
		}

		/** This JVM's user, home and host names. */
		static Privacy fromSystem() {
			List<String> hostNames = new ArrayList<>();
			try {
				InetAddress local = InetAddress.getLocalHost();
				hostNames.add(local.getHostName());
				hostNames.add(local.getCanonicalHostName());
			} catch (IOException ignored) {
			}
			for (String env : List.of("HOSTNAME", "COMPUTERNAME"))
				if (System.getenv(env) != null)
					hostNames.add(System.getenv(env));
			List<String> withShort = new ArrayList<>(hostNames);
			for (String h : hostNames)
				if (h != null && h.indexOf('.') > 0 && !Character.isDigit(h.charAt(0)))
					withShort.add(h.substring(0, h.indexOf('.')));
			return new Privacy(System.getProperty("user.name"), System.getProperty("user.home"), withShort);
		}

		String scrub(String text) {
			String s = text;
			for (Pattern p : homes)
				s = p.matcher(s).replaceAll(Matcher.quoteReplacement("<home>"));
			for (Pattern p : hosts)
				s = p.matcher(s).replaceAll("<host>");
			if (user != null)
				s = user.matcher(s).replaceAll("<user>");
			s = MAC.matcher(s).replaceAll("<mac>");
			s = IPV4_MAPPED.matcher(s).replaceAll("<ip>");
			s = IPV6_FULL.matcher(s).replaceAll("<ip>");
			s = IPV6_SHORT.matcher(s).replaceAll(m -> m.group().equals("::") || m.group().equals("::1")
					? Matcher.quoteReplacement(m.group()) : "<ip>");
			s = IPV4.matcher(s).replaceAll(m -> m.group().startsWith("127.") || m.group().equals("0.0.0.0")
					? m.group() : "<ip>");
			return s;
		}
	}
}
