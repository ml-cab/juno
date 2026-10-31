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
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Runs Juno's unit tests with Maven in the checkout the kit was built from and reports each failed test as a
 * check: one JUnit tag group ({@code groups}: {@code gpu} on a CUDA machine, {@code rocm} on an AMD one), or
 * every test outside the excluded groups ({@code excludedGroups}), which is the CPU suite. Maven's output goes
 * to {@code <probe id>.maven.log} beside the report.
 */
final class UnitTestsProbe implements KitProbe {

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		String groups = args.getOrDefault("groups", "");
		String excluded = args.getOrDefault("excludedGroups", "");
		String modules = args.getOrDefault("modules", "node");
		Path root = Path.of(System.getProperty("juno.kit.root", ".")).toAbsolutePath().normalize();
		List<String> mvn = KitBootstrap.findMaven();
		if (mvn == null)
			throw new IOException("Maven not found: no mvn on PATH and no MAVEN_HOME");
		Path log = Path.of(args.getOrDefault("logDir", root.toString())).resolve(id + ".maven.log");
		long start = System.currentTimeMillis() - 1000;
		Process p = new ProcessBuilder(command(mvn, groups, excluded, modules)).directory(root.toFile())
				.redirectErrorStream(true).redirectOutput(log.toFile()).start();
		p.getOutputStream().close();
		int exit = p.waitFor();
		SurefireReports.Totals t = SurefireReports.read(SurefireReports.find(root, start));
		List<ProbeReport.Check> checks = new ArrayList<>();
		boolean ok = exit == 0 && t.tests() > 0 && t.failures() + t.errors() == 0;
		String what = groups.isEmpty() ? "tests outside " + excluded : "group " + groups;
		checks.add(new ProbeReport.Check(what + " on " + modules, ok,
				t.tests() + " tests, " + t.failures() + " failures, " + t.errors() + " errors, " + t.skipped()
						+ " skipped; Maven exit " + exit + (t.tests() == 0 ? "; no test ran" : "")));
		checks.addAll(t.failed());
		Map<String, Object> facts = new LinkedHashMap<>();
		facts.put("groups", groups);
		facts.put("excludedGroups", excluded);
		facts.put("modules", modules);
		facts.put("mavenExit", exit);
		facts.put("tests", t.tests());
		facts.put("skipped", t.skipped());
		return new ProbeReport(id, ProbeKind.CHECK, ok, checks, List.of(), facts);
	}

	/**
	 * Builds the named modules and what they need; runs the group's tests, or every test outside the excluded
	 * groups; failures reported, not fatal.
	 */
	static List<String> command(List<String> mvn, String groups, String excludedGroups, String modules) {
		List<String> cmd = new ArrayList<>(mvn);
		cmd.addAll(List.of("-B", "test"));
		if (!groups.isEmpty())
			cmd.add("-Dgroups=" + groups);
		if (!excludedGroups.isEmpty())
			cmd.add("-DexcludedGroups=" + excludedGroups);
		cmd.addAll(List.of("-pl", modules, "-am", "-Dsurefire.failIfNoSpecifiedTests=false",
				"-Dmaven.test.failure.ignore=true"));
		return cmd;
	}
}
