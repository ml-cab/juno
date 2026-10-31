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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

/** Reads Maven Surefire's {@code TEST-*.xml} reports: totals and each failed test by name. */
final class SurefireReports {

	record Totals(int tests, int failures, int errors, int skipped, List<ProbeReport.Check> failed) {

		Totals {
			failed = List.copyOf(failed);
		}
	}

	private SurefireReports() {
	}

	static Totals read(List<Path> reports) throws IOException {
		int tests = 0, failures = 0, errors = 0, skipped = 0;
		List<ProbeReport.Check> failed = new ArrayList<>();
		DocumentBuilderFactory f = DocumentBuilderFactory.newInstance();
		try {
			f.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
			for (Path p : reports) {
				Document d = f.newDocumentBuilder().parse(p.toFile());
				Element suite = d.getDocumentElement();
				tests += intAttr(suite, "tests");
				failures += intAttr(suite, "failures");
				errors += intAttr(suite, "errors");
				skipped += intAttr(suite, "skipped");
				NodeList cases = suite.getElementsByTagName("testcase");
				for (int i = 0; i < cases.getLength(); i++) {
					Element c = (Element) cases.item(i);
					Element bad = first(c, "failure");
					if (bad == null)
						bad = first(c, "error");
					if (bad == null)
						continue;
					String cls = c.getAttribute("classname");
					String name = cls.substring(cls.lastIndexOf('.') + 1) + "." + c.getAttribute("name");
					failed.add(new ProbeReport.Check(name, false,
							bad.getTagName() + ": " + bad.getAttribute("type") + ": " + bad.getAttribute("message")));
				}
			}
		} catch (javax.xml.parsers.ParserConfigurationException | org.xml.sax.SAXException e) {
			throw new IOException("unreadable test report: " + e.getMessage(), e);
		}
		return new Totals(tests, failures, errors, skipped, failed);
	}

	/**
	 * Every {@code TEST-*.xml} in a module's {@code target/surefire-reports} under {@code root}, written at or
	 * after {@code sinceMillis}.
	 */
	static List<Path> find(Path root, long sinceMillis) throws IOException {
		try (Stream<Path> s = Files.walk(root, 4)) {
			return s.filter(p -> p.getFileName().toString().startsWith("TEST-") && p.toString().endsWith(".xml"))
					.filter(p -> p.getParent() != null && p.getParent().getFileName().toString().equals("surefire-reports"))
					.filter(p -> {
						try {
							return Files.getLastModifiedTime(p).toMillis() >= sinceMillis;
						} catch (IOException e) {
							return false;
						}
					}).sorted().toList();
		}
	}

	private static Element first(Element parent, String tag) {
		NodeList l = parent.getElementsByTagName(tag);
		return l.getLength() == 0 ? null : (Element) l.item(0);
	}

	private static int intAttr(Element e, String name) {
		String v = e.getAttribute(name);
		try {
			return v.isEmpty() ? 0 : Integer.parseInt(v);
		} catch (NumberFormatException ex) {
			return 0;
		}
	}
}
