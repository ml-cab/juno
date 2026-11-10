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

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.junit.jupiter.api.Test;

class PrivacyTest {

	private final KitBootstrap.Privacy linux = new KitBootstrap.Privacy("alice", "/home/alice",
			List.of("alice-desktop", "alice-desktop.lan"));
	private final KitBootstrap.Privacy windows = new KitBootstrap.Privacy("Bob", "C:\\Users\\Bob",
			List.of("BOB-PC"));
	private final KitBootstrap.Privacy mac = new KitBootstrap.Privacy("carol", "/Users/carol",
			List.of("Carols-MacBook-Pro.local", "Carols-MacBook-Pro"));

	@Test
	void linuxLogLosesUserHomeHostAddressesAndMacs() {
		String in = """
				[INFO] Building juno-kit in /home/alice/src/juno
				user.name=alice host=alice-desktop.lan
				eth0: inet 192.168.1.23 ether 3c:97:0e:12:ab:cd inet6 fe80::1c2d:3e4f:5a6b:7c8d
				""";
		String out = linux.scrub(in);
		assertThat(out).doesNotContain("alice", "192.168.1.23", "3c:97:0e:12:ab:cd", "fe80::1c2d:3e4f:5a6b:7c8d");
		assertThat(out).contains("<home>/src/juno", "user.name=<user>", "host=<host>", "inet <ip>", "ether <mac>");
	}

	@Test
	void windowsLogIncludingJsonEscapedPathsAndDashedMacs() {
		String in = """
				Java home C:\\Users\\Bob\\.jdks\\jdk-25 on BOB-PC
				{"cache":"C:\\\\Users\\\\Bob\\\\.cache\\\\juno","alt":"C:/Users/Bob/.m2"}
				Physical Address. . . : 3C-97-0E-12-AB-CD  IPv4 Address. . . : 10.0.0.7
				""";
		String out = windows.scrub(in);
		assertThat(out).doesNotContain("Bob", "BOB-PC", "3C-97-0E-12-AB-CD", "10.0.0.7");
		assertThat(out).contains("<home>\\.jdks\\jdk-25 on <host>", "\"cache\":\"<home>\\\\.cache\\\\juno\"",
				"\"alt\":\"<home>/.m2\"");
	}

	@Test
	void macLogWithFullyQualifiedHostName() {
		String out = mac.scrub("Carols-MacBook-Pro.local: /Users/carol/Library/Caches/juno, owner carol\n");
		assertThat(out).isEqualTo("<host>: <home>/Library/Caches/juno, owner <user>\n");
	}

	@Test
	void nothingElseChanges() {
		String in = """
				2026-10-28T10:23:43Z probe cpu.throughput.tinyllama.512 PASS 12.34 t/s (min 12.01, max 12.60)
				CPU: Intel(R) Xeon(R) CPU E5-1650 v2 @ 3.50GHz, GPU: NVIDIA GeForce GTX 1080, driver 580.173.02
				JDK 25.0.1+8-LTS, Maven 3.9.9, listening on 127.0.0.1:18080 and [::1]:18080, std::vector<float>
				sha256 9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0 at 12:34:56.789
				[INFO] Including com.google.android:annotations:jar:4.1.1.4 in the shaded jar, build-1.2.3.4
				""";
		assertThat(linux.scrub(in)).isEqualTo(in);
		assertThat(windows.scrub(in)).isEqualTo(in);
		assertThat(mac.scrub(in)).isEqualTo(in);
	}

	@Test
	void anIpv4MappedIpv6AddressIsStillAnAddress() {
		assertThat(linux.scrub("peer /[::ffff:192.168.1.5]:443 and ::FFFF:10.0.0.9")).isEqualTo("peer /[<ip>]:443 and <ip>");
		assertThat(linux.scrub("inet addr:192.168.1.23  Bcast:192.168.1.255")).startsWith("inet addr:<ip>");
	}

	/**
	 * A JVM error file lists environment variables, and {@code JAVA_TOOL_OPTIONS} or {@code _JAVA_OPTIONS} can hold
	 * a proxy password. The block is left out; the rest of the file is scrubbed as usual and kept.
	 */
	@Test
	void aJvmErrorFilesEnvironmentBlockIsLeftOut() {
		String in = """
				Command Line: -Xmx4g -XX:ErrorFile=C:\\Users\\Bob\\juno\\nodes\\hs_err_pid%p.log cab.ml.juno.node.NodeMain
				Host: Intel(R) Core(TM) i5-1240P, 16 cores, 15G,  Windows 10 , 64 bit Build 19041

				Environment Variables:
				JAVA_TOOL_OPTIONS=-Dhttp.proxyPassword=s3cret
				PATH=C:\\Users\\Bob\\.jdks\\jdk-25\\bin;C:\\Windows\\system32
				USERNAME=Bob
				PROCESSOR_IDENTIFIER=Intel64 Family 6 Model 154 Stepping 3, GenuineIntel

				---------------  S Y S T E M  ---------------
				Memory: 4k page, system-wide physical 16108M (1210M free)
				""";
		String out = windows.scrub(in);
		assertThat(out).doesNotContain("s3cret", "PROCESSOR_IDENTIFIER", "USERNAME", "Bob");
		assertThat(out).contains("-XX:ErrorFile=<home>\\juno\\nodes\\hs_err_pid%p.log",
				"Host: Intel(R) Core(TM) i5-1240P", "Environment Variables: (left out by the kit)\n\n",
				"---------------  S Y S T E M  ---------------", "system-wide physical 16108M (1210M free)");
		assertThat(windows.scrub(in.replace("\n", "\r\n"))).doesNotContain("s3cret", "PROCESSOR_IDENTIFIER")
				.contains("S Y S T E M");
	}

	@Test
	void aShortUserNameIsReplacedInPathsOnlyNotInsideOrdinaryWords() {
		KitBootstrap.Privacy shortName = new KitBootstrap.Privacy("al", "/home/al", List.of("box1"));
		assertThat(shortName.scrub("/home/al/x total allocation al")).isEqualTo("<home>/x total allocation al");
	}
}
