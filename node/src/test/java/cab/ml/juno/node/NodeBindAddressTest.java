package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.net.InetAddress;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * The address a node's gRPC server listens on: every interface unless {@code node.bind} names one. A local
 * cluster's forked nodes are given the loopback address, so a desktop firewall has nothing to ask about.
 */
@DisplayName("NodeBindAddress")
class NodeBindAddressTest {

	@Test
	@DisplayName("unset or blank means every interface")
	void unset_or_blank_means_every_interface() {
		assertThat(NodeBindAddress.resolve(null)).isNull();
		assertThat(NodeBindAddress.resolve("")).isNull();
		assertThat(NodeBindAddress.resolve("  ")).isNull();
	}

	@Test
	@DisplayName("a literal address resolves to itself")
	void a_literal_address_resolves_to_itself() throws Exception {
		assertThat(NodeBindAddress.resolve("127.0.0.1")).isEqualTo(InetAddress.getByName("127.0.0.1"));
		assertThat(NodeBindAddress.resolve(" 127.0.0.1 ").isLoopbackAddress()).isTrue();
	}

	@Test
	@DisplayName("the loopback value the cluster harness passes is the JVM's loopback address")
	void loopback_value_is_the_jvm_loopback_address() {
		assertThat(NodeBindAddress.resolve(NodeBindAddress.loopback())).isEqualTo(InetAddress.getLoopbackAddress());
	}

	@Test
	@DisplayName("an address that does not resolve is refused with one line naming the property")
	void an_unresolvable_address_is_refused() {
		assertThatThrownBy(() -> NodeBindAddress.resolve("no-such-host.invalid"))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining(NodeBindAddress.PROPERTY)
				.hasMessageContaining("no-such-host.invalid").message().doesNotContain("\n");
	}
}
