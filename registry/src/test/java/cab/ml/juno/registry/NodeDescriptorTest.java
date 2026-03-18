package cab.ml.juno.registry;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.time.Instant;

import org.junit.jupiter.api.Test;

class NodeDescriptorTest {

	private NodeDescriptor node(long vramTotal, long vramFree, NodeStatus status) {
		return new NodeDescriptor("n1", "192.168.1.10", 9092, vramTotal, vramFree, status, 0.8, Instant.now(),
				Instant.now());
	}

	@Test
	void vram_pressure_is_correct_fraction() {
		NodeDescriptor n = node(4_000_000_000L, 1_000_000_000L, NodeStatus.READY);
		assertThat(n.vramPressure()).isCloseTo(0.75, within(0.001));
	}

	@Test
	void vram_pressure_is_zero_when_fully_free() {
		NodeDescriptor n = node(4_000_000_000L, 4_000_000_000L, NodeStatus.IDLE);
		assertThat(n.vramPressure()).isCloseTo(0.0, within(0.001));
	}

	@Test
	void usable_vram_reserves_ten_percent_headroom() {
		NodeDescriptor n = node(4_000_000_000L, 4_000_000_000L, NodeStatus.IDLE);
		assertThat(n.usableVramBytes()).isEqualTo(3_600_000_000L);
	}

	@Test
	void is_available_for_idle_and_ready() {
		assertThat(node(1, 1, NodeStatus.IDLE).isAvailable()).isTrue();
		assertThat(node(1, 1, NodeStatus.READY).isAvailable()).isTrue();
		assertThat(node(1, 0, NodeStatus.OFFLINE).isAvailable()).isFalse();
		assertThat(node(1, 0, NodeStatus.DEGRADED).isAvailable()).isFalse();
	}

	@Test
	void withStatus_returns_new_record_with_updated_status() {
		NodeDescriptor n = node(4_000_000_000L, 4_000_000_000L, NodeStatus.IDLE);
		NodeDescriptor updated = n.withStatus(NodeStatus.READY);
		assertThat(updated.status()).isEqualTo(NodeStatus.READY);
		assertThat(n.status()).isEqualTo(NodeStatus.IDLE); // original unchanged
	}

	@Test
	void rejects_invalid_grpc_port() {
		assertThatThrownBy(
				() -> new NodeDescriptor("n1", "host", 99999, 1, 1, NodeStatus.IDLE, 0, Instant.now(), Instant.now()))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	void rejects_vram_free_exceeding_total() {
		assertThatThrownBy(() -> node(1_000L, 2_000L, NodeStatus.IDLE)).isInstanceOf(IllegalArgumentException.class);
	}
}
