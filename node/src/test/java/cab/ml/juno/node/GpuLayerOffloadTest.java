package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("GpuLayerOffload — parse and residency")
class GpuLayerOffloadTest {

	private String original;

	@AfterEach
	void restore() {
		if (original == null)
			System.clearProperty(GpuLayerOffload.ENV_PROPERTY);
		else
			System.setProperty(GpuLayerOffload.ENV_PROPERTY, original);
	}

	@Test
	@DisplayName("parse all|auto|0|N")
	void parse_values() {
		assertThat(GpuLayerOffload.parse("all").mode()).isEqualTo(GpuLayerOffload.Mode.ALL);
		assertThat(GpuLayerOffload.parse("AUTO").mode()).isEqualTo(GpuLayerOffload.Mode.AUTO);
		assertThat(GpuLayerOffload.parse("0").resolvedCount(32)).isZero();
		assertThat(GpuLayerOffload.parse("16").resolvedCount(32)).isEqualTo(16);
		assertThat(GpuLayerOffload.parse("99").resolvedCount(32)).isEqualTo(32);
		assertThat(GpuLayerOffload.parse(null).mode()).isEqualTo(GpuLayerOffload.Mode.ALL);
	}

	@Test
	@DisplayName("invalid spec fails closed")
	void parse_invalid() {
		assertThatThrownBy(() -> GpuLayerOffload.parse("half"))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("all|auto|0|N");
	}

	@Test
	@DisplayName("shard-local global indices")
	void residency_for_shard() {
		GpuLayerOffload policy = GpuLayerOffload.count(3);
		int total = 8;
		// gpu-layers=3 → global indices 0,1,2 on GPU
		assertThat(policy.residentForGlobalLayer(1, total)).isTrue();
		assertThat(policy.residentForGlobalLayer(2, total)).isTrue();
		assertThat(policy.residentForGlobalLayer(3, total)).isFalse();
		assertThat(policy.residentForGlobalLayer(4, total)).isFalse();
		assertThat(policy.residentOutputProjection(total)).isFalse();
	}

	@Test
	@DisplayName("output projection only when all layers resident")
	void output_projection_gate() {
		assertThat(GpuLayerOffload.all().residentOutputProjection(32)).isTrue();
		assertThat(GpuLayerOffload.count(32).residentOutputProjection(32)).isTrue();
		assertThat(GpuLayerOffload.count(31).residentOutputProjection(32)).isFalse();
		assertThat(GpuLayerOffload.none().residentOutputProjection(32)).isFalse();
	}

	@Test
	@DisplayName("auto resolved count")
	void auto_resolved() {
		GpuLayerOffload auto = GpuLayerOffload.auto().withAutoResolved(12);
		assertThat(auto.resolvedCount(32)).isEqualTo(12);
		assertThat(auto.policyLabel(32)).isEqualTo("auto:12");
	}

	@Test
	@DisplayName("fromEnv reads JUNO_GPU_LAYERS")
	void from_env() {
		original = System.getProperty(GpuLayerOffload.ENV_PROPERTY);
		System.setProperty(GpuLayerOffload.ENV_PROPERTY, "5");
		assertThat(GpuLayerOffload.fromEnv().resolvedCount(32)).isEqualTo(5);
	}
}
