package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeFalse;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("LoraTrainDevice")
class LoraTrainDeviceTest {

	private String originalGpuFlag;

	@BeforeEach
	void saveFlag() {
		originalGpuFlag = System.getProperty("JUNO_USE_GPU");
	}

	@AfterEach
	void restoreFlag() {
		if (originalGpuFlag == null)
			System.clearProperty("JUNO_USE_GPU");
		else
			System.setProperty("JUNO_USE_GPU", originalGpuFlag);
	}

	@Test
	@DisplayName("normalize accepts auto|gpu|cpu")
	void normalize_ok() {
		assertThat(LoraTrainDevice.normalize("auto")).isEqualTo(LoraTrainDevice.AUTO);
		assertThat(LoraTrainDevice.normalize("GPU")).isEqualTo(LoraTrainDevice.GPU);
		assertThat(LoraTrainDevice.normalize(" Cpu ")).isEqualTo(LoraTrainDevice.CPU);
	}

	@Test
	@DisplayName("normalize rejects unknown modes")
	void normalize_rejects() {
		assertThatThrownBy(() -> LoraTrainDevice.normalize("tpu"))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("auto|gpu|cpu");
	}

	@Test
	@DisplayName("cpu mode always returns CpuMatVec")
	void cpu_forces_cpu() {
		System.clearProperty("JUNO_USE_GPU");
		assertThat(LoraTrainDevice.selectBackend(LoraTrainDevice.CPU)).isInstanceOf(CpuMatVec.class);
	}

	@Test
	@DisplayName("gpu mode fails closed when no CUDA/ROCm bindings")
	void gpu_fails_closed_without_device() {
		assumeFalse(CudaAvailability.isAvailable(), "CUDA present — skipping");
		assumeFalse(RocmAvailability.isAvailable(), "ROCm present — skipping");
		assertThatThrownBy(() -> LoraTrainDevice.selectBackend(LoraTrainDevice.GPU))
				.isInstanceOf(IllegalStateException.class)
				.hasMessageContaining("gpu")
				.hasMessageContaining("CUDA")
				.hasMessageContaining("ROCm");
	}

	@Test
	@DisplayName("auto with JUNO_USE_GPU=false returns CpuMatVec")
	void auto_respects_cpu_flag() {
		System.setProperty("JUNO_USE_GPU", "false");
		assertThat(LoraTrainDevice.selectBackend(LoraTrainDevice.AUTO)).isInstanceOf(CpuMatVec.class);
	}

	@Test
	@DisplayName("labelFor maps MatVec to cpu|cuda|rocm")
	void label_for_backend() {
		assertThat(LoraTrainDevice.labelFor(CpuMatVec.INSTANCE)).isEqualTo("cpu");
	}
}
