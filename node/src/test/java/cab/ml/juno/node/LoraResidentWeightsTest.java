package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("LoraResidentWeights")
class LoraResidentWeightsTest {

	private String originalDevice;

	@BeforeEach
	void saveDevice() {
		originalDevice = System.getProperty("juno.lora.train.device");
	}

	@AfterEach
	void restoreDevice() {
		if (originalDevice == null)
			System.clearProperty("juno.lora.train.device");
		else
			System.setProperty("juno.lora.train.device", originalDevice);
	}

	@Test
	@DisplayName("closeArray is null-safe and closes non-null entries")
	void closeArray_nullSafe() {
		LoraResidentWeights.closeArray(null);
		LoraResidentWeights.closeArray(new ResidentWeightMatrix[0]);
		LoraResidentWeights.closeArray(new ResidentWeightMatrix[] { null });
	}

	@Test
	@DisplayName("matVec / transposedMatVec fall back to CPU when resident is null")
	void cpuFallback_whenNoResident() {
		// 2×3 F32 row-major
		float[] A = { 1f, 2f, 3f, 4f, 5f, 6f };
		float[] x = { 1f, 0f, -1f };
		float[] y = LoraResidentWeights.matVecDense(A, null, x, 2, 3);
		assertThat(y).hasSize(2);
		assertThat(y[0]).isCloseTo(-2f, within(1e-5f));
		assertThat(y[1]).isCloseTo(-2f, within(1e-5f));

		float[] g = { 1f, 2f };
		float[] xT = LoraResidentWeights.transposedMatVecDense(A, null, g, 2, 3);
		assertThat(xT).hasSize(3);
		assertThat(xT[0]).isCloseTo(9f, within(1e-5f));
		assertThat(xT[1]).isCloseTo(12f, within(1e-5f));
		assertThat(xT[2]).isCloseTo(15f, within(1e-5f));
	}

	@Test
	@DisplayName("rowMajorSlice extracts contiguous row blocks")
	void rowMajorSlice() {
		float[] full = { 1f, 2f, 3f, 4f, 5f, 6f };
		assertThat(LoraResidentWeights.rowMajorSlice(full, 1, 1, 3)).containsExactly(4f, 5f, 6f);
	}

	@Test
	@DisplayName("isVramOom detects cudaMalloc / hipMalloc messages")
	void isVramOom() {
		assertThat(LoraResidentWeights.isVramOom(new IllegalStateException("cudaMalloc failed"))).isTrue();
		assertThat(LoraResidentWeights.isVramOom(new IllegalStateException("hipMalloc OOM"))).isTrue();
		assertThat(LoraResidentWeights.isVramOom(new IllegalStateException("other"))).isFalse();
		assertThat(LoraResidentWeights.isVramOom(new IllegalStateException())).isFalse();
	}

	@Test
	@DisplayName("tryRecoverFromUploadOom falls back under auto and runs closer")
	void tryRecover_autoFallback() {
		System.setProperty("juno.lora.train.device", LoraTrainDevice.AUTO);
		AtomicBoolean closed = new AtomicBoolean(false);
		boolean ok = LoraResidentWeights.tryRecoverFromUploadOom(
				new IllegalStateException("cudaMalloc: out of memory"),
				Logger.getLogger("test"),
				() -> closed.set(true));
		assertThat(ok).isTrue();
		assertThat(closed).isTrue();
	}

	@Test
	@DisplayName("tryRecoverFromUploadOom fails closed under gpu mode")
	void tryRecover_gpuFailsClosed() {
		System.setProperty("juno.lora.train.device", LoraTrainDevice.GPU);
		AtomicBoolean closed = new AtomicBoolean(false);
		assertThatThrownBy(() -> LoraResidentWeights.tryRecoverFromUploadOom(
				new IllegalStateException("hipMalloc failed"),
				Logger.getLogger("test"),
				() -> closed.set(true)))
				.isInstanceOf(IllegalStateException.class)
				.hasMessageContaining("--lora-train-device=gpu")
				.hasMessageContaining("VRAM");
		assertThat(closed).isTrue();
	}

	@Test
	@DisplayName("tryRecoverFromUploadOom rethrows non-VRAM errors after close")
	void tryRecover_rethrowsOther() {
		System.setProperty("juno.lora.train.device", LoraTrainDevice.AUTO);
		AtomicBoolean closed = new AtomicBoolean(false);
		assertThatThrownBy(() -> LoraResidentWeights.tryRecoverFromUploadOom(
				new IllegalStateException("kernel launch failed"),
				Logger.getLogger("test"),
				() -> closed.set(true)))
				.isInstanceOf(IllegalStateException.class)
				.hasMessageContaining("kernel launch");
		assertThat(closed).isTrue();
	}
}
