package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * GpuContext lifecycle tests.
 *
 * All tests are @Tag("gpu") — they require a CUDA device and are skipped on
 * CPU-only machines via assumeTrue().
 *
 * Run on AWS: mvn test -Dgroups=gpu -pl node
 */
@Tag("gpu")
@DisplayName("GpuContext — cuBLAS handle lifecycle")
class GpuContextTest {

	private static void assumeCuda() {
		assumeTrue(CudaAvailability.isAvailable(), "No CUDA device — skipping");
	}

	@Test
	@DisplayName("init(0) returns an open context with a valid handle")
	void init_returns_open_context() {
		assumeCuda();
		try (GpuContext ctx = GpuContext.init(0)) {
			assertThat(ctx.isClosed()).isFalse();
			assertThat(ctx.handle()).isNotNull();
			assertThat(ctx.deviceIndex()).isEqualTo(0);
		}
	}

	@Test
	@DisplayName("close() marks context as closed")
	void close_marks_closed() {
		assumeCuda();
		GpuContext ctx = GpuContext.init(0);
		assertThat(ctx.isClosed()).isFalse();
		ctx.close();
		assertThat(ctx.isClosed()).isTrue();
	}

	@Test
	@DisplayName("handle() after close() throws IllegalStateException")
	void handle_after_close_throws() {
		assumeCuda();
		GpuContext ctx = GpuContext.init(0);
		ctx.close();
		assertThatThrownBy(ctx::handle).isInstanceOf(IllegalStateException.class).hasMessageContaining("closed");
	}

	@Test
	@DisplayName("close() is idempotent — second call does not throw")
	void double_close_is_safe() {
		assumeCuda();
		GpuContext ctx = GpuContext.init(0);
		ctx.close();
		ctx.close(); // must not throw
	}

	@Test
	@DisplayName("init() without CUDA throws IllegalStateException")
	void init_without_cuda_throws() {
		assumeTrue(!CudaAvailability.isAvailable(), "Skipping — CUDA is present");
		assertThatThrownBy(() -> GpuContext.init(0)).isInstanceOf(IllegalStateException.class)
				.hasMessageContaining("CUDA not available");
	}

	@Test
	@DisplayName("try-with-resources closes context automatically")
	void try_with_resources_auto_closes() {
		assumeCuda();
		GpuContext ref;
		try (GpuContext ctx = GpuContext.init(0)) {
			ref = ctx;
			assertThat(ctx.isClosed()).isFalse();
			ref.close();
		}
		assertThat(ref.isClosed()).isTrue();
	}

	@Test
	@DisplayName("shared(0) returns singleton; close() does not destroy handle")
	void shared_singleton_close_is_noop() {
		assumeCuda();
		GpuContext a = GpuContext.shared(0);
		GpuContext b = GpuContext.shared(0);
		assertThat(a).isSameAs(b);
		assertThat(a.isProcessShared()).isTrue();
		a.close();
		assertThat(a.isClosed()).isFalse();
		assertThat(a.handle()).isNotNull();
	}
}