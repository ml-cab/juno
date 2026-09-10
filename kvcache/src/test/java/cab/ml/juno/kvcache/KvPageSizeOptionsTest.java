package cab.ml.juno.kvcache;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("KvPageSizeOptions")
class KvPageSizeOptionsTest {

	private String prev;

	@AfterEach
	void restore() {
		if (prev == null)
			System.clearProperty(KvPageSizeOptions.ENV);
		else
			System.setProperty(KvPageSizeOptions.ENV, prev);
	}

	@Test
	@DisplayName("default page size is 16")
	void defaults() {
		prev = System.getProperty(KvPageSizeOptions.ENV);
		System.clearProperty(KvPageSizeOptions.ENV);
		assertThat(KvPageSizeOptions.fromEnv().pageSize()).isEqualTo(16);
	}

	@Test
	@DisplayName("parse positive page size from property")
	void parse_property() {
		prev = System.getProperty(KvPageSizeOptions.ENV);
		System.setProperty(KvPageSizeOptions.ENV, "64");
		assertThat(KvPageSizeOptions.fromEnv().pageSize()).isEqualTo(64);
	}

	@Test
	@DisplayName("reject non-positive")
	void reject_bad() {
		assertThatThrownBy(() -> KvPageSizeOptions.of(0))
				.isInstanceOf(IllegalArgumentException.class);
	}
}
