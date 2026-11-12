package cab.ml.juno.kvcache;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class SharedPrefixOptionsTest {

	@Test
	void theDefaultIsAQuarterOfTheHeapAtMostOneContext() {
		SharedPrefixOptions d = SharedPrefixOptions.parse(null);
		assertThat(d.explicit()).isFalse();
		// 8 GiB heap, 256 KiB per position: a quarter holds 8192 positions
		assertThat(d.budget(32768, 8L << 30, 256L << 10)).isEqualTo(8192);
		assertThat(d.budget(4096, 8L << 30, 256L << 10)).isEqualTo(4096);
		assertThat(d.budget(4096, 8L << 30, 0)).as("size unknown: one context").isEqualTo(4096);
	}

	@Test
	void aGivenValueIsKeptAndZeroIsOff() {
		assertThat(SharedPrefixOptions.parse("0").budget(4096, 1L << 30, 1024)).isZero();
		assertThat(SharedPrefixOptions.parse(" 100000 ").budget(4096, 1L << 30, 1024)).isEqualTo(100000);
		assertThat(SharedPrefixOptions.parse("0").explicit()).isTrue();
	}

	@Test
	void anythingElseIsRefused() {
		assertThatThrownBy(() -> SharedPrefixOptions.parse("-1")).isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("--prefix-cache-tokens");
		assertThatThrownBy(() -> SharedPrefixOptions.parse("lots")).isInstanceOf(IllegalArgumentException.class);
	}
}
