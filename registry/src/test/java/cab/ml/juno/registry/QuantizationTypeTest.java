package cab.ml.juno.registry;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

import cab.ml.juno.registry.QuantizationType;

class QuantizationTypeTest {

	@Test
	void fp16_uses_two_bytes_per_param() {
		assertThat(QuantizationType.FP16.bytesPerParam()).isEqualTo(2.0);
	}

	@Test
	void int8_uses_one_byte_per_param() {
		assertThat(QuantizationType.INT8.bytesPerParam()).isEqualTo(1.0);
	}

	@Test
	void q4_k_m_uses_half_byte_per_param() {
		assertThat(QuantizationType.Q4_K_M.bytesPerParam()).isEqualTo(0.5);
	}

	@Test
	void quantized_types_use_fewer_bytes_than_fp16() {
		double fp16 = QuantizationType.FP16.bytesPerParam();
		for (QuantizationType q : QuantizationType.values()) {
			if (q != QuantizationType.FP32 && q != QuantizationType.FP16 && q != QuantizationType.BF16) {
				assertThat(q.bytesPerParam()).as("%s should use fewer bytes than FP16", q).isLessThan(fp16);
			}
		}
	}

	@Test
	void all_types_have_positive_bytes_per_param() {
		for (QuantizationType q : QuantizationType.values()) {
			assertThat(q.bytesPerParam()).as("%s bytesPerParam should be > 0", q).isGreaterThan(0.0);
		}
	}

	@Test
	void display_name_is_not_blank() {
		for (QuantizationType q : QuantizationType.values()) {
			assertThat(q.displayName()).isNotBlank();
		}
	}
}