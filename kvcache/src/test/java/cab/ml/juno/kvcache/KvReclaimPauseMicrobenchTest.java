package cab.ml.juno.kvcache;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("KvReclaimPauseMicrobench")
class KvReclaimPauseMicrobenchTest {

	@Test
	@DisplayName("a small cell releases every finished request's page and reports the pass and step times")
	void smallCell() throws Exception {
		KvReclaimPauseMicrobench.Cell c = KvReclaimPauseMicrobench.run(new KvReclaimPauseMicrobench.Model("tiny", 2),
				64, 2, 4, 8, 2, 8);
		// batch x ctx / page x layers x (K, V)
		assertThat(c.idlePages()).isEqualTo(2L * (64 / 4) * 2 * 2);
		assertThat(c.passMaxMs()).isGreaterThanOrEqualTo(c.passMedianMs()).isGreaterThanOrEqualTo(0);
		assertThat(c.holds()).isEqualTo(2);
		assertThat(c.stepMaxMs()).isGreaterThanOrEqualTo(c.stepP50Ms());
		assertThat(c.stepMaxContendedMs()).isGreaterThanOrEqualTo(c.stepP50ContendedMs());
		String report = KvReclaimPauseMicrobench.formatReport(List.of(c));
		assertThat(report).contains("| tiny | 2 | 64 | 2 | 4 | 8 | 128 |").contains("pass_ms max");
	}
}
