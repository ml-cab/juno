package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

class Phi3RopeLoadTest {

	private static boolean phiModelPresent() {
		return Path.of("models/Phi-3.5-mini-instruct-Q4_K_M.gguf").toFile().exists()
				|| Path.of("../models/Phi-3.5-mini-instruct-Q4_K_M.gguf").toFile().exists();
	}

	private static Path phiModelPath() {
		Path p = Path.of("models/Phi-3.5-mini-instruct-Q4_K_M.gguf");
		return p.toFile().exists() ? p : Path.of("../models/Phi-3.5-mini-instruct-Q4_K_M.gguf");
	}

	@Test
	@EnabledIf("phiModelPresent")
	void ropeFactors_haveExpectedLength() throws Exception {
		try (GgufReader r = GgufReader.open(phiModelPath())) {
			LlamaConfig cfg = LlamaConfig.from(r);
			Phi3RopeConfig rope = Phi3RopeConfig.from(r, cfg);
			assertThat(rope.ropeFactorsShort()).hasSize(cfg.headDim() / 2);
			assertThat(rope.ropeFactorsLong()).hasSize(cfg.headDim() / 2);
			assertThat(rope.attnFactor()).isBetween(1.18f, 1.20f);
			System.out.printf("short[0..2]=%.4f %.4f %.4f long[0..2]=%.4f %.4f %.4f%n",
					rope.ropeFactorsShort()[0], rope.ropeFactorsShort()[1], rope.ropeFactorsShort()[2],
					rope.ropeFactorsLong()[0], rope.ropeFactorsLong()[1], rope.ropeFactorsLong()[2]);
		}
	}
}
