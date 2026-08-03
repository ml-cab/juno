package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("LoraApiServer")
class LoraApiServerTest {

	@Test
	@DisplayName("parse accepts JSON array body for train-file-qa")
	void parse_body() {
		List<LoraQaFile.Pair> pairs = LoraQaFile.parse("""
				[{"Q":"What is my name?","A":"Dima"}]
				""");
		assertThat(pairs).containsExactly(new LoraQaFile.Pair("What is my name?", "Dima"));
	}

	@Test
	@DisplayName("parse rejects empty body")
	void parse_empty() {
		assertThatThrownBy(() -> LoraQaFile.parse(" "))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("empty");
	}

	@Test
	@DisplayName("trainResponse maps TrainingResult fields")
	void train_response() {
		var result = new LoraTrainingLoop.TrainingResult(1.25f, Float.NaN, Float.NaN, 7, -1, 7,
				LoraTrainingLoop.StopReason.TARGET_REACHED, true, null);
		Map<String, Object> json = LoraApiServer.trainResponse(2, result);
		assertThat(json.get("pairCount")).isEqualTo(2);
		assertThat(json.get("unitCount")).isEqualTo(8);
		assertThat(json.get("finalTrainLoss")).isEqualTo(1.25f);
		assertThat(json.get("passCount")).isEqualTo(7);
		assertThat(json.get("optimizerUpdateCount")).isEqualTo(7);
		assertThat(json.get("stopReason")).isEqualTo("TARGET_REACHED");
		assertThat(json.get("targetReached")).isEqualTo(true);
	}
}
