/*
 * Copyright 2026 Dmytro Soloviov (soulaway)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.logging.Logger;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("LoraMmqPolicy — playback-only MMQ gate")
class LoraMmqPolicyTest {

	private String originalMmq;
	private String originalPlayPath;
	private String originalMicrobatch;

	@BeforeEach
	void save() {
		originalMmq = System.getProperty(MmqOptions.ENV_PROPERTY);
		originalPlayPath = System.getProperty(LoraMmqPolicy.PLAY_PATH_PROPERTY);
		originalMicrobatch = System.getProperty(LoraMicrobatch.PROPERTY);
	}

	@AfterEach
	void restore() {
		restoreProp(MmqOptions.ENV_PROPERTY, originalMmq);
		restoreProp(LoraMmqPolicy.PLAY_PATH_PROPERTY, originalPlayPath);
		restoreProp(LoraMicrobatch.PROPERTY, originalMicrobatch);
		LoraMmqPolicy.resetWarnStateForTests();
	}

	private static void restoreProp(String key, String value) {
		if (value == null)
			System.clearProperty(key);
		else
			System.setProperty(key, value);
	}

	@Test
	@DisplayName("playback context is false unless play path is set")
	void playback_context() {
		System.clearProperty(LoraMmqPolicy.PLAY_PATH_PROPERTY);
		assertThat(LoraMmqPolicy.isPlaybackContext()).isFalse();
		System.setProperty(LoraMmqPolicy.PLAY_PATH_PROPERTY, "/tmp/x.lora");
		assertThat(LoraMmqPolicy.isPlaybackContext()).isTrue();
	}

	@Test
	@DisplayName("MMQ off never enables playback fused path")
	void mmq_off() {
		System.setProperty(MmqOptions.ENV_PROPERTY, "off");
		System.setProperty(LoraMmqPolicy.PLAY_PATH_PROPERTY, "/tmp/x.lora");
		assertThat(LoraMmqPolicy.enabledForPlayback(true)).isFalse();
	}

	@Test
	@DisplayName("play + mmq on + kernel support → enabled")
	void play_mmq_on() {
		System.setProperty(MmqOptions.ENV_PROPERTY, "on");
		System.setProperty(LoraMmqPolicy.PLAY_PATH_PROPERTY, "/tmp/x.lora");
		assertThat(LoraMmqPolicy.enabledForPlayback(true)).isTrue();
		assertThat(LoraMmqPolicy.enabledForPlayback(false)).isFalse();
	}

	@Test
	@DisplayName("train context (no play path) never enables MMQ even when on/auto")
	void train_ignores_mmq() {
		System.clearProperty(LoraMmqPolicy.PLAY_PATH_PROPERTY);
		System.setProperty(MmqOptions.ENV_PROPERTY, "on");
		assertThat(LoraMmqPolicy.enabledForPlayback(true)).isFalse();
		System.setProperty(MmqOptions.ENV_PROPERTY, "auto");
		assertThat(LoraMmqPolicy.enabledForPlayback(true)).isFalse();
	}

	@Test
	@DisplayName("play + mmq on + microbatch > 1 → disabled")
	void play_microbatch_blocks() {
		System.setProperty(MmqOptions.ENV_PROPERTY, "on");
		System.setProperty(LoraMmqPolicy.PLAY_PATH_PROPERTY, "/tmp/x.lora");
		System.setProperty(LoraMicrobatch.PROPERTY, "8");
		assertThat(LoraMmqPolicy.enabledForPlayback(true)).isFalse();
		System.setProperty(LoraMicrobatch.PROPERTY, "1");
		assertThat(LoraMmqPolicy.enabledForPlayback(true)).isTrue();
	}

	@Test
	@DisplayName("unset microbatch during play does not block (play never applies train microbatch)")
	void play_unset_microbatch_allows() {
		System.setProperty(MmqOptions.ENV_PROPERTY, "on");
		System.setProperty(LoraMmqPolicy.PLAY_PATH_PROPERTY, "/tmp/x.lora");
		System.clearProperty(LoraMicrobatch.PROPERTY);
		assertThat(LoraMmqPolicy.enabledForPlayback(true)).isTrue();
	}

	@Test
	@DisplayName("warnIfTrainIgnoresMmq logs once when train + preferMmq")
	void train_warn_once() {
		System.clearProperty(LoraMmqPolicy.PLAY_PATH_PROPERTY);
		System.setProperty(MmqOptions.ENV_PROPERTY, "on");
		Logger log = Logger.getLogger("LoraMmqPolicyTest.warn");
		assertThat(LoraMmqPolicy.warnIfTrainIgnoresMmq(log)).isTrue();
		assertThat(LoraMmqPolicy.warnIfTrainIgnoresMmq(log)).isFalse();
	}

	@Test
	@DisplayName("warnIfTrainIgnoresMmq is silent under playback or mmq off")
	void no_warn_play_or_off() {
		Logger log = Logger.getLogger("LoraMmqPolicyTest.nowarn");
		System.setProperty(MmqOptions.ENV_PROPERTY, "off");
		assertThat(LoraMmqPolicy.warnIfTrainIgnoresMmq(log)).isFalse();
		System.setProperty(MmqOptions.ENV_PROPERTY, "on");
		System.setProperty(LoraMmqPolicy.PLAY_PATH_PROPERTY, "/tmp/x.lora");
		assertThat(LoraMmqPolicy.warnIfTrainIgnoresMmq(log)).isFalse();
	}
}
