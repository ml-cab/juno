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
package cab.ml.juno.kvcache;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * 4-bit keys lose quality on some models and 4-bit values do not, so the warning follows the key type alone.
 */
class KvKeyPrecisionNoticeTest {

	private final List<LogRecord> records = new ArrayList<>();
	private final Handler capture = new Handler() {
		@Override
		public void publish(LogRecord record) {
			records.add(record);
		}

		@Override
		public void flush() {
		}

		@Override
		public void close() {
		}
	};
	private final Logger logger = Logger.getLogger(KvKeyPrecisionNotice.class.getName());

	@BeforeEach
	void setUp() {
		KvKeyPrecisionNotice.resetForTest();
		logger.addHandler(capture);
	}

	@AfterEach
	void tearDown() {
		logger.removeHandler(capture);
		KvKeyPrecisionNotice.resetForTest();
	}

	@Test
	void fourBitKeysAreWarnedAboutWhateverTheValues() {
		for (KvElementType v : KvElementType.values()) {
			String notice = KvKeyPrecisionNotice.notice(CacheTypeOptions.of(KvElementType.Q4_0, v));
			assertThat(notice).as("q4_0 keys with %s values", v).isNotNull()
					.contains("--cache-type-k q4_0")
					.contains("--cache-type-k q8_0 --cache-type-v q4_0");
		}
	}

	@Test
	void noWarningWithoutFourBitKeys() {
		for (KvElementType k : new KvElementType[] { KvElementType.F16, KvElementType.Q8_0 })
			for (KvElementType v : KvElementType.values())
				assertThat(KvKeyPrecisionNotice.notice(CacheTypeOptions.of(k, v))).as("%s keys, %s values", k, v)
						.isNull();
	}

	@Test
	void announceLogsOneWarningPerProcess() {
		KvKeyPrecisionNotice.announce(CacheTypeOptions.of(KvElementType.Q8_0, KvElementType.Q4_0));
		assertThat(records).isEmpty();
		KvKeyPrecisionNotice.announce(CacheTypeOptions.of(KvElementType.Q4_0, KvElementType.Q4_0));
		KvKeyPrecisionNotice.announce(CacheTypeOptions.of(KvElementType.Q4_0, KvElementType.Q4_0));
		assertThat(records).singleElement().satisfies(r -> {
			assertThat(r.getLevel()).isEqualTo(Level.WARNING);
			assertThat(r.getMessage()).isEqualTo(
					KvKeyPrecisionNotice.notice(CacheTypeOptions.of(KvElementType.Q4_0, KvElementType.Q4_0)));
		});
	}
}
