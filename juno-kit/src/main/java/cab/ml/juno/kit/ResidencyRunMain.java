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
package cab.ml.juno.kit;


import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import cab.ml.juno.player.JunoPlayer;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;

/**
 * One mode of {@link GpuResidencyProbe}, in a JVM of its own as each mode runs in its own server process in
 * the smoke script, with {@code -DJUNO_GPU_RESIDENCY} set: greedy requests in local mode (three in-process
 * nodes), the device memory this process holds after each, and the region's activation or decline notice.
 * Prints one {@code JUNO-KIT-RESIDENCY {json}} line. Arguments: {@code model requests nGen memNGen gpu}.
 */
public final class ResidencyRunMain {

	static final String PREFIX = "JUNO-KIT-RESIDENCY ";
	static final String PROMPT = "Write three sentences about the history of the printing press.";

	private ResidencyRunMain() {
	}

	public static void main(String[] args) throws Exception {
		String model = args[0];
		int requests = Integer.parseInt(args[1]);
		int nGen = Integer.parseInt(args[2]);
		int memNGen = Integer.parseInt(args[3]);
		boolean gpu = Boolean.parseBoolean(args[4]);
		List<String> texts = new ArrayList<>();
		List<Long> mem = new ArrayList<>();
		String notice;
		try (NoticeCapture log = NoticeCapture.install();
				JunoPlayer player = JunoPlayer.builder(java.nio.file.Path.of(model)).useGpu(gpu).build()) {
			for (int r = 1; r <= requests; r++) {
				int n = r == 1 || r == requests || memNGen <= 0 ? nGen : memNGen;
				SamplingParams p = SamplingParams.deterministic().withMaxTokens(n).withMinTokens(n);
				texts.add(player.scheduler().submitAndWait(cab.ml.juno.coordinator.InferenceRequest.of(player.modelId(),
						List.of(ChatMessage.user(PROMPT)), p, cab.ml.juno.coordinator.RequestPriority.NORMAL)).text());
				mem.add(GpuLoad.computeApps(GpuLoad.query(GpuLoad.COMPUTE_APPS),
						Set.of(ProcessHandle.current().pid())).ownMiB());
			}
			String active = log.first("GPU-resident decode region active");
			String declined = log.first("requested, but");
			notice = active != null ? active : declined != null ? declined : "";
		}
		Map<String, Object> out = new LinkedHashMap<>();
		out.put("texts", texts);
		out.put("memMiB", mem);
		out.put("notice", notice);
		System.out.println(PREFIX + Json.MAPPER.writer()
				.without(com.fasterxml.jackson.databind.SerializationFeature.INDENT_OUTPUT).writeValueAsString(out));
		System.out.flush();
		System.exit(0);
	}
}
