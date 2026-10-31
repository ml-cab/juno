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


import java.util.List;

/**
 * The models the kit can fetch: ungated Hugging Face files pinned to a commit and a SHA-256. The hashes are
 * those of the reference host's files, matched against the Hub's published LFS hashes, so a tester's
 * download is byte-identical to the file the reference run used.
 *
 * @param heap the probe JVM's heap for this model, as the throughput harness fixes it
 */
record CatalogModel(String id, String title, String repo, String revision, String file, long bytes, String sha256,
		String heap) {

	static final List<CatalogModel> ALL = List.of(
			new CatalogModel("tinyllama-1.1b", "TinyLlama 1.1B Chat Q4_K_M", "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
					"52e7645ba7c309695bec7ac98f4f005b139cf465", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", 668_788_096L,
					"9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0", "4g"),
			new CatalogModel("qwen2.5-3b", "Qwen2.5-3B Instruct Q4_K_M", "Qwen/Qwen2.5-3B-Instruct-GGUF",
					"7dabda4d13d513e3e842b20f0d435c732f172cbe", "qwen2.5-3b-instruct-q4_k_m.gguf", 2_104_932_768L,
					"626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d", "5g"),
			new CatalogModel("phi-3.5-mini", "Phi-3.5-mini Instruct Q4_K_M", "bartowski/Phi-3.5-mini-instruct-GGUF",
					"6d70da17e749a471ccb62ade694486011a75cda3", "Phi-3.5-mini-instruct-Q4_K_M.gguf", 2_393_232_672L,
					"e4165e3a71af97f1b4820da61079826d8752a2088e313af0c7d346796c38eff5", "6g"),
			new CatalogModel("mistral-7b", "Mistral 7B Instruct v0.1 Q4_K_M",
					"itlwas/Mistral-7B-Instruct-v0.1-Q4_K_M-GGUF", "1a037a56dcd9c6ad39ff8459dec3f47f287dcaf4",
					"mistral-7b-instruct-v0.1-q4_k_m.gguf", 4_368_440_864L,
					"1af6bf966c4113f0cb5de7af42f38572075c9798bc16e6c72faee051bd7d6d33", "9g"),
			new CatalogModel("qwen3-1.7b", "Qwen3-1.7B Q4_K_M", "unsloth/Qwen3-1.7B-GGUF",
					"d7f544eead698dbd1f15126ef60b45a1e1933222", "Qwen3-1.7B-Q4_K_M.gguf", 1_107_409_472L,
					"b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897", "4g"),
			new CatalogModel("moondream2", "moondream2 Q5_K (vision, llamafile)", "cjpais/moondream2-llamafile",
					"8abde7e79de0c3bbbfeb29827ef594f36bc5fe67", "moondream2-q5_k.llamafile", 1_977_619_136L,
					"df6cf6bc9799946435704dafdf925c65295af5aa4e48740c1bcff57856607564", "4g"));

	static CatalogModel byId(String id) {
		for (CatalogModel m : ALL)
			if (m.id().equals(id))
				return m;
		throw new IllegalArgumentException("no catalog model " + id);
	}
}
