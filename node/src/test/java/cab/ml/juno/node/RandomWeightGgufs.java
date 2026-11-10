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

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

/**
 * Synthetic GGUF files with random F32 weights (Gaussian, standard deviation 0.05; norm
 * weights 1), for tests whose oracle needs logits that depend on what attention reads.
 * The zero-weight builders other tests use give the same logits whatever the KV holds.
 * Tensor names and shapes follow those builders.
 */
final class RandomWeightGgufs {

	static final int H = 256;
	static final int HEADS = 8;
	static final int HEAD_DIM = H / HEADS;
	static final int I = 256;
	static final int VOCAB = 256;

	private RandomWeightGgufs() {
	}

	/** {@code llama}: separate Q, K, V; 4 KV heads (grouping 2); SwiGLU feed-forward. */
	static Path llama(Path dir, int layers) throws IOException {
		int kvDim = 4 * HEAD_DIM;
		Phi3TransformerHandlerTest.GgufAssembler gguf = new Phi3TransformerHandlerTest.GgufAssembler();
		gguf.addString("general.architecture", "llama");
		gguf.addUInt32("llama.embedding_length", H);
		gguf.addUInt32("llama.block_count", layers);
		gguf.addUInt32("llama.attention.head_count", HEADS);
		gguf.addUInt32("llama.attention.head_count_kv", 4);
		gguf.addUInt32("llama.vocab_size", VOCAB);
		gguf.addUInt32("llama.feed_forward_length", I);
		gguf.addFloat32("llama.attention.layer_norm_rms_epsilon", 1e-5f);
		gguf.addFloat32("llama.rope.freq_base", 10000.0f);
		globals(gguf);
		for (int li = 0; li < layers; li++) {
			String p = "blk." + li + ".";
			gguf.addTensor(p + "attn_norm.weight", 0, new long[] { H }, ones(H));
			gguf.addTensor(p + "ffn_norm.weight", 0, new long[] { H }, ones(H));
			gguf.addTensor(p + "attn_q.weight", 0, new long[] { H, H }, gaussian((long) H * H, 100L + li));
			gguf.addTensor(p + "attn_k.weight", 0, new long[] { kvDim, H }, gaussian((long) kvDim * H, 110L + li));
			gguf.addTensor(p + "attn_v.weight", 0, new long[] { kvDim, H }, gaussian((long) kvDim * H, 120L + li));
			gguf.addTensor(p + "attn_output.weight", 0, new long[] { H, H }, gaussian((long) H * H, 200L + li));
			gguf.addTensor(p + "ffn_gate.weight", 0, new long[] { I, H }, gaussian((long) I * H, 300L + li));
			gguf.addTensor(p + "ffn_up.weight", 0, new long[] { I, H }, gaussian((long) I * H, 310L + li));
			gguf.addTensor(p + "ffn_down.weight", 0, new long[] { H, I }, gaussian((long) H * I, 320L + li));
		}
		return write(dir, "random_llama.gguf", gguf);
	}

	/** {@code phi3}: fused QKV, gate and up fused into one {@code [2I, H]} tensor; no grouping. */
	static Path phi3(Path dir, int layers) throws IOException {
		int kvDim = HEADS * HEAD_DIM;
		Phi3TransformerHandlerTest.GgufAssembler gguf = new Phi3TransformerHandlerTest.GgufAssembler();
		gguf.addString("general.architecture", "phi3");
		gguf.addUInt32("phi3.embedding_length", H);
		gguf.addUInt32("phi3.block_count", layers);
		gguf.addUInt32("phi3.attention.head_count", HEADS);
		gguf.addUInt32("phi3.attention.head_count_kv", HEADS);
		gguf.addUInt32("phi3.vocab_size", VOCAB);
		gguf.addUInt32("phi3.feed_forward_length", I);
		gguf.addFloat32("phi3.attention.layer_norm_rms_epsilon", 1e-5f);
		gguf.addFloat32("phi3.rope.freq_base", 10000.0f);
		globals(gguf);
		for (int li = 0; li < layers; li++) {
			String p = "blk." + li + ".";
			long qkvRows = H + 2L * kvDim;
			gguf.addTensor(p + "attn_norm.weight", 0, new long[] { H }, ones(H));
			gguf.addTensor(p + "ffn_norm.weight", 0, new long[] { H }, ones(H));
			gguf.addTensor(p + "attn_qkv.weight", 0, new long[] { qkvRows, H }, gaussian(qkvRows * H, 100L + li));
			gguf.addTensor(p + "attn_output.weight", 0, new long[] { H, H }, gaussian((long) H * H, 200L + li));
			gguf.addTensor(p + "ffn_up.weight", 0, new long[] { 2L * I, H }, gaussian(2L * I * H, 300L + li));
			gguf.addTensor(p + "ffn_down.weight", 0, new long[] { H, I }, gaussian((long) H * I, 400L + li));
		}
		return write(dir, "random_phi3.gguf", gguf);
	}

	/** {@code phi2}: fused QKV, parallel attention and feed-forward off one layer norm; no grouping. */
	static Path phi2(Path dir, int layers) throws IOException {
		int kvDim = HEADS * HEAD_DIM;
		Phi3TransformerHandlerTest.GgufAssembler gguf = new Phi3TransformerHandlerTest.GgufAssembler();
		gguf.addString("general.architecture", "phi2");
		gguf.addUInt32("phi2.embedding_length", H);
		gguf.addUInt32("phi2.block_count", layers);
		gguf.addUInt32("phi2.attention.head_count", HEADS);
		gguf.addUInt32("phi2.attention.head_count_kv", HEADS);
		gguf.addUInt32("phi2.vocab_size", VOCAB);
		gguf.addUInt32("phi2.feed_forward_length", I);
		gguf.addFloat32("phi2.attention.layer_norm_epsilon", 1e-5f);
		gguf.addFloat32("phi2.rope.freq_base", 10000.0f);
		gguf.addUInt32("phi2.rope.dimension_count", HEAD_DIM);
		globals(gguf);
		for (int li = 0; li < layers; li++) {
			String p = "blk." + li + ".";
			long qkvRows = H + 2L * kvDim;
			gguf.addTensor(p + "attn_norm.weight", 0, new long[] { H }, ones(H));
			gguf.addTensor(p + "attn_qkv.weight", 0, new long[] { qkvRows, H }, gaussian(qkvRows * H, 100L + li));
			gguf.addTensor(p + "attn_output.weight", 0, new long[] { H, H }, gaussian((long) H * H, 200L + li));
			gguf.addTensor(p + "ffn_up.weight", 0, new long[] { I, H }, gaussian((long) I * H, 300L + li));
			gguf.addTensor(p + "ffn_down.weight", 0, new long[] { H, I }, gaussian((long) H * I, 400L + li));
		}
		return write(dir, "random_phi2.gguf", gguf);
	}

	/** {@code qwen3moe}: 4 KV heads (grouping 2), Q/K norms, 4 experts with 2 used per token. */
	static Path qwen3moe(Path dir, int layers) throws IOException {
		int kvHeads = 4;
		int kvDim = kvHeads * HEAD_DIM;
		int experts = 4;
		int expertFf = 128;
		Phi3TransformerHandlerTest.GgufAssembler gguf = new Phi3TransformerHandlerTest.GgufAssembler();
		gguf.addString("general.architecture", "qwen3moe");
		gguf.addUInt32("qwen3moe.embedding_length", H);
		gguf.addUInt32("qwen3moe.block_count", layers);
		gguf.addUInt32("qwen3moe.attention.head_count", HEADS);
		gguf.addUInt32("qwen3moe.attention.head_count_kv", kvHeads);
		gguf.addUInt32("qwen3moe.attention.key_length", HEAD_DIM);
		gguf.addUInt32("qwen3moe.vocab_size", VOCAB);
		gguf.addUInt32("qwen3moe.feed_forward_length", expertFf);
		gguf.addUInt32("qwen3moe.expert_count", experts);
		gguf.addUInt32("qwen3moe.expert_used_count", 2);
		gguf.addUInt32("qwen3moe.expert_feed_forward_length", expertFf);
		gguf.addFloat32("qwen3moe.attention.layer_norm_rms_epsilon", 1e-5f);
		gguf.addFloat32("qwen3moe.rope.freq_base", 1000000.0f);
		globals(gguf);
		for (int li = 0; li < layers; li++) {
			String p = "blk." + li + ".";
			gguf.addTensor(p + "attn_norm.weight", 0, new long[] { H }, ones(H));
			gguf.addTensor(p + "ffn_norm.weight", 0, new long[] { H }, ones(H));
			gguf.addTensor(p + "attn_q_norm.weight", 0, new long[] { HEAD_DIM }, ones(HEAD_DIM));
			gguf.addTensor(p + "attn_k_norm.weight", 0, new long[] { HEAD_DIM }, ones(HEAD_DIM));
			gguf.addTensor(p + "attn_q.weight", 0, new long[] { H, H }, gaussian((long) H * H, 100L + li));
			gguf.addTensor(p + "attn_k.weight", 0, new long[] { kvDim, H }, gaussian((long) kvDim * H, 110L + li));
			gguf.addTensor(p + "attn_v.weight", 0, new long[] { kvDim, H }, gaussian((long) kvDim * H, 120L + li));
			gguf.addTensor(p + "attn_output.weight", 0, new long[] { H, H }, gaussian((long) H * H, 200L + li));
			gguf.addTensor(p + "ffn_gate_inp.weight", 0, new long[] { experts, H },
					gaussian((long) experts * H, 300L + li));
			gguf.addTensor(p + "ffn_gate_exps.weight", 0, new long[] { experts, expertFf, H },
					gaussian((long) experts * expertFf * H, 310L + li));
			gguf.addTensor(p + "ffn_up_exps.weight", 0, new long[] { experts, expertFf, H },
					gaussian((long) experts * expertFf * H, 320L + li));
			gguf.addTensor(p + "ffn_down_exps.weight", 0, new long[] { experts, H, expertFf },
					gaussian((long) experts * H * expertFf, 330L + li));
		}
		return write(dir, "random_qwen3moe.gguf", gguf);
	}

	private static void globals(Phi3TransformerHandlerTest.GgufAssembler gguf) {
		gguf.addTensor("token_embd.weight", 0, new long[] { VOCAB, H }, gaussian((long) VOCAB * H, 7));
		gguf.addTensor("output_norm.weight", 0, new long[] { H }, ones(H));
		gguf.addTensor("output.weight", 0, new long[] { VOCAB, H }, gaussian((long) VOCAB * H, 11));
	}

	private static Path write(Path dir, String name, Phi3TransformerHandlerTest.GgufAssembler gguf)
			throws IOException {
		Files.createDirectories(dir);
		Path out = dir.resolve(name);
		Files.write(out, gguf.build());
		return out;
	}

	private static byte[] ones(int n) {
		ByteBuffer bb = ByteBuffer.allocate(n * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
		for (int i = 0; i < n; i++)
			bb.putFloat(1f);
		return bb.array();
	}

	private static byte[] gaussian(long n, long seed) {
		Random r = new Random(seed);
		ByteBuffer bb = ByteBuffer.allocate(Math.toIntExact(n * Float.BYTES)).order(ByteOrder.LITTLE_ENDIAN);
		for (long i = 0; i < n; i++)
			bb.putFloat((float) (r.nextGaussian() * 0.05));
		return bb.array();
	}
}
