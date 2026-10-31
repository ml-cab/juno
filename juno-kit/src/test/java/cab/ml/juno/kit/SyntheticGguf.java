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

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Writes a tiny LLaMA-architecture GGUF (three layers, six heads, hidden size 96, F32 random weights, a SentencePiece
 * vocabulary with byte fallback) that Juno loads like a real model, so the probe self-tests run with no
 * model file. Its output is meaningless; the probes' mechanics are what the tests exercise.
 */
final class SyntheticGguf {

	// Three layers and six heads so three pipeline shards and a three-way tensor split divide evenly.
	static final int HIDDEN = 96;
	static final int LAYERS = 3;
	static final int HEADS = 6;
	static final int FFN = 192;
	static final int CONTEXT = 2048;

	private final ByteArrayOutputStream kv = new ByteArrayOutputStream();
	private int kvCount;
	private final List<String> names = new ArrayList<>();
	private final List<long[]> dims = new ArrayList<>();
	private final List<float[]> data = new ArrayList<>();

	private SyntheticGguf() {
	}

	/** Writes the model to {@code dir/tinyllama-synthetic.gguf} and returns its path. */
	static Path write(Path dir) throws IOException {
		List<String> tokens = new ArrayList<>(List.of("<unk>", "<s>", "</s>"));
		List<Integer> types = new ArrayList<>(List.of(2, 3, 3));
		for (int b = 0; b < 256; b++) {
			tokens.add(String.format(Locale.ROOT, "<0x%02X>", b));
			types.add(6);
		}
		for (String piece : List.of("▁", "▁x", "x", "▁Hi", "H", "i")) {
			tokens.add(piece);
			types.add(1);
		}
		int vocab = tokens.size();
		SyntheticGguf g = new SyntheticGguf();
		g.string("general.architecture", "llama");
		g.string("general.name", "tinyllama synthetic");
		g.u32("general.file_type", 0);
		g.u32("llama.context_length", CONTEXT);
		g.u32("llama.embedding_length", HIDDEN);
		g.u32("llama.block_count", LAYERS);
		g.u32("llama.feed_forward_length", FFN);
		g.u32("llama.attention.head_count", HEADS);
		g.u32("llama.attention.head_count_kv", HEADS);
		g.u32("llama.rope.dimension_count", HIDDEN / HEADS);
		g.u32("llama.vocab_size", vocab);
		g.f32("llama.attention.layer_norm_rms_epsilon", 1e-5f);
		g.f32("llama.rope.freq_base", 10000f);
		g.string("tokenizer.ggml.model", "llama");
		g.stringArray("tokenizer.ggml.tokens", tokens);
		float[] scores = new float[vocab];
		for (int i = 0; i < vocab; i++)
			scores[i] = -i;
		g.floatArray("tokenizer.ggml.scores", scores);
		g.intArray("tokenizer.ggml.token_type", types);
		g.u32("tokenizer.ggml.bos_token_id", 1);
		g.u32("tokenizer.ggml.eos_token_id", 2);
		g.u32("tokenizer.ggml.unknown_token_id", 0);

		Random rnd = new Random(7);
		g.tensor("token_embd.weight", rnd, HIDDEN, vocab);
		g.ones("output_norm.weight", HIDDEN);
		g.tensor("output.weight", rnd, HIDDEN, vocab);
		for (int l = 0; l < LAYERS; l++) {
			String p = "blk." + l + ".";
			g.ones(p + "attn_norm.weight", HIDDEN);
			g.tensor(p + "attn_q.weight", rnd, HIDDEN, HIDDEN);
			g.tensor(p + "attn_k.weight", rnd, HIDDEN, HIDDEN);
			g.tensor(p + "attn_v.weight", rnd, HIDDEN, HIDDEN);
			g.tensor(p + "attn_output.weight", rnd, HIDDEN, HIDDEN);
			g.ones(p + "ffn_norm.weight", HIDDEN);
			g.tensor(p + "ffn_gate.weight", rnd, HIDDEN, FFN);
			g.tensor(p + "ffn_up.weight", rnd, HIDDEN, FFN);
			g.tensor(p + "ffn_down.weight", rnd, FFN, HIDDEN);
		}
		Path out = dir.resolve("tinyllama-synthetic.gguf");
		Files.write(out, g.build());
		return out;
	}

	private void tensor(String name, Random rnd, long cols, long rows) {
		float[] v = new float[(int) (cols * rows)];
		for (int i = 0; i < v.length; i++)
			v[i] = (float) (rnd.nextGaussian() * 0.05);
		names.add(name);
		dims.add(new long[] { cols, rows });
		data.add(v);
	}

	private void ones(String name, int n) {
		float[] v = new float[n];
		java.util.Arrays.fill(v, 1f);
		names.add(name);
		dims.add(new long[] { n });
		data.add(v);
	}

	private void key(String k, int type) {
		long8(kv, k.getBytes(StandardCharsets.UTF_8).length);
		kv.writeBytes(k.getBytes(StandardCharsets.UTF_8));
		int4(kv, type);
		kvCount++;
	}

	private void string(String k, String v) {
		key(k, 8);
		str(kv, v);
	}

	private void u32(String k, int v) {
		key(k, 4);
		int4(kv, v);
	}

	private void f32(String k, float v) {
		key(k, 6);
		int4(kv, Float.floatToIntBits(v));
	}

	private void stringArray(String k, List<String> v) {
		key(k, 9);
		int4(kv, 8);
		long8(kv, v.size());
		for (String s : v)
			str(kv, s);
	}

	private void floatArray(String k, float[] v) {
		key(k, 9);
		int4(kv, 6);
		long8(kv, v.length);
		for (float f : v)
			int4(kv, Float.floatToIntBits(f));
	}

	private void intArray(String k, List<Integer> v) {
		key(k, 9);
		int4(kv, 5);
		long8(kv, v.size());
		for (int i : v)
			int4(kv, i);
	}

	private byte[] build() {
		final int align = 32;
		ByteArrayOutputStream out = new ByteArrayOutputStream();
		int4(out, 0x46554747);
		int4(out, 3);
		long8(out, names.size());
		long8(out, kvCount);
		out.writeBytes(kv.toByteArray());
		long offset = 0;
		List<Long> offsets = new ArrayList<>();
		for (int t = 0; t < names.size(); t++) {
			offsets.add(offset);
			offset += pad(data.get(t).length * 4L, align);
		}
		for (int t = 0; t < names.size(); t++) {
			str(out, names.get(t));
			int4(out, dims.get(t).length);
			for (long d : dims.get(t))
				long8(out, d);
			int4(out, 0);
			long8(out, offsets.get(t));
		}
		while (out.size() % align != 0)
			out.write(0);
		for (float[] v : data) {
			ByteBuffer bb = ByteBuffer.allocate((int) pad(v.length * 4L, align)).order(ByteOrder.LITTLE_ENDIAN);
			for (float f : v)
				bb.putFloat(f);
			out.writeBytes(bb.array());
		}
		return out.toByteArray();
	}

	private static long pad(long n, int align) {
		return (n + align - 1) / align * align;
	}

	private static void str(ByteArrayOutputStream o, String s) {
		byte[] b = s.getBytes(StandardCharsets.UTF_8);
		long8(o, b.length);
		o.writeBytes(b);
	}

	private static void int4(ByteArrayOutputStream o, int v) {
		o.writeBytes(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array());
	}

	private static void long8(ByteArrayOutputStream o, long v) {
		o.writeBytes(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(v).array());
	}
}
