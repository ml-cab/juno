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

import java.io.IOException;
import java.io.InputStream;
import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeElement;
import java.lang.classfile.CodeModel;
import java.lang.classfile.Instruction;
import java.lang.classfile.Label;
import java.lang.classfile.MethodModel;
import java.lang.classfile.Opcode;
import java.lang.classfile.instruction.ArrayLoadInstruction;
import java.lang.classfile.instruction.ArrayStoreInstruction;
import java.lang.classfile.instruction.ConstantInstruction;
import java.lang.classfile.instruction.ConvertInstruction;
import java.lang.classfile.instruction.FieldInstruction;
import java.lang.classfile.instruction.IncrementInstruction;
import java.lang.classfile.instruction.InvokeDynamicInstruction;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.classfile.instruction.LabelTarget;
import java.lang.classfile.instruction.LoadInstruction;
import java.lang.classfile.instruction.LocalVariable;
import java.lang.classfile.instruction.OperatorInstruction;
import java.lang.classfile.instruction.StoreInstruction;
import java.lang.classfile.instruction.TypeCheckInstruction;
import java.lang.management.ManagementFactory;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.api.Test;

/**
 * The CPU integer and attention kernels keep every vector inside the method that computes it.
 *
 * <p>C2 boxes a vector passed to, or returned from, a method it does not inline, and the boxed value then takes the
 * Vector API's generic path, which allocates on every operation. Whether a helper is inlined depends on the size of its
 * caller and on the order in which methods were compiled, so a kernel that allocates nothing in a small test can
 * allocate tens of gigabytes per request in a server: the prefill matmul's one-row remainder loop did, through a
 * {@code widenPairs(ShortVector)} helper, once its four-row loop made the tile method large. The first check forbids
 * the shape outright: no production kernel class declares a method with a Vector API parameter or return type. The
 * second measures the matmul with remainder rows directly. The third forbids, in the batched matmul, a Vector API
 * call on a vector carried from one loop iteration to the next.
 */
class CpuKernelVectorEscapeTest {

	/** The production CPU kernels: the decode and prefill integer kernels at both widths and the attention head. */
	private static final Class<?>[] KERNELS = { CpuIntegerKernels.class, CpuIntegerGemm.class,
			CpuIntegerKernels256.class, CpuIntegerGemm256.class, GqaVectorHead.class };

	@Test
	void noKernelMethodTakesOrReturnsAVector() {
		List<String> escapes = new ArrayList<>();
		for (Class<?> k : KERNELS)
			for (Method m : k.getDeclaredMethods()) {
				if (isVector(m.getReturnType()))
					escapes.add(k.getSimpleName() + "." + m.getName() + " returns " + m.getReturnType().getSimpleName());
				for (Class<?> p : m.getParameterTypes())
					if (isVector(p))
						escapes.add(k.getSimpleName() + "." + m.getName() + " takes " + p.getSimpleName());
			}
		assertThat(escapes).as("methods through which a vector would cross a call").isEmpty();
	}

	@ParameterizedTest(name = "type {0}, B={1}")
	@CsvSource({ "12, 1", "12, 5", "12, 7", "14, 1", "14, 5", "14, 7", "13, 3", "8, 6" })
	void matmulWithRemainderRowsAllocatesNothingPerRow(int type, int batch) {
		int rows = 256, cols = 1024;
		Random rnd = new Random(7L * type + batch);
		GgufReader.QuantizedTensor w = CpuIntegerKernelsTest.synthetic(type, rows, cols, rnd);
		float[][] x = new float[batch][cols];
		for (float[] r : x)
			for (int i = 0; i < cols; i++)
				r[i] = (float) rnd.nextGaussian();
		float[][] y = new float[batch][rows];
		for (int i = 0; i < 400; i++) // compile the tile before measuring
			CpuIntegerGemm.sgemmInto(w, x, y, 0, rows, cols);
		com.sun.management.ThreadMXBean mx = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
		int calls = 50;
		long before = allThreads(mx);
		for (int i = 0; i < calls; i++)
			CpuIntegerGemm.sgemmInto(w, x, y, 0, rows, cols);
		long perCall = (allThreads(mx) - before) / calls;
		// One boxed vector per sub-block of one row is already 256 rows x 4 blocks x 8 sub-blocks x 16 bytes = 128 KB
		// per call; task bookkeeping is a few hundred bytes per task.
		assertThat(perCall).as("bytes allocated per call, all threads").isLessThan(16 * 1024);
	}

	/**
	 * The batched matmul calls no Vector API method on a vector carried from one loop iteration to the next. While
	 * C2 parses a loop it types such a value only as {@code FloatVector}, so a method called on it is bound to
	 * {@code Float128Vector} through the JDK's shared type profile, behind a guard; in some server JVMs C2 then boxed
	 * the accumulators and the matmul allocated on every operation, 2x to 5.5x slower. A carried vector may only be an
	 * argument. Found from the bytecode: the instruction that pushed each Vector API call's receiver.
	 */
	@Test
	void batchedMatmulCallsNothingOnALoopCarriedVector() throws IOException {
		byte[] bytes;
		try (InputStream in = CpuIntegerGemm.class.getResourceAsStream("CpuIntegerGemm.class")) {
			bytes = in.readAllBytes();
		}
		assertThat(vectorLocals(bytes)).as("vector locals in CpuIntegerGemm's local variable tables (needs javac -g)")
				.isPositive();
		assertThat(carriedReceivers(bytes)).as("Vector API calls on a loop-carried vector in CpuIntegerGemm").isEmpty();
	}

	/**
	 * The same rule for the attention head, whose dot products and weighted value sums carry their accumulators
	 * across every key and, reading paged KV in place, across every page: a carried vector is only an argument.
	 */
	@Test
	void attentionHeadCallsNothingOnALoopCarriedVector() throws IOException {
		byte[] bytes;
		try (InputStream in = GqaVectorHead.class.getResourceAsStream("GqaVectorHead.class")) {
			bytes = in.readAllBytes();
		}
		assertThat(vectorLocals(bytes)).as("vector locals in GqaVectorHead's local variable tables (needs javac -g)")
				.isPositive();
		assertThat(carriedReceivers(bytes)).as("Vector API calls on a loop-carried vector in GqaVectorHead").isEmpty();
	}

	/** Vector-typed entries in the class's local variable tables; zero when it was compiled without them. */
	static int vectorLocals(byte[] classBytes) {
		int n = 0;
		for (MethodModel m : ClassFile.of().parse(classBytes).methods())
			for (CodeElement e : m.code().map(CodeModel::elementList).orElse(List.of()))
				if (e instanceof LocalVariable lv
						&& lv.typeSymbol().descriptorString().startsWith("Ljdk/incubator/vector/"))
					n++;
		return n;
	}

	/** {@code method: name on variable} for every Vector API call whose receiver is a reassigned vector local. */
	static List<String> carriedReceivers(byte[] classBytes) {
		List<String> found = new ArrayList<>();
		for (MethodModel m : ClassFile.of().parse(classBytes).methods()) {
			Optional<CodeModel> code = m.code();
			if (code.isEmpty())
				continue;
			List<CodeElement> el = code.get().elementList();
			Map<Label, Integer> at = new HashMap<>();
			for (int i = 0; i < el.size(); i++)
				if (el.get(i) instanceof LabelTarget t)
					at.put(t.label(), i);
			// vector locals stored to inside their own scope: reassigned, which in a loop means carried
			List<int[]> carried = new ArrayList<>();
			List<String> names = new ArrayList<>();
			for (CodeElement e : el)
				if (e instanceof LocalVariable lv
						&& lv.typeSymbol().descriptorString().startsWith("Ljdk/incubator/vector/")) {
					int from = at.get(lv.startScope()), to = at.get(lv.endScope());
					for (int i = from; i < to; i++)
						if (el.get(i) instanceof StoreInstruction s && s.slot() == lv.slot()) {
							carried.add(new int[] { lv.slot(), from, to });
							names.add(lv.name().stringValue());
							break;
						}
				}
			for (int i = 0; i < el.size(); i++) {
				if (!(el.get(i) instanceof InvokeInstruction inv) || inv.opcode() == Opcode.INVOKESTATIC
						|| !inv.owner().asInternalName().startsWith("jdk/incubator/vector/"))
					continue;
				int k = producer(el, i, inv.typeSymbol().parameterCount() + 1);
				if (k >= 0 && el.get(k) instanceof LoadInstruction ld)
					for (int c = 0; c < carried.size(); c++) {
						int[] v = carried.get(c);
						if (ld.slot() == v[0] && k >= v[1] && k < v[2])
							found.add(m.methodName().stringValue() + ": " + inv.name().stringValue() + " on "
									+ names.get(c));
					}
			}
		}
		return found;
	}

	/**
	 * Index of the instruction that pushed the value at {@code depth} (1 = top) just before element {@code i}, walking
	 * back through straight-line code; -1 where the walk meets an instruction it does not model.
	 */
	private static int producer(List<CodeElement> el, int i, int depth) {
		for (int k = i - 1; k >= 0; k--) {
			if (!(el.get(k) instanceof Instruction ins))
				continue;
			int[] e = effect(ins);
			if (e == null)
				return -1;
			if (depth <= e[1])
				return k;
			depth = depth - e[1] + e[0];
		}
		return -1;
	}

	/** {pops, pushes}, counted in values, of an instruction the walk models; {@code null} for any other. */
	private static int[] effect(Instruction ins) {
		return switch (ins) {
		case LoadInstruction _, ConstantInstruction _ -> new int[] { 0, 1 };
		case StoreInstruction _ -> new int[] { 1, 0 };
		case ArrayLoadInstruction _ -> new int[] { 2, 1 };
		case ArrayStoreInstruction _ -> new int[] { 3, 0 };
		case ConvertInstruction _, TypeCheckInstruction _ -> new int[] { 1, 1 };
		case IncrementInstruction _ -> new int[] { 0, 0 };
		case FieldInstruction f -> switch (f.opcode()) {
		case GETSTATIC -> new int[] { 0, 1 };
		case GETFIELD -> new int[] { 1, 1 };
		case PUTSTATIC -> new int[] { 1, 0 };
		default -> new int[] { 2, 0 };
		};
		case InvokeInstruction c -> new int[] {
				c.typeSymbol().parameterCount() + (c.opcode() == Opcode.INVOKESTATIC ? 0 : 1), returns(c.typeSymbol()) };
		case InvokeDynamicInstruction c -> new int[] { c.typeSymbol().parameterCount(), returns(c.typeSymbol()) };
		case OperatorInstruction o -> switch (o.opcode()) {
		case INEG, LNEG, FNEG, DNEG, ARRAYLENGTH -> new int[] { 1, 1 };
		default -> new int[] { 2, 1 };
		};
		default -> null;
		};
	}

	private static int returns(java.lang.constant.MethodTypeDesc t) {
		return t.returnType().descriptorString().equals("V") ? 0 : 1;
	}

	private static boolean isVector(Class<?> c) {
		return c.getName().startsWith("jdk.incubator.vector.");
	}

	private static long allThreads(com.sun.management.ThreadMXBean mx) {
		long total = 0;
		for (long b : mx.getThreadAllocatedBytes(mx.getAllThreadIds()))
			if (b > 0)
				total += b;
		return total;
	}
}
