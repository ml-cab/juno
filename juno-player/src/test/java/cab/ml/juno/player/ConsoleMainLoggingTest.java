package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.lang.reflect.Method;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.node.ActivationDtype;

/**
 * Unit tests for ConsoleMain.parseDtype().
 *
 * Regression coverage for a bug where any unrecognized {@code --dtype} value
 * (a typo, or an unsupported quantization label such as {@code INT4}) was
 * silently coerced to {@code FLOAT32} with no feedback — the switch's
 * {@code default} branch was shared by both "explicitly requested FLOAT32"
 * and "garbage input", so there was no way to distinguish the two from the
 * resulting {@link ActivationDtype}. A user passing {@code --dtype INT4} saw
 * the raw string echoed back in the startup banner and had no indication
 * that it had actually been ignored.
 *
 * {@code parseDtype} is private with no side effects beyond stdout/stderr, so
 * these tests call it via reflection and capture {@code System.err}.
 */
class ConsoleMainDtypeTest {

	private final PrintStream originalErr = System.err;

	@AfterEach
	void restoreStderr() {
		System.setErr(originalErr);
	}

	@Test
	@DisplayName("explicit FLOAT32 is accepted without a warning")
	void explicitFloat32NoWarning() throws Exception {
		ByteArrayOutputStream captured = new ByteArrayOutputStream();
		System.setErr(new PrintStream(captured));

		assertThat(invokeParseDtype("FLOAT32")).isEqualTo(ActivationDtype.FLOAT32);
		assertThat(captured.toString()).isEmpty();
	}

	@Test
	@DisplayName("FLOAT16 and its aliases parse correctly")
	void float16Aliases() throws Exception {
		assertThat(invokeParseDtype("FLOAT16")).isEqualTo(ActivationDtype.FLOAT16);
		assertThat(invokeParseDtype("f16")).isEqualTo(ActivationDtype.FLOAT16);
		assertThat(invokeParseDtype("fp16")).isEqualTo(ActivationDtype.FLOAT16);
	}

	@Test
	@DisplayName("INT8 and its alias parse correctly")
	void int8Aliases() throws Exception {
		assertThat(invokeParseDtype("INT8")).isEqualTo(ActivationDtype.INT8);
		assertThat(invokeParseDtype("i8")).isEqualTo(ActivationDtype.INT8);
	}

	@Test
	@DisplayName("null input defaults to FLOAT16 (documented default), no warning needed")
	void nullDefaultsToFloat16() throws Exception {
		assertThat(invokeParseDtype(null)).isEqualTo(ActivationDtype.FLOAT16);
	}

	@Test
	@DisplayName("an unrecognized value (e.g. INT4) falls back to FLOAT32 AND prints a warning")
	void unrecognizedValueWarnsAndFallsBackToFloat32() throws Exception {
		ByteArrayOutputStream captured = new ByteArrayOutputStream();
		System.setErr(new PrintStream(captured));

		ActivationDtype result = invokeParseDtype("INT4");

		assertThat(result).isEqualTo(ActivationDtype.FLOAT32);
		assertThat(captured.toString()).contains("WARNING").contains("INT4").contains("FLOAT32");
	}

	private static ActivationDtype invokeParseDtype(String value) throws Exception {
		Method m = ConsoleMain.class.getDeclaredMethod("parseDtype", String.class);
		m.setAccessible(true);
		return (ActivationDtype) m.invoke(null, value);
	}
}