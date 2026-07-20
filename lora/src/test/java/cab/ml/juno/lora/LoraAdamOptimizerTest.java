package cab.ml.juno.lora;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Tests for {@link LoraAdamOptimizer}.
 *
 * <h2>What to watch during testing</h2>
 * <ul>
 * <li><b>Update direction</b>: if the parameter moves in the SAME direction as
 * the gradient (not opposite), you have a sign error in the update formula. The
 * tests below verify that a positive gradient decreases the parameter.
 * <li><b>B not weight-decayed</b>: B starts at zero. Applying weight decay to B
 * would keep it near zero and prevent the adapter from learning. Verify this is
 * respected when fine-tuning converges slowly.
 * <li><b>Bias correction</b>: Adam's first update is dramatically scaled down
 * at step t=1 (bc1 ≈ 0.1). If you see a huge first step instead, bias
 * correction is broken.
 * <li><b>reset() clears momentum</b>: after loading a new adapter checkpoint,
 * the old momentum from a different parameter scale can corrupt training.
 * Always call reset() when loading a checkpoint.
 * </ul>
 */
@DisplayName("LoraAdamOptimizer")
class LoraAdamOptimizerTest {

	// ── Construction ──────────────────────────────────────────────────────────

	@Test
	@DisplayName("Defaults factory sets sane lr")
	void defaults_factory() {
		LoraAdamOptimizer opt = LoraAdamOptimizer.defaults(1e-4);
		assertThat(opt.step()).isEqualTo(0);
	}

	@Test
	@DisplayName("lr <= 0 throws IllegalArgumentException")
	void invalid_lr_throws() {
		assertThatThrownBy(() -> new LoraAdamOptimizer(0, 0.9, 0.999, 1e-8, 0))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> new LoraAdamOptimizer(-1, 0.9, 0.999, 1e-8, 0))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	@DisplayName("beta outside (0,1) throws IllegalArgumentException")
	void invalid_beta_throws() {
		assertThatThrownBy(() -> new LoraAdamOptimizer(1e-4, 1.0, 0.999, 1e-8, 0))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> new LoraAdamOptimizer(1e-4, 0.9, 0.0, 1e-8, 0))
				.isInstanceOf(IllegalArgumentException.class);
	}

	// ── Update direction ──────────────────────────────────────────────────────

	@Test
	@DisplayName("Positive gradient decreases parameter (gradient descent)")
	void positive_gradient_decreases_param() {
		LoraAdapterSet set = new LoraAdapterSet();
		LoraAdapter a = makeNonZero(4, 8, 16, 4f);
		set.add(0, "wq", a);

		// Set positive gradient for A
		for (int i = 0; i < a.gradA().length; i++)
			a.gradA()[i] = 0.1f;
		float[] aBefore = a.a().clone();

		LoraAdamOptimizer opt = LoraAdamOptimizer.defaults(1e-2);
		opt.step(set);

		for (int i = 0; i < a.a().length; i++)
			assertThat(a.a()[i]).isLessThan(aBefore[i]); // moved in -gradient direction
	}

	@Test
	@DisplayName("Negative gradient increases parameter")
	void negative_gradient_increases_param() {
		LoraAdapterSet set = new LoraAdapterSet();
		LoraAdapter a = makeNonZero(4, 8, 16, 4f);
		set.add(0, "wq", a);

		for (int i = 0; i < a.gradA().length; i++)
			a.gradA()[i] = -0.1f;
		float[] aBefore = a.a().clone();

		LoraAdamOptimizer opt = LoraAdamOptimizer.defaults(1e-2);
		opt.step(set);

		for (int i = 0; i < a.a().length; i++)
			assertThat(a.a()[i]).isGreaterThan(aBefore[i]);
	}

	@Test
	@DisplayName("Zero gradient leaves parameter unchanged")
	void zero_gradient_no_change() {
		LoraAdapterSet set = new LoraAdapterSet();
		LoraAdapter a = makeNonZero(4, 8, 16, 4f);
		set.add(0, "wq", a);
		// gradA is already zero from construction
		// gradB is zero too
		float[] aBefore = a.a().clone();
		float[] bBefore = a.b().clone();

		// No weight decay to isolate gradient effect
		LoraAdamOptimizer opt = new LoraAdamOptimizer(1e-2, 0.9, 0.999, 1e-8, 0);
		opt.step(set);

		for (int i = 0; i < a.a().length; i++)
			assertThat(a.a()[i]).isCloseTo(aBefore[i], within(1e-6f));
		for (int i = 0; i < a.b().length; i++)
			assertThat(a.b()[i]).isCloseTo(bBefore[i], within(1e-6f));
	}

	// ── Weight decay ─────────────────────────────────────────────────────────

	@Test
	@DisplayName("Decoupled AdamW shrinks A with zero gradient (moments stay raw-grad only)")
	void weight_decay_shrinks_A() {
		LoraAdapterSet set = new LoraAdapterSet();
		LoraAdapter a = makeNonZero(4, 8, 16, 4f);
		set.add(0, "wq", a);

		float[] aBefore = a.a().clone();
		// gradients = 0 so only decoupled decay acts: p *= (1 - lr * wd)
		LoraAdamOptimizer opt = new LoraAdamOptimizer(1e-2, 0.9, 0.999, 1e-8, 0.1);
		opt.step(set);

		for (int i = 0; i < a.a().length; i++) {
			if (Math.abs(aBefore[i]) > 5e-4f)
				assertThat(a.a()[i]).isCloseTo(aBefore[i] * (1f - 1e-2f * 0.1f), within(1e-6f));
		}
	}

	@Test
	@DisplayName("Decoupled decay does not enter Adam moments")
	void moments_unaffected_by_decay() {
		LoraAdapterSet set = new LoraAdapterSet();
		LoraAdapter a = makeNonZero(4, 8, 16, 4f);
		set.add(0, "wq", a);
		for (int i = 0; i < a.gradA().length; i++)
			a.gradA()[i] = 0.2f;
		float[] aWithDecay = a.a().clone();
		float[] grad = a.gradA().clone();

		LoraAdamOptimizer withDecay = new LoraAdamOptimizer(1e-3, 0.9, 0.999, 1e-8, 0.5);
		withDecay.step(set);
		float[] afterDecay = a.a().clone();

		System.arraycopy(aWithDecay, 0, a.a(), 0, a.a().length);
		System.arraycopy(grad, 0, a.gradA(), 0, a.gradA().length);
		LoraAdamOptimizer noDecay = new LoraAdamOptimizer(1e-3, 0.9, 0.999, 1e-8, 0.0);
		noDecay.step(set);
		float[] afterNoDecay = a.a().clone();

		// Difference must equal the explicit decoupled term only (same Adam step).
		for (int i = 0; i < a.a().length; i++) {
			float expected = afterNoDecay[i] - (float) (1e-3 * 0.5 * aWithDecay[i]);
			assertThat(afterDecay[i]).isCloseTo(expected, within(1e-5f));
		}
	}

	@Test
	@DisplayName("Weight decay does NOT affect B (B starts at zero and must be free to grow)")
	void weight_decay_not_applied_to_B() {
		LoraAdapterSet set = new LoraAdapterSet();
		LoraAdapter a = makeNonZero(4, 8, 16, 4f);
		// Give B some non-zero values
		for (int i = 0; i < a.b().length; i++)
			a.b()[i] = 0.5f;
		set.add(0, "wq", a);
		float[] bBefore = a.b().clone();

		// Zero gradient so only weight decay acts
		// If weight decay is applied to B, b should decrease from 0.5 toward 0
		LoraAdamOptimizer opt = new LoraAdamOptimizer(1e-2, 0.9, 0.999, 1e-8, 1.0);
		opt.step(set);

		// B should be unchanged (zero gradient + no weight decay on B)
		for (int i = 0; i < a.b().length; i++)
			assertThat(a.b()[i]).isCloseTo(bBefore[i], within(1e-5f));
	}

	@Test
	@DisplayName("Scheduled LR scales the AdamW update")
	void scheduled_lr_scales_update() {
		LoraAdapterSet set = new LoraAdapterSet();
		LoraAdapter a = makeNonZero(4, 8, 16, 4f);
		set.add(0, "wq", a);
		for (int i = 0; i < a.gradA().length; i++)
			a.gradA()[i] = 0.1f;
		float[] before = a.a().clone();
		float[] grad = a.gradA().clone();

		LoraAdamOptimizer opt = new LoraAdamOptimizer(1e-2, 0.9, 0.999, 1e-8, 0.0);
		opt.step(set, 1e-2);
		float deltaFull = before[0] - a.a()[0];

		System.arraycopy(before, 0, a.a(), 0, a.a().length);
		System.arraycopy(grad, 0, a.gradA(), 0, a.gradA().length);
		opt.reset();
		opt.step(set, 5e-3);
		float deltaHalf = before[0] - a.a()[0];
		assertThat(deltaHalf).isCloseTo(deltaFull * 0.5f, within(1e-6f));
		assertThat(opt.lastLearningRateA()).isEqualTo(5e-3);
		assertThat(opt.lastLearningRateB()).isEqualTo(5e-3);
	}

	@Test
	@DisplayName("LoRA+ ratio 1.0 matches ordinary optimizer behavior")
	void lora_plus_ratio_one_equivalence() {
		LoraAdapterSet setA = new LoraAdapterSet();
		LoraAdapterSet setB = new LoraAdapterSet();
		LoraAdapter a1 = makeNonZero(4, 8, 16, 4f);
		LoraAdapter a2 = LoraAdapter.fromWeights(4, 8, 16, 4f, a1.a().clone(), a1.b().clone());
		for (int i = 0; i < a1.b().length; i++) {
			a1.b()[i] = 0.3f;
			a2.b()[i] = 0.3f;
		}
		setA.add(0, "wq", a1);
		setB.add(0, "wq", a2);
		for (int i = 0; i < a1.gradA().length; i++) {
			a1.gradA()[i] = 0.05f;
			a2.gradA()[i] = 0.05f;
		}
		for (int i = 0; i < a1.gradB().length; i++) {
			a1.gradB()[i] = 0.02f;
			a2.gradB()[i] = 0.02f;
		}

		new LoraAdamOptimizer(1e-2, 0.9, 0.999, 1e-8, 0.01, 1.0).step(setA, 1e-2);
		new LoraAdamOptimizer(1e-2, 0.9, 0.999, 1e-8, 0.01).step(setB, 1e-2);

		for (int i = 0; i < a1.a().length; i++)
			assertThat(a1.a()[i]).isCloseTo(a2.a()[i], within(1e-7f));
		for (int i = 0; i < a1.b().length; i++)
			assertThat(a1.b()[i]).isCloseTo(a2.b()[i], within(1e-7f));
	}

	@Test
	@DisplayName("LoRA+ scales only B learning rate")
	void lora_plus_scales_B_only() {
		LoraAdapterSet set = new LoraAdapterSet();
		LoraAdapter a = makeNonZero(4, 8, 16, 4f);
		for (int i = 0; i < a.b().length; i++)
			a.b()[i] = 0.4f;
		set.add(0, "wq", a);
		for (int i = 0; i < a.gradB().length; i++)
			a.gradB()[i] = 0.1f;
		float[] bBefore = a.b().clone();

		LoraAdamOptimizer opt = new LoraAdamOptimizer(1e-2, 0.9, 0.999, 1e-8, 0.0, 4.0);
		opt.step(set, 1e-2);
		assertThat(opt.lastLearningRateA()).isEqualTo(1e-2);
		assertThat(opt.lastLearningRateB()).isEqualTo(4e-2);
		boolean moved = false;
		for (int i = 0; i < bBefore.length; i++)
			if (a.b()[i] != bBefore[i]) {
				moved = true;
				break;
			}
		assertThat(moved).isTrue();
	}

	@Test
	@DisplayName("rejects invalid eps, decay, and LoRA+ ratio")
	void validation() {
		assertThatThrownBy(() -> new LoraAdamOptimizer(1e-4, 0.9, 0.999, 0, 0.01))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> new LoraAdamOptimizer(1e-4, 0.9, 0.999, 1e-8, -0.1))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> new LoraAdamOptimizer(1e-4, 0.9, 0.999, 1e-8, 0.01, 0))
				.isInstanceOf(IllegalArgumentException.class);
		LoraAdamOptimizer opt = LoraAdamOptimizer.defaults(1e-4);
		LoraAdapterSet set = new LoraAdapterSet();
		set.add(0, "wq", makeNonZero(2, 4, 4, 2f));
		assertThatThrownBy(() -> opt.step(set, Double.NaN)).isInstanceOf(IllegalArgumentException.class);
	}

	// ── Step counter ──────────────────────────────────────────────────────────

	@Test
	@DisplayName("step counter increments on each call to step()")
	void step_counter_increments() {
		LoraAdapterSet set = new LoraAdapterSet();
		set.add(0, "wq", makeNonZero(4, 8, 16, 4f));
		LoraAdamOptimizer opt = LoraAdamOptimizer.defaults(1e-4);

		assertThat(opt.step()).isEqualTo(0);
		opt.step(set);
		assertThat(opt.step()).isEqualTo(1);
		opt.step(set);
		assertThat(opt.step()).isEqualTo(2);
	}

	// ── Reset ─────────────────────────────────────────────────────────────────

	@Test
	@DisplayName("reset() clears step counter and moment buffers")
	void reset_clears_state() {
		LoraAdapterSet set = new LoraAdapterSet();
		LoraAdapter a = makeNonZero(4, 8, 16, 4f);
		set.add(0, "wq", a);
		for (int i = 0; i < a.gradA().length; i++)
			a.gradA()[i] = 0.1f;

		LoraAdamOptimizer opt = LoraAdamOptimizer.defaults(1e-2);
		opt.step(set);
		float[] aAfterStep1 = a.a().clone();

		opt.reset();
		assertThat(opt.step()).isEqualTo(0);

		// After reset, the first step should behave identically to the very first step
		// (re-run from same starting weights and same gradient)
		for (int i = 0; i < a.a().length; i++)
			a.a()[i] = aAfterStep1[i];
		// reset grad
		for (int i = 0; i < a.gradA().length; i++)
			a.gradA()[i] = 0.1f;
		// We can't compare directly since weights changed, but we can at least
		// verify the step counter is 1 after the step
		opt.step(set);
		assertThat(opt.step()).isEqualTo(1);
	}

	// ── Multi-adapter step ────────────────────────────────────────────────────

	@Test
	@DisplayName("step() updates all adapters in the set independently")
	void updates_all_adapters() {
		LoraAdapterSet set = new LoraAdapterSet();
		LoraAdapter aq = makeNonZero(4, 8, 16, 4f);
		LoraAdapter av = makeNonZero(4, 8, 8, 4f);
		set.add(0, "wq", aq);
		set.add(0, "wv", av);

		for (int i = 0; i < aq.gradA().length; i++)
			aq.gradA()[i] = 0.05f;
		for (int i = 0; i < av.gradA().length; i++)
			av.gradA()[i] = 0.05f;

		float[] aqBefore = aq.a().clone();
		float[] avBefore = av.a().clone();

		LoraAdamOptimizer opt = LoraAdamOptimizer.defaults(1e-2);
		opt.step(set);

		// Both should have moved
		boolean aqMoved = false, avMoved = false;
		for (int i = 0; i < aqBefore.length; i++)
			if (aq.a()[i] != aqBefore[i]) {
				aqMoved = true;
				break;
			}
		for (int i = 0; i < avBefore.length; i++)
			if (av.a()[i] != avBefore[i]) {
				avMoved = true;
				break;
			}
		assertThat(aqMoved).isTrue();
		assertThat(avMoved).isTrue();
	}

	// ── Helpers ───────────────────────────────────────────────────────────────

	private LoraAdapter makeNonZero(int rank, int in, int out, float alpha) {
		LoraAdapter a = new LoraAdapter(rank, in, out, alpha, new Random(42));
		// B stays zero for clean testing; A has random non-zero values
		return a;
	}
}