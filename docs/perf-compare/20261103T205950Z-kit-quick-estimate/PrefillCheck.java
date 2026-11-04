import java.util.EnumSet;

import cab.ml.juno.master.ModelLiveChecks;

/** Runs live check 9 (long-prompt prefill, both schedules) alone on the CPU backend and prints its result and time. */
public class PrefillCheck {
	public static void main(String[] args) throws Exception {
		long t0 = System.nanoTime();
		var results = ModelLiveChecks.run(args[0], EnumSet.of(ModelLiveChecks.Suite.PIPELINE, ModelLiveChecks.Suite.TENSOR, ModelLiveChecks.Suite.PREFILL),
				ModelLiveChecks.Backend.CPU, System.out);
		System.out.println("RESULT " + results + " wallMs=" + (System.nanoTime() - t0) / 1_000_000);
	}
}
