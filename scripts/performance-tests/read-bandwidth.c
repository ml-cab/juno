/*
 * Native cross-check for ReadBandwidth.java: the same sequential multi-threaded
 * read (static partition, four accumulators per thread, `passes` reads of the
 * buffer per timing), compiled with the host's own vector extensions. If the
 * JVM reading sits well below this one, the gap is the JVM's, not the host's.
 *
 * Build and run (read-bandwidth-microbench.sh does both):
 *   cc -O3 -march=native -fopenmp read-bandwidth.c -o read-bandwidth
 *   ./read-bandwidth <gib> <threads,...> <reps> <passes>
 */
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ns(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static int cmp(const void *a, const void *b) {
	double x = *(const double *)a, y = *(const double *)b;
	return (x > y) - (x < y);
}

static volatile uint64_t sink;

static double measure(const uint64_t *buf, size_t n, int threads, int passes) {
	uint64_t total = 0;
	double t0 = now_ns();
#pragma omp parallel num_threads(threads) reduction(+ : total)
	{
		int w = omp_get_thread_num();
		size_t per = n / threads;
		const uint64_t *a = buf + (size_t)w * per;
		uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
		for (int p = 0; p < passes; p++)
			for (size_t i = 0; i + 3 < per; i += 4) {
				s0 += a[i];
				s1 += a[i + 1];
				s2 += a[i + 2];
				s3 += a[i + 3];
			}
		total += s0 + s1 + s2 + s3;
	}
	double ns = now_ns() - t0;
	sink += total;
	return ns;
}

int main(int argc, char **argv) {
	if (argc != 5) {
		fprintf(stderr, "usage: %s <gib> <threads,...> <reps> <passes>\n", argv[0]);
		return 2;
	}
	double gib = atof(argv[1]);
	int reps = atoi(argv[3]);
	int passes = atoi(argv[4]);
	size_t n = (size_t)(gib * (1UL << 30) / 8) / 96 * 96;
	uint64_t *buf = aligned_alloc(64, n * 8);
	if (!buf) {
		perror("aligned_alloc");
		return 1;
	}
#pragma omp parallel for
	for (size_t i = 0; i < n; i++)
		buf[i] = i;

	char *list = strdup(argv[2]);
	for (char *tok = strtok(list, ","); tok; tok = strtok(NULL, ",")) {
		int t = atoi(tok);
		measure(buf, n, t, 1);
		double gbs[64];
		for (int r = 0; r < reps && r < 64; r++)
			gbs[r] = (double)n * 8 * passes / measure(buf, n, t, passes);
		printf("{\"tool\":\"c\",\"threads\":%d,\"buffer_bytes\":%zu,\"passes\":%d,\"gbs\":[", t, n * 8, passes);
		for (int r = 0; r < reps; r++)
			printf("%s%.2f", r ? "," : "", gbs[r]);
		qsort(gbs, reps, sizeof(double), cmp);
		printf("],\"median_gbs\":%.2f}\n", gbs[reps / 2]);
	}
	free(list);
	free(buf);
	return 0;
}
