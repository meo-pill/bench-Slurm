#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double now_sec(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
#endif
}

static uint64_t bench_kernel(double duration_s) {
    // Kernel simple: accumulation de calculs flottants pour occuper le CPU
    // et éviter l'optimisation excessive.
    volatile double acc = 0.0;
    uint64_t iters = 0;
    double t0 = now_sec();
    double t;
    do {
        // 256 opérations flottantes approximatives
        for (int i = 1; i <= 256; ++i) {
            acc += sin((double)i) * cos((double)i) + sqrt((double)i);
        }
        iters++;
        t = now_sec();
    } while ((t - t0) < duration_s);
    (void)acc; // évite d'être optimisé
    return iters * 256ULL; // événements approximatifs
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s [--duration <seconds>] [--verbose]\n", prog);
}

int main(int argc, char **argv) {
    double dur = 3.0;
    int verbose = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--duration") == 0 && i + 1 < argc) {
            dur = atof(argv[++i]);
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    // Multi-thread contrôlé par OMP_NUM_THREADS
    int threads = 1;
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single
        {
            threads = omp_get_num_threads();
        }
    }
#endif

    if (verbose) {
        printf("START threads=%d duration=%.3f\n", threads, dur);
    }

    uint64_t total = 0;
#ifdef _OPENMP
    #pragma omp parallel reduction(+:total)
#endif
    {
        total += bench_kernel(dur);
    }

    double score = (double)total / dur; // events per second
    printf("THREADS %d\n", threads);
    printf("DURATION %.3f\n", dur);
    printf("SCORE %.3f\n", score);
    return 0;
}
