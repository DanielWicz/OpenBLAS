/*
 * Helpers for “big vector” parallelization thresholds.
 *
 * Allows runtime override via OPENBLAS_BIGVEC_THRESHOLD (bytes).
 * Otherwise chooses half of L3_SIZE when available, falling back
 * to the historic 128KB threshold.
 */
#ifndef OPENBLAS_BIGVEC_H
#define OPENBLAS_BIGVEC_H

#include <stdlib.h>
#include <stdint.h>

static inline size_t openblas_bigvec_threshold_bytes(void) {
    static size_t cached = 0;
    if (cached == 0) {
        const char *env = getenv("OPENBLAS_BIGVEC_THRESHOLD");
        if (env && *env) {
            cached = (size_t)strtoull(env, NULL, 0);
        }
#if defined(L3_SIZE) && (L3_SIZE > 0)
        if (cached == 0) cached = (size_t)(L3_SIZE / 16);
#endif
        if (cached == 0) cached = (size_t)131072; /* 128KB default */
        /* Clamp to a sensible window: 64KB .. 128KB */
        if (cached < (size_t)65536) cached = (size_t)65536;
        if (cached > (size_t)131072) cached = (size_t)131072;
    }
    return cached;
}

#endif /* OPENBLAS_BIGVEC_H */
