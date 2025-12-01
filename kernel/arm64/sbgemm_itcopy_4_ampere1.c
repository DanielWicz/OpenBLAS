/***************************************************************************
 * Pack A (op(A) shape: m rows, n columns) for AmpereOne BF16 GEMM.
 *
 * Layout produced for each 8-row tile:
 *   for kk in [0..n) step 4:
 *     Row0 kk..kk+3
 *     Row1 kk..kk+3
 *     ...
 *     Row7 kk..kk+3    (32 bf16)
 *   remaining k (<4): row0..row7 for that k (8 bf16 each)
 *
 * Remaining rows (<8) are packed row-major (row contiguous, length=n).
 * This layout matches the AmpereOne 8x4 micro-kernel, which loads
 * row blocks with offsets {0,4,8,...} inside each 4-wide K chunk and
 * expects leftover-K values as consecutive rows.
 ***************************************************************************/
#define BFLOAT16
#define SBGEMM
#include "common.h"

int CNAME(BLASLONG m, BLASLONG n, IFLOAT *src, BLASLONG lda, IFLOAT *dst) {
  BLASLONG mb8 = m >> 3;       /* number of 8-row tiles   */
  BLASLONG rem_m = m & 7;      /* leftover rows           */

  IFLOAT *out = dst;

  /* 8-row tiles */
  for (BLASLONG ib = 0; ib < mb8; ++ib) {
    IFLOAT *row_base = src + ib * 8;  /* points to row0 of the tile */

    /* K blocks of 4 */
    BLASLONG kk = 0;
    for (; kk + 3 < n; kk += 4) {
      for (BLASLONG r = 0; r < 8; ++r) {
        IFLOAT *row_ptr = row_base + r;
        out[0] = row_ptr[kk * lda];
        out[1] = row_ptr[(kk + 1) * lda];
        out[2] = row_ptr[(kk + 2) * lda];
        out[3] = row_ptr[(kk + 3) * lda];
        out += 4;
      }
    }

    /* K tail (<4): store row0..row7 for each remaining k */
    for (; kk < n; ++kk) {
      for (BLASLONG r = 0; r < 8; ++r) {
        IFLOAT *row_ptr = row_base + r;
        *out++ = row_ptr[kk * lda];
      }
    }
  }

  /* Remaining rows (<8), packed row-major */
  if (rem_m) {
    BLASLONG row0 = mb8 * 8;
    for (BLASLONG r = 0; r < rem_m; ++r) {
      IFLOAT *row_ptr = src + row0 + r;
      for (BLASLONG kk = 0; kk < n; ++kk) {
        *out++ = row_ptr[kk * lda];
      }
    }
  }

  return 0;
}

/* No extra aliases: the build compiles this source twice, once with
 * CNAME=sbgemm_incopy and once with CNAME=bgemm_incopy. */
