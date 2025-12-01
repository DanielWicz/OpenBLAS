/***************************************************************************
 * Pack B (op(B) shape: m rows = K, n cols = Nblock) for AmpereOne BF16 GEMM.
 *
 * Layout for each 4-column tile:
 *   for kk in [0..m) step 4:
 *     Col0 kk..kk+3
 *     Col1 kk..kk+3
 *     Col2 kk..kk+3
 *     Col3 kk..kk+3   (16 bf16)
 *   remaining k (<4): [col0, col1, col2, col3] for that k (4 bf16)
 *
 * Remaining columns (<4) are stored column-major, contiguous K values.
 * The same layout serves for packing A (INCOPY) and B (ONCOPY) because
 * the driver supplies appropriate pointers/lda for the requested transpose.
***************************************************************************/
#define BFLOAT16
#define SBGEMM
#include "common.h"

int CNAME(BLASLONG m, BLASLONG n, IFLOAT *src, BLASLONG ldb, IFLOAT *dst) {
  BLASLONG n4 = n >> 2;      /* blocks of 4 columns */
  BLASLONG rem_n = n & 3;    /* leftover columns     */

  IFLOAT *out = dst;

  /* 4-column tiles */
  for (BLASLONG jb = 0; jb < n4; ++jb) {
    IFLOAT *col0 = src + jb * 4 * ldb;
    IFLOAT *col1 = col0 + ldb;
    IFLOAT *col2 = col1 + ldb;
    IFLOAT *col3 = col2 + ldb;

    BLASLONG kk = 0;
    for (; kk + 3 < m; kk += 4) {
      out[0]  = col0[kk + 0]; out[1]  = col0[kk + 1]; out[2]  = col0[kk + 2]; out[3]  = col0[kk + 3];
      out[4]  = col1[kk + 0]; out[5]  = col1[kk + 1]; out[6]  = col1[kk + 2]; out[7]  = col1[kk + 3];
      out[8]  = col2[kk + 0]; out[9]  = col2[kk + 1]; out[10] = col2[kk + 2]; out[11] = col2[kk + 3];
      out[12] = col3[kk + 0]; out[13] = col3[kk + 1]; out[14] = col3[kk + 2]; out[15] = col3[kk + 3];
      out += 16;
    }
    /* K tail (<4): one value per column */
    for (; kk < m; ++kk) {
      out[0] = col0[kk];
      out[1] = col1[kk];
      out[2] = col2[kk];
      out[3] = col3[kk];
      out += 4;
    }
  }

  /* Remaining columns */
  if (rem_n) {
    for (BLASLONG col = 0; col < rem_n; ++col) {
      IFLOAT *cptr = src + (n4 * 4 + col) * ldb;
      for (BLASLONG kk = 0; kk < m; ++kk) {
        *out++ = cptr[kk];
      }
    }
  }

  return 0;
}
