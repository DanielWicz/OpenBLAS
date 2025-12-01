/***************************************************************************
 * Pack B for AmpereOne BF16 GEMM.
 * Input: B (K x N), column-major.
 * Arguments:
 *   m: Number of rows (K)
 *   n: Number of columns (N)
 * Output layout (per 4-column block):
 *  for kk in 0..K-1 step 4:
 *    store B[kk:kk+3, col0], then col1, col2, col3  (16 bf16).
 ***************************************************************************/
#define BFLOAT16
#define SBGEMM
#include "common.h"
#include <stdio.h>

int CNAME(BLASLONG m, BLASLONG n, IFLOAT *src, BLASLONG ldb, IFLOAT *dst) {
  static int debug_print = 0;
  if (!debug_print) {
      printf("NCOPY: IFLOAT=%d\n", (int)sizeof(IFLOAT));
      debug_print = 1;
  }
  // m is K (rows of B)
  // n is N (cols of B)
  BLASLONG k = m;
  BLASLONG n4 = n >> 2;
  BLASLONG rem = n & 3;

  // ... (main loop)

  for (BLASLONG jb = 0; jb < n4; ++jb) {
    // ...
  }

  // leftover columns
  if (rem) {
    BLASLONG jb = n4 * 4;
    IFLOAT *out = dst + n4 * (k * 4);
    for (BLASLONG col = 0; col < rem; ++col) {
      IFLOAT *cptr = src + (jb + col) * ldb;
      for (BLASLONG kk = 0; kk < k; ++kk) {
        // Debug print
        if (m==2 && n==2 && k==2) {
             uint16_t val = *(uint16_t*)&cptr[kk];
             printf("NCOPY: col=%ld kk=%ld src_val=%04x addr=%p out_addr=%p\n", col, kk, val, &cptr[kk], out);
             fflush(stdout);
        }
        *out++ = cptr[kk];
      }
    }
  }
  return 0;
}
    IFLOAT *col0 = src + jb * 4 * ldb;
    IFLOAT *col1 = col0 + ldb;
    IFLOAT *col2 = col1 + ldb;
    IFLOAT *col3 = col2 + ldb;
    IFLOAT *out = dst + jb * (k * 4);

    BLASLONG kk = 0;
    for (; kk + 3 < k; kk += 4) {
      out[0]  = col0[kk + 0];
      out[1]  = col0[kk + 1];
      out[2]  = col0[kk + 2];
      out[3]  = col0[kk + 3];
      out[4]  = col1[kk + 0];
      out[5]  = col1[kk + 1];
      out[6]  = col1[kk + 2];
      out[7]  = col1[kk + 3];
      out[8]  = col2[kk + 0];
      out[9]  = col2[kk + 1];
      out[10] = col2[kk + 2];
      out[11] = col2[kk + 3];
      out[12] = col3[kk + 0];
      out[13] = col3[kk + 1];
      out[14] = col3[kk + 2];
      out[15] = col3[kk + 3];
      out += 16;
    }
    for (; kk < k; ++kk) { // tail K
      out[0]  = col0[kk];
      out[1]  = col1[kk];
      out[2]  = col2[kk];
      out[3]  = col3[kk];
      out += 4;
    }
  }

  // leftover columns
  if (rem) {
    BLASLONG jb = n4 * 4;
    IFLOAT *out = dst + n4 * (k * 4);
    for (BLASLONG col = 0; col < rem; ++col) {
      IFLOAT *cptr = src + (jb + col) * ldb;
      for (BLASLONG kk = 0; kk < k; ++kk) {
        *out++ = cptr[kk];
      }
    }
  }
  return 0;
}
