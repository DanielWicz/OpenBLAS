/***************************************************************************
 * Pack A for AmpereOne BF16 GEMM (8 rows).
 * Case: A is Transposed. Input is K x M.
 * We want 8 rows of Op(A) -> 8 cols of Input.
 * Input Col i is at: src + i * lda. Elements contiguous.
 * Output layout (Interleaved K=4):
 *  Row0[0..3], Row1[0..3] ...
 ***************************************************************************/
#include "common.h"

int CNAME(BLASLONG m, BLASLONG k, IFLOAT *src, BLASLONG lda, IFLOAT *dst) {
  BLASLONG m8 = m >> 3;
  BLASLONG rem = m & 7;

  for (BLASLONG ib = 0; ib < m8; ++ib) {
    IFLOAT *col0 = src + ib * 8 * lda;
    IFLOAT *col1 = col0 + lda;
    IFLOAT *col2 = col1 + lda;
    IFLOAT *col3 = col2 + lda;
    IFLOAT *col4 = col3 + lda;
    IFLOAT *col5 = col4 + lda;
    IFLOAT *col6 = col5 + lda;
    IFLOAT *col7 = col6 + lda;
    IFLOAT *out = dst + ib * (k * 8);

    BLASLONG kk = 0;
    for (; kk + 3 < k; kk += 4) {
      out[0] = col0[kk+0]; out[1] = col0[kk+1]; out[2] = col0[kk+2]; out[3] = col0[kk+3];
      out[4] = col1[kk+0]; out[5] = col1[kk+1]; out[6] = col1[kk+2]; out[7] = col1[kk+3];
      out[8] = col2[kk+0]; out[9] = col2[kk+1]; out[10] = col2[kk+2]; out[11] = col2[kk+3];
      out[12] = col3[kk+0]; out[13] = col3[kk+1]; out[14] = col3[kk+2]; out[15] = col3[kk+3];
      out[16] = col4[kk+0]; out[17] = col4[kk+1]; out[18] = col4[kk+2]; out[19] = col4[kk+3];
      out[20] = col5[kk+0]; out[21] = col5[kk+1]; out[22] = col5[kk+2]; out[23] = col5[kk+3];
      out[24] = col6[kk+0]; out[25] = col6[kk+1]; out[26] = col6[kk+2]; out[27] = col6[kk+3];
      out[28] = col7[kk+0]; out[29] = col7[kk+1]; out[30] = col7[kk+2]; out[31] = col7[kk+3];
      out += 32;
    }
    for (; kk < k; ++kk) {
      out[0] = col0[kk];
      out[1] = col1[kk];
      out[2] = col2[kk];
      out[3] = col3[kk];
      out[4] = col4[kk];
      out[5] = col5[kk];
      out[6] = col6[kk];
      out[7] = col7[kk];
      out += 8;
    }
  }

  if (rem) {
      BLASLONG ib = m8 * 8;
      IFLOAT *out = dst + m8 * (k * 8);
      for (BLASLONG r = 0; r < rem; ++r) {
          IFLOAT *col = src + (ib + r) * lda;
          for (BLASLONG kk = 0; kk < k; ++kk) {
              *out++ = col[kk];
          }
      }
  }

  return 0;
}
