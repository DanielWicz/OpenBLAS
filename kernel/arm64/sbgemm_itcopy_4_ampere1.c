/***************************************************************************
 * Pack B for AmpereOne BF16 GEMM (4 cols).
 * Case: B is Transposed. Input is N x K.
 * Arguments:
 *   m: Number of rows of Op(B) -> K
 *   n: Number of cols of Op(B) -> N
 * Input Row j is at: src + j + k * lda. (Stride lda).
 * Output layout (Interleaved K=4):
 *  Col0[0..3], Col1[0..3] ...
 ***************************************************************************/
#define BFLOAT16
#define SBGEMM
#include "common.h"

int CNAME(BLASLONG m, BLASLONG n, IFLOAT *src, BLASLONG lda, IFLOAT *dst) {
  // m is K
  // n is N
  BLASLONG k = m;
  BLASLONG n4 = n >> 2;
  BLASLONG rem = n & 3;

  for (BLASLONG jb = 0; jb < n4; ++jb) {
    IFLOAT *out = dst + jb * (k * 4);
    
    // Pack 4 columns of B (Rows of B^T) interleaved
    // For each K, store B(k, 0..3).
    // src is Col Major B^T.
    // B(k, j) is at src + k*lda + j.
    
    for (BLASLONG kk = 0; kk < k; ++kk) {
      IFLOAT *ptr = src + jb * 4 + kk * lda;
      out[0] = ptr[0];
      out[1] = ptr[1];
      out[2] = ptr[2];
      out[3] = ptr[3];
      out += 4;
    }
  }

  if (rem) {
      BLASLONG jb = n4 * 4;
      IFLOAT *out = dst + n4 * (k * 4);
      for (BLASLONG c = 0; c < rem; ++c) {
          IFLOAT *col_ptr = src + (jb + c);
          for (BLASLONG kk = 0; kk < k; ++kk) {
              *out++ = *(col_ptr + kk * lda);
          }
      }
  }

  return 0;
}
