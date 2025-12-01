/***************************************************************************
 * Pack B for AmpereOne BF16 GEMM (4 cols).
 * Case: B is Transposed. Input is N x K.
 * We want 4 cols of Op(B) -> 4 rows of Input.
 * Input Row j is at: src + j + k * lda. (Stride lda).
 * Output layout (Interleaved K=4):
 *  Col0[0..3], Col1[0..3] ...
 ***************************************************************************/
#define SBGEMM
#include "common.h"

int CNAME(BLASLONG n, BLASLONG k, IFLOAT *src, BLASLONG lda, IFLOAT *dst) {
  BLASLONG n4 = n >> 2;
  BLASLONG rem = n & 3;

  for (BLASLONG jb = 0; jb < n4; ++jb) {
    IFLOAT *row_ptr = src + jb * 4;
    IFLOAT *out = dst + jb * (k * 4);

    BLASLONG kk = 0;
    for (; kk + 3 < k; kk += 4) {
      IFLOAT *ptr0 = row_ptr + kk * lda;
      IFLOAT *ptr1 = ptr0 + lda;
      IFLOAT *ptr2 = ptr1 + lda;
      IFLOAT *ptr3 = ptr2 + lda;

      out[0]  = ptr0[0]; out[1]  = ptr1[0]; out[2]  = ptr2[0]; out[3]  = ptr3[0];
      out[4]  = ptr0[1]; out[5]  = ptr1[1]; out[6]  = ptr2[1]; out[7]  = ptr3[1];
      out[8]  = ptr0[2]; out[9]  = ptr1[2]; out[10] = ptr2[2]; out[11] = ptr3[2];
      out[12] = ptr0[3]; out[13] = ptr1[3]; out[14] = ptr2[3]; out[15] = ptr3[3];
      out += 16;
    }
    for (; kk < k; ++kk) {
      IFLOAT *ptr = row_ptr + kk * lda;
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
          IFLOAT *row_ptr = src + (jb + c);
          for (BLASLONG kk = 0; kk < k; ++kk) {
              *out++ = *(row_ptr + kk * lda);
          }
      }
  }

  return 0;
}