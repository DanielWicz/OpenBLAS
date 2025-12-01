/***************************************************************************
 * Pack A for AmpereOne BF16 GEMM (8 rows).
 * Case: A is Normal (No Transpose). M x K Column Major.
 * We need to pack 8 rows of A.
 * Row i is at: src + i + k * lda. (Stride lda).
 * Output layout (Interleaved K=4):
 *  Row0[0..3], Row1[0..3] ... Row7[0..3]
 ***************************************************************************/ 
#define BFLOAT16
#define SBGEMM
#include "common.h"
#include <stdio.h>

int CNAME(BLASLONG m, BLASLONG n, IFLOAT *src, BLASLONG ldb, IFLOAT *dst) {
  static int debug_print = 0;
  if (!debug_print) {
      printf("ONCOPY called m=%ld n=%ld\n", m, n);
      debug_print = 1;
  }
  
  // Correct argument mapping for ONCOPY (A is M x K):
  // OpenBLAS passes (m, k, ...) to ONCOPY.
  // So 'm' is M, 'n' is K. 'ldb' is actually lda.
  BLASLONG k = n;
  BLASLONG lda = ldb;
  
  BLASLONG m8 = m >> 3;
  BLASLONG rem = m & 7;

  for (BLASLONG ib = 0; ib < m8; ++ib) {
    IFLOAT *block_base = src + ib * 8;
    IFLOAT *out = dst + ib * (k * 8);

    BLASLONG kk = 0;
    for (; kk + 3 < k; kk += 4) {
      IFLOAT *ptr0 = block_base + kk * lda;
      IFLOAT *ptr1 = ptr0 + lda;
      IFLOAT *ptr2 = ptr1 + lda;
      IFLOAT *ptr3 = ptr2 + lda;

      // Row 0 (offset 0)
      out[0] = ptr0[0]; out[1] = ptr1[0]; out[2] = ptr2[0]; out[3] = ptr3[0];
      // Row 1 (offset 1)
      out[4] = ptr0[1]; out[5] = ptr1[1]; out[6] = ptr2[1]; out[7] = ptr3[1];
      // Row 2
      out[8] = ptr0[2]; out[9] = ptr1[2]; out[10] = ptr2[2]; out[11] = ptr3[2];
      // Row 3
      out[12] = ptr0[3]; out[13] = ptr1[3]; out[14] = ptr2[3]; out[15] = ptr3[3];
      // Row 4
      out[16] = ptr0[4]; out[17] = ptr1[4]; out[18] = ptr2[4]; out[19] = ptr3[4];
      // Row 5
      out[20] = ptr0[5]; out[21] = ptr1[5]; out[22] = ptr2[5]; out[23] = ptr3[5];
      // Row 6
      out[24] = ptr0[6]; out[25] = ptr1[6]; out[26] = ptr2[6]; out[27] = ptr3[6];
      // Row 7
      out[28] = ptr0[7]; out[29] = ptr1[7]; out[30] = ptr2[7]; out[31] = ptr3[7];
      
      out += 32;
    }
    for (; kk < k; ++kk) {
      IFLOAT *ptr = block_base + kk * lda;
      out[0] = ptr[0];
      out[1] = ptr[1];
      out[2] = ptr[2];
      out[3] = ptr[3];
      out[4] = ptr[4];
      out[5] = ptr[5];
      out[6] = ptr[6];
      out[7] = ptr[7];
      out += 8;
    }
  }

  if (rem) {
      BLASLONG ib = m8 * 8;
      IFLOAT *out = dst + m8 * (k * 8);
      for (BLASLONG r = 0; r < rem; ++r) {
          // Row (ib + r) is at src + (ib+r) + k*lda
          IFLOAT *row_ptr = src + (ib + r);
          for (BLASLONG kk = 0; kk < k; ++kk) {
              *out++ = *(row_ptr + kk * lda);
          }
      }
  }

  return 0;
}