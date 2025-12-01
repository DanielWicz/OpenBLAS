/***************************************************************************
 * Pack A for AmpereOne BF16 GEMM (8 rows).
 * Input: A (M x K).
 * Output layout (per 8-row block):
 *  for kk in 0..K-1 step 4:
 *    store A[row0, kk..kk+3], A[row1, kk..kk+3] ... A[row7, kk..kk+3].
 *    (Total 32 bf16s per block).
 *  Tail K:
 *    store A[row0, kk], A[row1, kk] ... A[row7, kk].
 ***************************************************************************/
#include "common.h"

int CNAME(BLASLONG m, BLASLONG k, IFLOAT *src, BLASLONG lda, IFLOAT *dst) {
  BLASLONG m8 = m >> 3;
  BLASLONG rem = m & 7;

  for (BLASLONG ib = 0; ib < m8; ++ib) {
    IFLOAT *row0 = src + ib * 8 * lda;
    IFLOAT *row1 = row0 + lda;
    IFLOAT *row2 = row1 + lda;
    IFLOAT *row3 = row2 + lda;
    IFLOAT *row4 = row3 + lda;
    IFLOAT *row5 = row4 + lda;
    IFLOAT *row6 = row5 + lda;
    IFLOAT *row7 = row6 + lda;
    IFLOAT *out = dst + ib * (k * 8);

    BLASLONG kk = 0;
    for (; kk + 3 < k; kk += 4) {
      // Row 0
      out[0] = row0[kk+0]; out[1] = row0[kk+1]; out[2] = row0[kk+2]; out[3] = row0[kk+3];
      // Row 1
      out[4] = row1[kk+0]; out[5] = row1[kk+1]; out[6] = row1[kk+2]; out[7] = row1[kk+3];
      // Row 2
      out[8] = row2[kk+0]; out[9] = row2[kk+1]; out[10] = row2[kk+2]; out[11] = row2[kk+3];
      // Row 3
      out[12] = row3[kk+0]; out[13] = row3[kk+1]; out[14] = row3[kk+2]; out[15] = row3[kk+3];
      // Row 4
      out[16] = row4[kk+0]; out[17] = row4[kk+1]; out[18] = row4[kk+2]; out[19] = row4[kk+3];
      // Row 5
      out[20] = row5[kk+0]; out[21] = row5[kk+1]; out[22] = row5[kk+2]; out[23] = row5[kk+3];
      // Row 6
      out[24] = row6[kk+0]; out[25] = row6[kk+1]; out[26] = row6[kk+2]; out[27] = row6[kk+3];
      // Row 7
      out[28] = row7[kk+0]; out[29] = row7[kk+1]; out[30] = row7[kk+2]; out[31] = row7[kk+3];
      out += 32;
    }
    for (; kk < k; ++kk) {
      out[0] = row0[kk];
      out[1] = row1[kk];
      out[2] = row2[kk];
      out[3] = row3[kk];
      out[4] = row4[kk];
      out[5] = row5[kk];
      out[6] = row6[kk];
      out[7] = row7[kk];
      out += 8;
    }
  }

  // Leftover rows are not handled by this routine usually?
  // OpenBLAS usually handles the edge cases by calling smaller kernels or generic copy?
  // But ONCOPY usually needs to handle the full range?
  // No, the kernel will handle the main blocks. The copy function typically handles strict blocking?
  // Wait, if rem > 0, we still need to pack them?
  // Generic sgemm_ncopy handles remainders.
  // But let's implement it for completeness if OpenBLAS calls it with odd M.
  
  if (rem) {
      BLASLONG ib = m8 * 8;
      IFLOAT *out = dst + m8 * (k * 8);
      // For remainder, we just pack them sequentially or as 1-row strips?
      // The kernel remainder loop expects: 
      // "remaining rows (<8) slow path".
      // It accesses `packA + (i0 + ir) * k`.
      // This implies for the tail, the kernel expects NON-interleaved, simple contiguous rows.
      // So we just copy them row by row.
      
      for (BLASLONG r = 0; r < rem; ++r) {
          IFLOAT *row = src + (ib + r) * lda;
          for (BLASLONG kk = 0; kk < k; ++kk) {
              *out++ = row[kk];
          }
      }
  }

  return 0;
}
