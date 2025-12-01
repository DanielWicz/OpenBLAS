/***************************************************************************
 * Pack A for AmpereOne BF16 GEMM.
 * Input: A (M x K), column-major.
 * Output: rows contiguous, K padded to 4. No interleave across rows.
 ***************************************************************************/
#include "common.h"

int CNAME(BLASLONG m, BLASLONG k, IFLOAT *src, BLASLONG lda, IFLOAT *dst) {
  for (BLASLONG i = 0; i < m; ++i) {
    IFLOAT *row = src + i;
    IFLOAT *out = dst + i * k;
    BLASLONG kk = 0;
    for (; kk + 3 < k; kk += 4) {
      out[kk + 0] = row[(kk + 0) * lda];
      out[kk + 1] = row[(kk + 1) * lda];
      out[kk + 2] = row[(kk + 2) * lda];
      out[kk + 3] = row[(kk + 3) * lda];
    }
    for (; kk < k; ++kk) {
      out[kk] = row[kk * lda];
    }
  }
  return 0;
}
