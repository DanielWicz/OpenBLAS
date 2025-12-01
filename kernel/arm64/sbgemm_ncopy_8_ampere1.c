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

int CNAME(BLASLONG m, BLASLONG n, IFLOAT *src, BLASLONG ldb, IFLOAT *dst) {
  // m is K (rows of B)
  // n is N (cols of B)
  // dst layout: Interleaved columns. 
  // For each K: B(k,0), B(k,1), B(k,2), B(k,3).
  
  BLASLONG k = m;
  BLASLONG n4 = n >> 2;
  BLASLONG rem = n & 3;

  for (BLASLONG jb = 0; jb < n4; ++jb) {
    IFLOAT *col0 = src + jb * 4 * ldb;
    IFLOAT *col1 = col0 + ldb;
    IFLOAT *col2 = col1 + ldb;
    IFLOAT *col3 = col2 + ldb;
    IFLOAT *out = dst + jb * (k * 4);

    for (BLASLONG kk = 0; kk < k; ++kk) {
      out[0] = col0[kk];
      out[1] = col1[kk];
      out[2] = col2[kk];
      out[3] = col3[kk];
      out += 4;
    }
  }

  if (rem) {
    BLASLONG jb = n4 * 4;
    IFLOAT *out = dst + n4 * (k * 4);
    // For remainder cols, we still pack K-major but pad with zeros or garbage?
    // The kernel handles remainder N via special loops? 
    // OpenBLAS usually packs sequentially for remainder, or pads.
    // My kernel 'rem_n' section iterates columns individually?
    // This implies Remainder Packing should be Sequential Columns (not interleaved).
    // Col0[0..k], Col1[0..k].
    
    for (BLASLONG col = 0; col < rem; ++col) {
      IFLOAT *cptr = src + (jb + col) * ldb;
      for (BLASLONG kk = 0; kk < k; ++kk) {
        *out++ = cptr[kk];
      }
    }
  }
  return 0;
}