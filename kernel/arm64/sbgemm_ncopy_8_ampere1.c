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

static inline float bf16_to_float_local(uint16_t h) {
  union { uint32_t u; float f; } v;
  v.u = ((uint32_t)h) << 16;
  return v.f;
}
#define bf16_to_float bf16_to_float_local

int CNAME(BLASLONG m, BLASLONG n, IFLOAT *src, BLASLONG ldb, IFLOAT *dst) {
  static int debug_print = 0;
  static int tail_print = 0;
  if (!debug_print) {
      BLASLONG k = m;
      BLASLONG n4 = n >> 2;
      BLASLONG rem = n & 3;
      printf("NCOPY (ncopy file) called m=%ld n=%ld ldb=%ld k=%ld n4=%ld rem_n=%ld dst=%p src=%p\n",
             m, n, ldb, k, n4, rem, dst, src);
      debug_print = 1;
  }
  
  BLASLONG k = m;
  BLASLONG n4 = n >> 2;
  BLASLONG rem = n & 3;

  for (BLASLONG jb = 0; jb < n4; ++jb) {
    IFLOAT *col0 = src + jb * 4 * ldb;
    IFLOAT *col1 = col0 + ldb;
    IFLOAT *col2 = col1 + ldb;
    IFLOAT *col3 = col2 + ldb;
    IFLOAT *out = dst + jb * (k * 4);

    for (BLASLONG kk = 0; kk < k; kk += 2) {
      if (m==2 && n==2) {
          printf("NCOPY_MAIN: jb=%ld kk=%ld col0_val=%f col1_val=%f col2_val=%f col3_val=%f\n",
                 jb, kk, bf16_to_float(*(uint16_t*)&col0[kk]), bf16_to_float(*(uint16_t*)&col1[kk]),
                 bf16_to_float(*(uint16_t*)&col2[kk]), bf16_to_float(*(uint16_t*)&col3[kk]));
      }
      // Pack pairs of K for each column
      // B(k, 0), B(k+1, 0)
      out[0] = col0[kk];
      out[1] = (kk + 1 < k) ? col0[kk + 1] : 0;
      
      // B(k, 1), B(k+1, 1)
      out[2] = col1[kk];
      out[3] = (kk + 1 < k) ? col1[kk + 1] : 0;
      
      // B(k, 2), B(k+1, 2)
      out[4] = col2[kk];
      out[5] = (kk + 1 < k) ? col2[kk + 1] : 0;
      
      // B(k, 3), B(k+1, 3)
      out[6] = col3[kk];
      out[7] = (kk + 1 < k) ? col3[kk + 1] : 0;
      
      out += 8;
    }
  }

  if (rem) {
    if (!tail_print) {
        printf("NCOPY tail path triggered rem_n=%ld k=%ld dst=%p src=%p\n", rem, k, dst, src);
        tail_print = 1;
    }
    BLASLONG jb = n4 * 4;
    IFLOAT *out = dst + n4 * (k * 4);
    for (BLASLONG col = 0; col < rem; ++col) {
      IFLOAT *cptr = src + (jb + col) * ldb;
      for (BLASLONG kk = 0; kk < k; ++kk) {
          if (m==2 && n==2 && k==2) { // 2x2x2 case
              printf("NCOPY_TAIL_REM: col=%ld kk=%ld ldb=%ld src_base=%p addr=%p val=%f out_addr=%p\n", 
                     col, kk, ldb, src, (cptr + kk), bf16_to_float(*(uint16_t*)&cptr[kk]), out);
          }
        *out++ = cptr[kk];
      }
    }
  }
  return 0;
}
