/***************************************************************************
 * Pack A for AmpereOne BF16 GEMM (8 rows).
 * Case: A is Transposed. Input is K x M.
 * We want 8 rows of Op(A) -> 8 cols of Input.
 * Input Col i is at: src + i * lda. Elements contiguous.
 * Output layout (Interleaved K=4):
 *  Row0[0..3], Row1[0..3] ...
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
  if (!debug_print) {
      printf("OTCOPY (otcopy file) called m=%ld n=%ld lda=%ld\n", m, n, ldb);
      debug_print = 1;
  }
  
  BLASLONG k = n;
  BLASLONG lda = ldb;

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
      if (m==2 && n==2) {
          printf("OTCOPY_MAIN: ib=%ld kk=%ld col0_val=%f col1_val=%f col2_val=%f col3_val=%f\n",
                 ib, kk, bf16_to_float(*(uint16_t*)&col0[kk]), bf16_to_float(*(uint16_t*)&col1[kk]),
                 bf16_to_float(*(uint16_t*)&col2[kk]), bf16_to_float(*(uint16_t*)&col3[kk]));
      }

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
      IFLOAT *ptr = col0 + kk * lda;
      out[0] = ptr[0];
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
              if (m==2 && n==2 && k==2) {
                  printf("OTCOPY_TAIL: r=%ld kk=%ld lda=%ld src_base=%p addr=%p val=%f out_addr=%p\n",
                         r, kk, lda, src, (col + kk), bf16_to_float(*(uint16_t*)&col[kk]), out);
              }
              *out++ = col[kk];
          }
      }
  }

  return 0;
}
