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
#include <stdio.h>

static inline float bf16_to_float_local(uint16_t h) {
  union { uint32_t u; float f; } v;
  v.u = ((uint32_t)h) << 16;
  return v.f;
}
#define bf16_to_float bf16_to_float_local

int CNAME(BLASLONG m, BLASLONG n, IFLOAT *src, BLASLONG lda, IFLOAT *dst) {
  static int debug_print = 0;
  if (!debug_print) {
      printf("ITCOPY (itcopy file) called m=%ld n=%ld lda=%ld\n", m, n, lda);
      debug_print = 1;
  }
  
  BLASLONG k = m;
  BLASLONG n4 = n >> 2;
  BLASLONG rem = n & 3;

  for (BLASLONG jb = 0; jb < n4; ++jb) {
    IFLOAT *out = dst + jb * (k * 4);
    
    for (BLASLONG kk = 0; kk < k; kk += 2) {
      IFLOAT *ptr = src + jb * 4 + kk * lda;
      IFLOAT *ptr_next = ptr + lda;
      
      if (m==2 && n==2) {
          printf("ITCOPY_MAIN: jb=%ld kk=%ld ptr_val=%f ptr_next_val=%f\n",
                 jb, kk, bf16_to_float(*(uint16_t*)&ptr[0]), bf16_to_float(*(uint16_t*)&ptr_next[0]));
      }

      // B(k, 0), B(k+1, 0)
      out[0] = ptr[0];
      out[1] = (kk + 1 < k) ? ptr_next[0] : 0;
      
      // B(k, 1), B(k+1, 1)
      out[2] = ptr[1];
      out[3] = (kk + 1 < k) ? ptr_next[1] : 0;
      
      // B(k, 2), B(k+1, 2)
      out[4] = ptr[2];
      out[5] = (kk + 1 < k) ? ptr_next[2] : 0;
      
      // B(k, 3), B(k+1, 3)
      out[6] = ptr[3];
      out[7] = (kk + 1 < k) ? ptr_next[3] : 0;
      
      out += 8;
    }
  }

  if (rem) {
      BLASLONG jb = n4 * 4;
      IFLOAT *out = dst + n4 * (k * 4);
      for (BLASLONG c = 0; c < rem; ++c) {
          IFLOAT *col_ptr = src + (jb + c);
          for (BLASLONG kk = 0; kk < k; ++kk) {
              if (m==2 && n==2 && k==2) { // 2x2x2 case
                  printf("ITCOPY_TAIL_REM: col=%ld kk=%ld lda=%ld src_base=%p addr=%p val=%f out_addr=%p\n", 
                         c, kk, lda, src, (col_ptr + kk * lda), bf16_to_float(*(uint16_t*)&(col_ptr[kk * lda])), out);
              }
              *out++ = *(col_ptr + kk * lda);
          }
      }
  }

  return 0;
}
