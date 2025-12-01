/***************************************************************************
 * AmpereOne BF16 GEMM 8x4 micro-kernel (bf16 x bf16 -> fp32 accumulate).
 * Layout expectations:
 *  - Packed B (pb): for each block of 4 columns, K is padded to 4.
 *    For kk in 0..K-1 step 4, store 4 bf16 from col0, then col1, col2, col3.
 *  - Packed A (pa): rows are contiguous, padded to 4 in K, no interleave.
 * Accumulators use NEON vbfdot (Armv8.6+).
 ***************************************************************************/

#include "common.h"
#include <arm_neon.h>

static inline float bf16_to_float(uint16_t h) {
  union {
    uint32_t u;
    float f;
  } v;
  v.u = ((uint32_t)h) << 16;
  return v.f;
}

int CNAME(BLASLONG m, BLASLONG n, BLASLONG k, FLOAT alpha_in,
          IFLOAT *packA, IFLOAT *packB, FLOAT *C, BLASLONG ldc) {
  const BLASLONG nb4 = n >> 2;          // blocks of 4 cols
  const BLASLONG mb8 = m >> 3;          // blocks of 8 rows
  const BLASLONG rem_m = m & 7;
  const BLASLONG rem_n = n & 3;

  float32x4_t alpha = vdupq_n_f32(alpha_in);

  for (BLASLONG jb = 0; jb < nb4; ++jb) {
    IFLOAT *pb_block = packB + jb * (k * 4);
    for (BLASLONG ib = 0; ib < mb8; ++ib) {
      IFLOAT *pa = packA + ib * (k * 8);
      IFLOAT *pb = pb_block;
      FLOAT *pc = C + (jb * 4) * ldc + ib * 8;

      float32x4_t acc01_r0 = vdupq_n_f32(0);
      float32x4_t acc23_r0 = vdupq_n_f32(0);
      float32x4_t acc01_r1 = vdupq_n_f32(0);
      float32x4_t acc23_r1 = vdupq_n_f32(0);
      float32x4_t acc01_r2 = vdupq_n_f32(0);
      float32x4_t acc23_r2 = vdupq_n_f32(0);
      float32x4_t acc01_r3 = vdupq_n_f32(0);
      float32x4_t acc23_r3 = vdupq_n_f32(0);
      float32x4_t acc01_r4 = vdupq_n_f32(0);
      float32x4_t acc23_r4 = vdupq_n_f32(0);
      float32x4_t acc01_r5 = vdupq_n_f32(0);
      float32x4_t acc23_r5 = vdupq_n_f32(0);
      float32x4_t acc01_r6 = vdupq_n_f32(0);
      float32x4_t acc23_r6 = vdupq_n_f32(0);
      float32x4_t acc01_r7 = vdupq_n_f32(0);
      float32x4_t acc23_r7 = vdupq_n_f32(0);

      for (BLASLONG kk = 0; kk < k; kk += 4, pb += 16) {
        bfloat16x8_t b01 = vld1q_bf16((const bfloat16_t *)pb);      // col0/1
        bfloat16x8_t b23 = vld1q_bf16((const bfloat16_t *)pb + 8);  // col2/3

        // macro to process one row
#define DOT_ROW(pa_row, acc01, acc23)                         \
        {                                                     \
          bfloat16x4_t a4 = vld1_bf16((const bfloat16_t *)(pa_row + kk));           \
          bfloat16x8_t a8 = vcombine_bf16(a4, a4);            \
          acc01 = vbfdotq_f32(acc01, a8, b01);                \
          acc23 = vbfdotq_f32(acc23, a8, b23);                \
        }

        DOT_ROW(pa,           acc01_r0, acc23_r0);
        DOT_ROW(pa + k,       acc01_r1, acc23_r1);
        DOT_ROW(pa + k * 2,   acc01_r2, acc23_r2);
        DOT_ROW(pa + k * 3,   acc01_r3, acc23_r3);
        DOT_ROW(pa + k * 4,   acc01_r4, acc23_r4);
        DOT_ROW(pa + k * 5,   acc01_r5, acc23_r5);
        DOT_ROW(pa + k * 6,   acc01_r6, acc23_r6);
        DOT_ROW(pa + k * 7,   acc01_r7, acc23_r7);
#undef DOT_ROW
      }

      // Store accumulators (only lanes 0/1 carry cols 0/1 and 2/3)
      float tmp[4];
#define STORE_ROW(pc_row, acc01, acc23)                                     \
      {                                                                     \
        float32x2_t c01 = vget_low_f32(acc01);                              \
        float32x2_t c23 = vget_low_f32(acc23);                              \
        float32x4_t cvec = vmulq_f32(vcombine_f32(c01, c23), alpha);        \
        vst1q_f32(tmp, cvec);                                               \
        pc_row[0] += tmp[0]; pc_row[ldc] += tmp[1];                         \
        pc_row[2 * ldc] += tmp[2]; pc_row[3 * ldc] += tmp[3];               \
      }

      STORE_ROW(pc,           acc01_r0, acc23_r0);
      STORE_ROW(pc + 1,       acc01_r1, acc23_r1);
      STORE_ROW(pc + 2,       acc01_r2, acc23_r2);
      STORE_ROW(pc + 3,       acc01_r3, acc23_r3);
      STORE_ROW(pc + 4,       acc01_r4, acc23_r4);
      STORE_ROW(pc + 5,       acc01_r5, acc23_r5);
      STORE_ROW(pc + 6,       acc01_r6, acc23_r6);
      STORE_ROW(pc + 7,       acc01_r7, acc23_r7);
#undef STORE_ROW
    }

    // remaining rows (<8) slow path
    if (rem_m) {
      BLASLONG i0 = mb8 * 8;
      for (BLASLONG ir = 0; ir < rem_m; ++ir) {
        IFLOAT *pa = packA + (i0 + ir) * k;
        IFLOAT *pb = pb_block;
        FLOAT *pc = C + (jb * 4) * ldc + (i0 + ir);

        float32x4_t acc01 = vdupq_n_f32(0);
        float32x4_t acc23 = vdupq_n_f32(0);

        for (BLASLONG kk = 0; kk < k; kk += 4, pb += 16) {
          bfloat16x8_t b01 = vld1q_bf16((const bfloat16_t *)pb);
          bfloat16x8_t b23 = vld1q_bf16((const bfloat16_t *)pb + 8);
          bfloat16x4_t a4 = vld1_bf16((const bfloat16_t *)(pa + kk));
          bfloat16x8_t a8 = vcombine_bf16(a4, a4);
          acc01 = vbfdotq_f32(acc01, a8, b01);
          acc23 = vbfdotq_f32(acc23, a8, b23);
        }

        float32x2_t c01 = vget_low_f32(acc01);
        float32x2_t c23 = vget_low_f32(acc23);
        float32x4_t cvec = vmulq_f32(vcombine_f32(c01, c23), alpha);
        float tmp4[4];
        vst1q_f32(tmp4, cvec);
        pc[0] += tmp4[0]; pc[ldc] += tmp4[1];
        pc[2 * ldc] += tmp4[2]; pc[3 * ldc] += tmp4[3];
      }
    }
  }

  // remaining cols (<4)
  if (rem_n) {
    BLASLONG jb = nb4 * 4;
    IFLOAT *pb_block = packB + nb4 * (k * 4);
    for (BLASLONG col = 0; col < rem_n; ++col) {
      for (BLASLONG i = 0; i < m; ++i) {
        IFLOAT *pb = pb_block + col * k;
        IFLOAT *pa = packA + i * k;
        FLOAT *pc = C + (jb + col) * ldc + i;
        float acc = 0.f;
        for (BLASLONG kk = 0; kk < k; kk += 4, pb += 4) {
          uint16_t *b16 = (uint16_t *)(pb);
          uint16_t *a16 = (uint16_t *)(pa + kk);
          acc += bf16_to_float(b16[0]) * bf16_to_float(a16[0]);
          acc += bf16_to_float(b16[1]) * bf16_to_float(a16[1]);
          acc += bf16_to_float(b16[2]) * bf16_to_float(a16[2]);
          acc += bf16_to_float(b16[3]) * bf16_to_float(a16[3]);
        }
        pc[0] += acc * alpha_in;
      }
    }
  }

  return 0;
}
