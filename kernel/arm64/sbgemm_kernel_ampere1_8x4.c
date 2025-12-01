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

static inline bfloat16 float_to_bf16(float x) {
  union { float f; uint32_t u; } v = { x };
  return (bfloat16)(v.u >> 16);
}

int CNAME(BLASLONG m, BLASLONG n, BLASLONG k, FLOAT alpha_in,
          IFLOAT *packA, IFLOAT *packB, FLOAT *C, BLASLONG ldc) {
  const BLASLONG nb4 = n >> 2;          // blocks of 4 cols
  const BLASLONG mb8 = m >> 3;          // blocks of 8 rows
  const BLASLONG rem_m = m & 7;
  const BLASLONG rem_n = n & 3;

  float alpha_f = (float)alpha_in;
#ifdef BGEMM
  alpha_f = bf16_to_float((uint16_t)alpha_in);
#endif
  float32x4_t alpha = vdupq_n_f32(alpha_f);

  for (BLASLONG jb = 0; jb < nb4; ++jb) {
    IFLOAT *pb_block = packB + jb * (k * 4);
    for (BLASLONG ib = 0; ib < mb8; ++ib) {
      IFLOAT *pa = packA + ib * (k * 8);
      IFLOAT *pb = pb_block;
#ifdef BGEMM
      bfloat16 *pc = C + (jb * 4) * ldc + ib * 8;
#else
      float *pc = C + (jb * 4) * ldc + ib * 8;
#endif

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
        bfloat16x8_t b23 = vld1q_bf16(((const bfloat16_t *)pb) + 8);  // col2/3

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
      float32x2_t c01, c23;
      float32x4_t cvec;
      float out0, out1, out2, out3;

      // row 0..7
      c01 = vget_low_f32(acc01_r0); c23 = vget_low_f32(acc23_r0);
      cvec = vmulq_f32(vcombine_f32(c01, c23), alpha);
      out0 = vgetq_lane_f32(cvec, 0); out1 = vgetq_lane_f32(cvec, 1);
      out2 = vgetq_lane_f32(cvec, 2); out3 = vgetq_lane_f32(cvec, 3);
#ifdef BGEMM
      pc[0]             = float_to_bf16(out0);
      pc[ldc]           = float_to_bf16(out1);
      pc[2 * ldc]       = float_to_bf16(out2);
      pc[3 * ldc]       = float_to_bf16(out3);
#else
      pc[0]             += out0;
      pc[ldc]           += out1;
      pc[2 * ldc]       += out2;
      pc[3 * ldc]       += out3;
#endif

      c01 = vget_low_f32(acc01_r1); c23 = vget_low_f32(acc23_r1);
      cvec = vmulq_f32(vcombine_f32(c01, c23), alpha);
      out0 = vgetq_lane_f32(cvec, 0); out1 = vgetq_lane_f32(cvec, 1);
      out2 = vgetq_lane_f32(cvec, 2); out3 = vgetq_lane_f32(cvec, 3);
#ifdef BGEMM
      pc[1]             = float_to_bf16(out0);
      pc[1 + ldc]       = float_to_bf16(out1);
      pc[1 + 2 * ldc]   = float_to_bf16(out2);
      pc[1 + 3 * ldc]   = float_to_bf16(out3);
#else
      pc[1]             += out0;
      pc[1 + ldc]       += out1;
      pc[1 + 2 * ldc]   += out2;
      pc[1 + 3 * ldc]   += out3;
#endif

      c01 = vget_low_f32(acc01_r2); c23 = vget_low_f32(acc23_r2);
      cvec = vmulq_f32(vcombine_f32(c01, c23), alpha);
      out0 = vgetq_lane_f32(cvec, 0); out1 = vgetq_lane_f32(cvec, 1);
      out2 = vgetq_lane_f32(cvec, 2); out3 = vgetq_lane_f32(cvec, 3);
#ifdef BGEMM
      pc[2]             = float_to_bf16(out0);
      pc[2 + ldc]       = float_to_bf16(out1);
      pc[2 + 2 * ldc]   = float_to_bf16(out2);
      pc[2 + 3 * ldc]   = float_to_bf16(out3);
#else
      pc[2]             += out0;
      pc[2 + ldc]       += out1;
      pc[2 + 2 * ldc]   += out2;
      pc[2 + 3 * ldc]   += out3;
#endif

      c01 = vget_low_f32(acc01_r3); c23 = vget_low_f32(acc23_r3);
      cvec = vmulq_f32(vcombine_f32(c01, c23), alpha);
      out0 = vgetq_lane_f32(cvec, 0); out1 = vgetq_lane_f32(cvec, 1);
      out2 = vgetq_lane_f32(cvec, 2); out3 = vgetq_lane_f32(cvec, 3);
#ifdef BGEMM
      pc[3]             = float_to_bf16(out0);
      pc[3 + ldc]       = float_to_bf16(out1);
      pc[3 + 2 * ldc]   = float_to_bf16(out2);
      pc[3 + 3 * ldc]   = float_to_bf16(out3);
#else
      pc[3]             += out0;
      pc[3 + ldc]       += out1;
      pc[3 + 2 * ldc]   += out2;
      pc[3 + 3 * ldc]   += out3;
#endif

      c01 = vget_low_f32(acc01_r4); c23 = vget_low_f32(acc23_r4);
      cvec = vmulq_f32(vcombine_f32(c01, c23), alpha);
      out0 = vgetq_lane_f32(cvec, 0); out1 = vgetq_lane_f32(cvec, 1);
      out2 = vgetq_lane_f32(cvec, 2); out3 = vgetq_lane_f32(cvec, 3);
#ifdef BGEMM
      pc[4]             = float_to_bf16(out0);
      pc[4 + ldc]       = float_to_bf16(out1);
      pc[4 + 2 * ldc]   = float_to_bf16(out2);
      pc[4 + 3 * ldc]   = float_to_bf16(out3);
#else
      pc[4]             += out0;
      pc[4 + ldc]       += out1;
      pc[4 + 2 * ldc]   += out2;
      pc[4 + 3 * ldc]   += out3;
#endif

      c01 = vget_low_f32(acc01_r5); c23 = vget_low_f32(acc23_r5);
      cvec = vmulq_f32(vcombine_f32(c01, c23), alpha);
      out0 = vgetq_lane_f32(cvec, 0); out1 = vgetq_lane_f32(cvec, 1);
      out2 = vgetq_lane_f32(cvec, 2); out3 = vgetq_lane_f32(cvec, 3);
#ifdef BGEMM
      pc[5]             = float_to_bf16(out0);
      pc[5 + ldc]       = float_to_bf16(out1);
      pc[5 + 2 * ldc]   = float_to_bf16(out2);
      pc[5 + 3 * ldc]   = float_to_bf16(out3);
#else
      pc[5]             += out0;
      pc[5 + ldc]       += out1;
      pc[5 + 2 * ldc]   += out2;
      pc[5 + 3 * ldc]   += out3;
#endif

      c01 = vget_low_f32(acc01_r6); c23 = vget_low_f32(acc23_r6);
      cvec = vmulq_f32(vcombine_f32(c01, c23), alpha);
      out0 = vgetq_lane_f32(cvec, 0); out1 = vgetq_lane_f32(cvec, 1);
      out2 = vgetq_lane_f32(cvec, 2); out3 = vgetq_lane_f32(cvec, 3);
#ifdef BGEMM
      pc[6]             = float_to_bf16(out0);
      pc[6 + ldc]       = float_to_bf16(out1);
      pc[6 + 2 * ldc]   = float_to_bf16(out2);
      pc[6 + 3 * ldc]   = float_to_bf16(out3);
#else
      pc[6]             += out0;
      pc[6 + ldc]       += out1;
      pc[6 + 2 * ldc]   += out2;
      pc[6 + 3 * ldc]   += out3;
#endif

      c01 = vget_low_f32(acc01_r7); c23 = vget_low_f32(acc23_r7);
      cvec = vmulq_f32(vcombine_f32(c01, c23), alpha);
      out0 = vgetq_lane_f32(cvec, 0); out1 = vgetq_lane_f32(cvec, 1);
      out2 = vgetq_lane_f32(cvec, 2); out3 = vgetq_lane_f32(cvec, 3);
#ifdef BGEMM
      pc[7]             = float_to_bf16(out0);
      pc[7 + ldc]       = float_to_bf16(out1);
      pc[7 + 2 * ldc]   = float_to_bf16(out2);
      pc[7 + 3 * ldc]   = float_to_bf16(out3);
#else
      pc[7]             += out0;
      pc[7 + ldc]       += out1;
      pc[7 + 2 * ldc]   += out2;
      pc[7 + 3 * ldc]   += out3;
#endif
    }

    // remaining rows (<8) slow path
    if (rem_m) {
      BLASLONG i0 = mb8 * 8;
      for (BLASLONG ir = 0; ir < rem_m; ++ir) {
        IFLOAT *pa = packA + (i0 + ir) * k;
        IFLOAT *pb = pb_block;
#ifdef BGEMM
        bfloat16 *pc = C + (jb * 4) * ldc + (i0 + ir);
#else
        float *pc = C + (jb * 4) * ldc + (i0 + ir);
#endif

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
        float out0 = vgetq_lane_f32(cvec, 0);
        float out1 = vgetq_lane_f32(cvec, 1);
        float out2 = vgetq_lane_f32(cvec, 2);
        float out3 = vgetq_lane_f32(cvec, 3);
#ifdef BGEMM
        pc[0]       = float_to_bf16(out0);
        pc[ldc]     = float_to_bf16(out1);
        pc[2 * ldc] = float_to_bf16(out2);
        pc[3 * ldc] = float_to_bf16(out3);
#else
        pc[0]       += out0;
        pc[ldc]     += out1;
        pc[2 * ldc] += out2;
        pc[3 * ldc] += out3;
#endif
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
#ifdef BGEMM
        bfloat16 *pc = C + (jb + col) * ldc + i;
#else
        float *pc = C + (jb + col) * ldc + i;
#endif
        float acc = 0.f;
        for (BLASLONG kk = 0; kk < k; kk += 4, pb += 4) {
          uint16_t *b16 = (uint16_t *)(pb);
          uint16_t *a16 = (uint16_t *)(pa + kk);
          acc += bf16_to_float(b16[0]) * bf16_to_float(a16[0]);
          acc += bf16_to_float(b16[1]) * bf16_to_float(a16[1]);
          acc += bf16_to_float(b16[2]) * bf16_to_float(a16[2]);
          acc += bf16_to_float(b16[3]) * bf16_to_float(a16[3]);
        }
#ifdef BGEMM
        pc[0] = float_to_bf16(acc * alpha_f);
#else
        pc[0] += acc * alpha_f;
#endif
      }
    }
  }

  return 0;
}
