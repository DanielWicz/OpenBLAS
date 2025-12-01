/***************************************************************************
 * AmpereOne BF16 GEMM 8x4 micro-kernel (bf16 x bf16 -> fp32 accumulate).
 * Layout expectations:
 *  - Packed B (pb): for each block of 4 columns, K is padded to 4.
 *    For kk in 0..K-1 step 4, store 4 bf16 from col0, then col1, col2, col3.
 *  - Packed A (pa): Interleaved 8 rows.
 *    For kk in 0..K-1 step 4:
 *      Row0[kk..kk+3], Row1[kk..kk+3] ... Row7[kk..kk+3]
 ***************************************************************************/

#define BFLOAT16
#define SBGEMM
#include "common.h"
#include <stdio.h>
#include <arm_neon.h>

static inline float bf16_to_float(uint16_t h) {
  union {
    uint32_t u;
    float f;
  } v;
  v.u = ((uint32_t)h) << 16;
  return v.f;
}

/* Round-to-nearest-even conversion */
static inline bfloat16 float_to_bf16(float x) {
  union { float f; uint32_t u; } v = { x };
  uint32_t lsb = (v.u >> 16) & 1;          /* current LSB in bf16 */
  uint32_t rounding_bias = 0x7FFF + lsb;   /* ties to even */
  v.u += rounding_bias;
  return (bfloat16)(v.u >> 16);
}

int CNAME(BLASLONG m, BLASLONG n, BLASLONG k, FLOAT alpha_in,
          IFLOAT *packA, IFLOAT *packB, FLOAT *C, BLASLONG ldc) {
  const BLASLONG nb4 = n >> 2;          // blocks of 4 cols
  const BLASLONG mb8 = m >> 3;          // blocks of 8 rows
  const BLASLONG rem_m = m & 7;
  const BLASLONG rem_n = n & 3;
  static int entry_print = 0;
  static int rem_m_print = 0;
  static int rem_n_print = 0;
  static int k_tail_print = 0;

  float alpha_f = (float)alpha_in;
#ifdef BGEMM
  alpha_f = bf16_to_float((uint16_t)alpha_in);
#endif
  float32x4_t alpha = vdupq_n_f32(alpha_f);

  if (!entry_print) {
    printf("AMP1 sbgemm kernel entry m=%ld n=%ld k=%ld ldc=%ld nb4=%ld mb8=%ld rem_m=%ld rem_n=%ld alpha=%g packA=%p packB=%p C=%p\n",
           m, n, k, ldc, nb4, mb8, rem_m, rem_n, (double)alpha_f, (void *)packA, (void *)packB, (void *)C);
    entry_print = 1;
  }
  if ((k & 3) && !k_tail_print) {
    printf("AMP1 sbgemm kernel K tail active (k mod 4 = %ld)\n", k & 3);
    k_tail_print = 1;
  }

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

      BLASLONG kk = 0;
      for (; kk + 3 < k; kk += 4, pb += 16, pa += 32) {
        bfloat16x8_t b01 = vld1q_bf16((const bfloat16_t *)pb);      // col0/1
        bfloat16x8_t b23 = vld1q_bf16(((const bfloat16_t *)pb) + 8);  // col2/3

        // macro to process one row (pa_offset is 0, 4, 8...)
#define DOT_ROW(pa_offset, acc01, acc23)                      \
        {                                                     \
          bfloat16x4_t a4 = vld1_bf16((const bfloat16_t *)(pa + pa_offset)); \
          bfloat16x8_t a8 = vcombine_bf16(a4, a4);            \
          acc01 = vbfdotq_f32(acc01, a8, b01);                \
          acc23 = vbfdotq_f32(acc23, a8, b23);                \
        }

        DOT_ROW(0,  acc01_r0, acc23_r0);
        DOT_ROW(4,  acc01_r1, acc23_r1);
        DOT_ROW(8,  acc01_r2, acc23_r2);
        DOT_ROW(12, acc01_r3, acc23_r3);
        DOT_ROW(16, acc01_r4, acc23_r4);
        DOT_ROW(20, acc01_r5, acc23_r5);
        DOT_ROW(24, acc01_r6, acc23_r6);
        DOT_ROW(28, acc01_r7, acc23_r7);
#undef DOT_ROW
      }
      
      for (; kk < k; ++kk, pb += 4, pa += 8) {
        float b0 = bf16_to_float(*(uint16_t*)&pb[0]);
        float b1 = bf16_to_float(*(uint16_t*)&pb[1]);
        float b2 = bf16_to_float(*(uint16_t*)&pb[2]);
        float b3 = bf16_to_float(*(uint16_t*)&pb[3]);
#define TAIL_FMA(acc01, acc23, offset)                        \
        {                                                     \
          float a_f = bf16_to_float(*(uint16_t*)(pa + offset));          \
          float tmp01[4];                                     \
          float tmp23[4];                                     \
          vst1q_f32(tmp01, acc01);                            \
          vst1q_f32(tmp23, acc23);                            \
          tmp01[0] += a_f * b0;                               \
          tmp01[2] += a_f * b1;                               \
          tmp23[0] += a_f * b2;                               \
          tmp23[2] += a_f * b3;                               \
          acc01 = vld1q_f32(tmp01);                           \
          acc23 = vld1q_f32(tmp23);                           \
        }
        TAIL_FMA(acc01_r0, acc23_r0, 0);
        TAIL_FMA(acc01_r1, acc23_r1, 1);
        TAIL_FMA(acc01_r2, acc23_r2, 2);
        TAIL_FMA(acc01_r3, acc23_r3, 3);
        TAIL_FMA(acc01_r4, acc23_r4, 4);
        TAIL_FMA(acc01_r5, acc23_r5, 5);
        TAIL_FMA(acc01_r6, acc23_r6, 6);
        TAIL_FMA(acc01_r7, acc23_r7, 7);
#undef TAIL_FMA
      }

      // Store accumulators
      float32x4_t cvec;
      float out0, out1, out2, out3;

      // helper macro
#define STORE_ROW(acc01, acc23, idx) \
      cvec = vpaddq_f32(acc01, acc23); \
      cvec = vmulq_f32( cvec, alpha); \
      out0 = vgetq_lane_f32( cvec, 0); out1 = vgetq_lane_f32( cvec, 1); \
      out2 = vgetq_lane_f32( cvec, 2); out3 = vgetq_lane_f32( cvec, 3); \
      { \
        BLASLONG off = idx; \
        IFLOAT *dst = (IFLOAT*)(pc + off); \
        dst[0]             = float_to_bf16(out0 + bf16_to_float(*(uint16_t*)&dst[0])); \
        dst[ldc]           = float_to_bf16(out1 + bf16_to_float(*(uint16_t*)&dst[ldc])); \
        dst[2 * ldc]       = float_to_bf16(out2 + bf16_to_float(*(uint16_t*)&dst[2 * ldc])); \
        dst[3 * ldc]       = float_to_bf16(out3 + bf16_to_float(*(uint16_t*)&dst[3 * ldc])); \
      }

      STORE_ROW(acc01_r0, acc23_r0, 0);
      STORE_ROW(acc01_r1, acc23_r1, 1);
      STORE_ROW(acc01_r2, acc23_r2, 2);
      STORE_ROW(acc01_r3, acc23_r3, 3);
      STORE_ROW(acc01_r4, acc23_r4, 4);
      STORE_ROW(acc01_r5, acc23_r5, 5);
      STORE_ROW(acc01_r6, acc23_r6, 6);
      STORE_ROW(acc01_r7, acc23_r7, 7);
#undef STORE_ROW
    }

    // remaining rows (<8) slow path
    if (rem_m) {
      if (!rem_m_print) {
        printf("AMP1 sbgemm kernel rem_m path used rem_m=%ld k=%ld mb8=%ld\n", rem_m, k, mb8);
        rem_m_print = 1;
      }
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

        BLASLONG kk = 0;
        for (; kk + 3 < k; kk += 4, pb += 16) {
          bfloat16x8_t b01 = vld1q_bf16((const bfloat16_t *)pb);
          bfloat16x8_t b23 = vld1q_bf16((const bfloat16_t *)pb + 8);
          bfloat16x4_t a4 = vld1_bf16((const bfloat16_t *)(pa + kk));
          bfloat16x8_t a8 = vcombine_bf16(a4, a4);
          acc01 = vbfdotq_f32(acc01, a8, b01);
          acc23 = vbfdotq_f32(acc23, a8, b23);
        }
        for (; kk < k; ++kk, pb += 4) {
          float b0 = bf16_to_float(*(uint16_t*)&pb[0]);
          float b1 = bf16_to_float(*(uint16_t*)&pb[1]);
          float b2 = bf16_to_float(*(uint16_t*)&pb[2]);
          float b3 = bf16_to_float(*(uint16_t*)&pb[3]);
          float a_f = bf16_to_float(*(uint16_t*)&pa[kk]);
          float tmp01[4], tmp23[4];
          vst1q_f32(tmp01, acc01);
          vst1q_f32(tmp23, acc23);
          tmp01[0] += a_f * b0; tmp01[2] += a_f * b1;
          tmp23[0] += a_f * b2; tmp23[2] += a_f * b3;
          acc01 = vld1q_f32(tmp01);
          acc23 = vld1q_f32(tmp23);
        }

        float32x4_t cvec = vpaddq_f32(acc01, acc23);
        cvec = vmulq_f32(cvec, alpha);
        float out0 = vgetq_lane_f32(cvec, 0);
        float out1 = vgetq_lane_f32( cvec, 1);
        float out2 = vgetq_lane_f32( cvec, 2);
        float out3 = vgetq_lane_f32( cvec, 3);
#ifdef BGEMM
        pc[0]       = float_to_bf16(out0 + bf16_to_float(*(uint16_t*)&pc[0]));
        pc[ldc]     = float_to_bf16(out1 + bf16_to_float(*(uint16_t*)&pc[ldc]));
        pc[2 * ldc] = float_to_bf16(out2 + bf16_to_float(*(uint16_t*)&pc[2 * ldc]));
        pc[3 * ldc] = float_to_bf16(out3 + bf16_to_float(*(uint16_t*)&pc[3 * ldc]));
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
    if (!rem_n_print) {
      printf("AMP1 sbgemm kernel rem_n path used rem_n=%ld k=%ld nb4=%ld\n", rem_n, k, nb4);
      rem_n_print = 1;
    }
    IFLOAT *pb_base = packB + nb4 * (k * 4);
    for (BLASLONG col = 0; col < rem_n; ++col) {
      IFLOAT *pb = pb_base + col * k;
      IFLOAT *pa = packA;
      
      // Process blocks of 8 rows
      for (BLASLONG ib = 0; ib < mb8; ++ib) {
        float acc[8] = {0,0,0,0, 0,0,0,0};
        IFLOAT *pb_ptr = pb;
        
        BLASLONG kk = 0;
        for (; kk + 3 < k; kk += 4) {
           float b0 = bf16_to_float(*(uint16_t*)&pb_ptr[0]);
           float b1 = bf16_to_float(*(uint16_t*)&pb_ptr[1]);
           float b2 = bf16_to_float(*(uint16_t*)&pb[2]);
           float b3 = bf16_to_float(*(uint16_t*)&pb[3]);
           
           for (int r = 0; r < 8; ++r) {
              float a0 = bf16_to_float(*(uint16_t*)&pa[r*4 + 0]);
              float a1 = bf16_to_float(*(uint16_t*)&pa[r*4 + 1]);
              float a2 = bf16_to_float(*(uint16_t*)&pa[r*4 + 2]);
              float a3 = bf16_to_float(*(uint16_t*)&pa[r*4 + 3]);
              acc[r] += a0*b0 + a1*b1 + a2*b2 + a3*b3;
           }
           pa += 32;
           pb_ptr += 4;
        }
        for (; kk < k; ++kk) {
           float b = bf16_to_float(*(uint16_t*)pb_ptr++);
           for (int r = 0; r < 8; ++r) {
              acc[r] += bf16_to_float(*(uint16_t*)&pa[r]) * b;
           }
           pa += 8;
        }
        
        // Store 8 rows
        BLASLONG j = nb4 * 4 + col;
        for (int r = 0; r < 8; ++r) {
           BLASLONG row = ib * 8 + r;
#ifdef BGEMM
           bfloat16 *pc = C + j * ldc + row;
           *pc = float_to_bf16(acc[r] * alpha_f + bf16_to_float(*(uint16_t*)pc));
#else
           float *pc = C + j * ldc + row;
           *pc += acc[r] * alpha_f;
#endif
        }
      }
      
      // remaining rows (<8) slow path
      if (rem_m) {
         BLASLONG row_base = mb8 * 8;
         for (BLASLONG r = 0; r < rem_m; ++r) {
            float acc = 0;
            IFLOAT *pb_ptr = pb;
            for (BLASLONG kk = 0; kk < k; ++kk) {
               float av = bf16_to_float(*(uint16_t*)pa++);
               float bv = bf16_to_float(*(uint16_t*)pb_ptr++);
               acc += av * bv;
            }
            BLASLONG row = row_base + r;
            BLASLONG j = nb4 * 4 + col;
#ifdef BGEMM
            bfloat16 *pc = C + j * ldc + row;
            *pc = float_to_bf16(acc * alpha_f + bf16_to_float(*(uint16_t*)pc));
#else
            float *pc = C + j * ldc + row;
            *pc += acc * alpha_f;
#endif
         }
      }
    }
  }

  return 0;
}
