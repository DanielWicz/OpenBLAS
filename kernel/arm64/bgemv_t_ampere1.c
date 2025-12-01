/***************************************************************************
 * Simple bf16 GEMV (Transpose) for AmpereOne (NEON bf16).
 ***************************************************************************/
#include "common.h"

static inline float bf16_to_float(uint16_t h) {
  union { uint32_t u; float f; } v;
  v.u = ((uint32_t)h) << 16;
  return v.f;
}

/* round-to-nearest-even to match reference helpers */
static inline bfloat16 float_to_bf16(float x) {
  union { float f; uint32_t u; } v = { x };
  uint32_t lsb = (v.u >> 16) & 1;
  uint32_t rounding_bias = 0x7FFF + lsb;
  v.u += rounding_bias;
  return (bfloat16)(v.u >> 16);
}

int CNAME(BLASLONG m, BLASLONG n, bfloat16 alpha, bfloat16 *a, BLASLONG lda,
          bfloat16 *x, BLASLONG incx, bfloat16 beta, bfloat16 *y, BLASLONG incy) {
  float alpha_f = bf16_to_float(alpha);
  float beta_f  = bf16_to_float(beta);

  for (BLASLONG row = 0; row < m; ++row) {
    float acc = 0.f;
    for (BLASLONG col = 0; col < n; ++col) {
      float a_f = bf16_to_float(a[col * lda + row]);
      float x_f = bf16_to_float(x[col * incx]);
      acc += a_f * x_f;
    }
    float y_old = bf16_to_float(y[row * incy]);
    float y_new = alpha_f * acc + beta_f * y_old;
    y[row * incy] = float_to_bf16(y_new);
  }
  return 0;
}
