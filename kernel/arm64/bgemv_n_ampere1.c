/***************************************************************************
 * Simple bf16 GEMV (NoTrans) for AmpereOne (NEON bf16). Scalar fallback to
 * ensure correct types for BGEMV when SVE is unavailable.
 ***************************************************************************/
#include "common.h"

static inline float bf16_to_float(uint16_t h) {
  union { uint32_t u; float f; } v;
  v.u = ((uint32_t)h) << 16;
  return v.f;
}

static inline bfloat16 float_to_bf16(float x) {
  union { float f; uint32_t u; } v = { x };
  return (bfloat16)(v.u >> 16);
}

int CNAME(BLASLONG m, BLASLONG n, bfloat16 alpha, bfloat16 *a, BLASLONG lda,
          bfloat16 *x, BLASLONG incx, bfloat16 beta, bfloat16 *y, BLASLONG incy) {
  float alpha_f = bf16_to_float(alpha);
  float beta_f  = bf16_to_float(beta);

  for (BLASLONG col = 0; col < n; ++col) {
    bfloat16 *acol = a + col * lda;
    bfloat16 x_bf = x[col * incx];
    float x_f = bf16_to_float(x_bf);

    for (BLASLONG row = 0; row < m; ++row) {
      float a_f = bf16_to_float(acol[row]);
      float y_old = bf16_to_float(y[row * incy]);
      float y_new = alpha_f * a_f * x_f + beta_f * y_old;
      y[row * incy] = float_to_bf16(y_new);
    }
  }
  return 0;
}
