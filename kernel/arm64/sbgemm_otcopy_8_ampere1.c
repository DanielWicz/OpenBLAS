#define BFLOAT16
#include "common.h"

/* Wrapper so the build can generate *_otcopy from the shared oncopy
 * implementation. The build system sets CNAME to sbgemm_otcopy or
 * bgemm_otcopy as appropriate. */
#ifdef BGEMM
extern int bgemm_oncopy(BLASLONG m, BLASLONG n, IFLOAT *src, BLASLONG ldb, IFLOAT *dst);
#define ONCOPY_NAME bgemm_oncopy
#else
extern int sbgemm_oncopy(BLASLONG m, BLASLONG n, IFLOAT *src, BLASLONG ldb, IFLOAT *dst);
#define ONCOPY_NAME sbgemm_oncopy
#endif

int CNAME(BLASLONG m, BLASLONG n, IFLOAT *src, BLASLONG ldb, IFLOAT *dst) {
  return ONCOPY_NAME(m, n, src, ldb, dst);
}
