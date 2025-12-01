/* AmpereOne-tuned SGEMM micro-kernel.
 * We reuse the proven 16x4 NEON kernel and retune prefetch distances
 * for AmpereOne's 2MB private L2 + 64KB write-through L1D.
 */
#define A_PRE_SIZE 3072
#define B_PRE_SIZE 512
#define C_PRE_SIZE 256
#include "sgemm_kernel_16x4.S"
