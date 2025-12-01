/* AmpereOne-tuned DGEMM micro-kernel.
 * Based on the baseline 8x4 NEON kernel with prefetch distances
 * lengthened to keep AmpereOne's 2MB L2 fed without over-polluting
 * the write-through L1D.
 */
#define A_PRE_SIZE 3072
#define B_PRE_SIZE 640
#define C_PRE_SIZE 192
#include "dgemm_kernel_8x4.S"
