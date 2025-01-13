#ifndef CONST_H
#define CONST_H

#if defined(__AVX__)
#define SIMD_BATCH_SIZE 32
#elif defined(__SSE2__)
#define SIMD_BATCH_SIZE 16
#else
#define SIMD_BATCH_SIZE 1
#endif // End of architecture checks

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

#endif // CONST_H
