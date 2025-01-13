#ifndef DCT_H
#define DCT_H
#include <math.h>

// Integers rounded to x` = x + 0.5 * 2^7 for rounding

#define PI 3.14159265358979323846
#define SHIFT 10
#define FAC (1 << SHIFT)         // 2^SHIFT
#define DELTA (1 << (SHIFT - 1)) // 2^(SHIFT - 1)

#define THETA (PI / 16.0)
#define C0 ((int)(1 / sqrt(2) * FAC))
#define C1 ((int)(cos(THETA) / 2 * FAC))
#define C2 ((int)(cos(THETA * 2) / 2 * FAC))
#define C3 ((int)(cos(THETA * 3) / 2 * FAC))
#define C4 ((int)(cos(THETA * 4) / 2 * FAC))
#define C5 ((int)(cos(THETA * 5) / 2 * FAC))
#define C6 ((int)(cos(THETA * 6) / 2 * FAC))
#define C7 ((int)(cos(THETA * 7) / 2 * FAC))

static const int quant_matrix_[8][8] = {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 110, 99},
    {72, 92, 95, 98, 112, 100, 103, 99}};

typedef struct
{
    unsigned char *x;
    unsigned char *y;
    int height;
    int width;
    int channels;
    int block_y;
    int block_x;
} DCTThreadData;

// 8x8 butterfly DFT implementation
void dct_ld(unsigned char *X, unsigned char *Y, int height, int width, int channel);
void dct_ld_threaded(unsigned char *X, unsigned char *Y, int height, int width, int channel, int n_threads);
void dct_ld_1d(int vec[8]);
void idct_ld(unsigned char *X, unsigned char *Y, int height, int width, int channel);
void idct_ld_threaded(unsigned char *Y, unsigned char *X, int height, int width, int channel, int n_threads);
#endif // DCT_H