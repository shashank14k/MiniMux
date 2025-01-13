#ifndef TRANSFORMS_CUH
#define TRANSFORMS_CUH
#include "frame.h"
#include <cuda_runtime.h>
#define SHARED_MEM_SIZE 256
#define MAX_THREADS_PER_BLOCK 1024

__global__ void rgb_to_yuv_ld(unsigned char *in_buf, unsigned char *out_buf, size_t tot_pixels);
__global__ void convert_rgb_to_gray(unsigned char *in_buf, unsigned char *out_buf, int tot_pixels);

void rgb2gray_cuda(unsigned char *in_buf, unsigned char *out_buf, size_t tot_pixels);
void ldframe_rgb_to_yuv_cuda(LDframe *rgbFrame, LDframe *yuvFrame);

#endif // TRANSFORMS_CUH
