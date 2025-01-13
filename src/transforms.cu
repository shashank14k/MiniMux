#include "transforms.cuh"
#include "const.h"

__global__ void rgb_to_yuv_ld(unsigned char *in_buf, unsigned char *out_buf, size_t tot_pixels)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int rgb_idx = tid * 3;
    if (rgb_idx < tot_pixels)
    {
        // in_buf is rgb
        float Y = 0.299 * in_buf[rgb_idx] + 0.587 * in_buf[rgb_idx + 1] + 0.114 * in_buf[rgb_idx + 2];
        float U = 0.492 * (in_buf[rgb_idx + 2] - Y);
        float V = 0.877 * (in_buf[rgb_idx] - Y);
        out_buf[rgb_idx] = (unsigned char)Y;
        out_buf[rgb_idx + 1] = (unsigned char)U;
        out_buf[rgb_idx + 2] = (unsigned char)V;
    }
}

__global__ void convert_rgb_to_gray(unsigned char *in_buf, unsigned char *out_buf, int tot_pixels)
{
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid_x < tot_pixels)
    {
        int rgb_idx = tid_x * 3;
        unsigned char r = in_buf[rgb_idx];
        unsigned char g = in_buf[rgb_idx + 1];
        unsigned char b = in_buf[rgb_idx + 2];
        out_buf[tid_x] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

void rgb2gray_cuda(unsigned char *in_buf, unsigned char *out_buf, size_t tot_pixels)
{
    size_t out_pixels = tot_pixels / 3;
    int req_blocks = (out_pixels + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

    unsigned char *out_buf_cu, *in_buf_cu;

    cudaMalloc(&out_buf_cu, out_pixels * sizeof(unsigned char));
    cudaMalloc(&in_buf_cu, tot_pixels * sizeof(unsigned char));

    cudaMemcpy(in_buf_cu, in_buf, tot_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threads(MAX_THREADS_PER_BLOCK, 1);
    dim3 blocks(req_blocks, 1);

    convert_rgb_to_gray<<<blocks, threads>>>(in_buf_cu, out_buf_cu, out_pixels);
    cudaMemcpy(out_buf, out_buf_cu, out_pixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}

void ldframe_rgb_to_yuv_cuda(LDframe *rgbFrame, LDframe *yuvFrame)
{

    size_t tot_pixels = rgbFrame->height * rgbFrame->width * rgbFrame->channels;
    unsigned char *cu_inbuf;
    cudaMalloc(&cu_inbuf, tot_pixels * sizeof(unsigned char));
    // cudaMalloc(&cu_outbuf, tot_pixels * sizeof(unsigned char));
    cudaMemcpy(cu_inbuf, rgbFrame->buf, tot_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int num_blocks = max(1, (tot_pixels + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK);
    rgb_to_yuv_ld<<<num_blocks, MAX_THREADS_PER_BLOCK>>>(cu_inbuf, cu_inbuf, tot_pixels);

    cudaMemcpy(cu_inbuf, yuvFrame->buf, tot_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaFree(cu_inbuf);
}