#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <immintrin.h>
#include <emmintrin.h>
#include "frame.h"
#include "const.h"
#include "transforms.cuh"

LDframe *allocate_ldframe(int height, int width, int channels, PIXEL_FMT pix_fmt)
{
    // assert(channels == 3);
    LDframe *ldFrame = (LDframe *)malloc(sizeof(LDframe));
    ldFrame->height = height;
    ldFrame->width = width;
    ldFrame->channels = channels;
    ldFrame->bufsize = height * width * channels * sizeof(unsigned char);
    unsigned char *buf = (unsigned char *)malloc(ldFrame->bufsize);
    if (buf == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for frame buffer\n");
        free(ldFrame);
        return NULL;
    }
    ldFrame->buf = buf;
    return ldFrame;
}

void ldframe_rgb_to_yuv(LDframe *rgbFrame, LDframe *yuvFrame)
{
    assert(rgbFrame->pix_fmt == PIXEL_FMT_RGB);
    if (SIMD_BATCH_SIZE == 8)
    {
        ldframe_rgb_to_yuv_avx(rgbFrame, yuvFrame);
    }
    else if (SIMD_BATCH_SIZE == 4)
    {
        ldframe_rgb_to_yuv_sse(rgbFrame, yuvFrame);
    }
    else
    {
        ldframe_rgb_to_yuv_cpu(rgbFrame, yuvFrame);
    }
}

void ldframe_rgb_to_yuv_cpu(LDframe *rgbFrame, LDframe *yuvFrame)
{
    assert(rgbFrame->pix_fmt == PIXEL_FMT_RGB);
    assert(rgbFrame->channels == 3);
    size_t frame_res = rgbFrame->height * rgbFrame->width;
    yuvFrame->pix_fmt = PIXEL_FMT_YUV;
    for (size_t i = 0; i < frame_res; i++)
    {
        size_t pix_loc = i * rgbFrame->channels;
        unsigned char r = rgbFrame->buf[pix_loc];
        unsigned char g = rgbFrame->buf[pix_loc + 1];
        unsigned char b = rgbFrame->buf[pix_loc + 2];

        yuvFrame->buf[pix_loc] = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
        yuvFrame->buf[pix_loc + 1] = (unsigned char)(-0.14713 * r - 0.28886 * g + 0.436 * b + 128);
        yuvFrame->buf[pix_loc + 2] = (unsigned char)(0.615 * r - 0.51499 * g - 0.10001 * b + 128);
    }
}

void ldframe_rgb_to_yuv_avx(LDframe *rgbFrame, LDframe *yuvFrame)
{
    assert(SIMD_BATCH_SIZE == 8);
    size_t frame_res = rgbFrame->height * rgbFrame->width;

    size_t i = 0;
    for (size_t i = 0; i + SIMD_BATCH_SIZE < frame_res; i += SIMD_BATCH_SIZE)
    {
        size_t pix_loc = i * rgbFrame->channels;

        // Load AVX registers
        __m256i r = _mm256_load_si256((__m256i *)&rgbFrame->buf[pix_loc]);
        __m256i g = _mm256_load_si256((__m256i *)&rgbFrame->buf[pix_loc + 1]);
        __m256i b = _mm256_load_si256((__m256i *)&rgbFrame->buf[pix_loc + 2]);

        // Convert to float
        __m256 r_f = _mm256_cvtepi32_ps(r);
        __m256 g_f = _mm256_cvtepi32_ps(g);
        __m256 b_f = _mm256_cvtepi32_ps(b);

        __m256 y_vals = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(r_f, _mm256_set1_ps(0.299f)),
                          _mm256_mul_ps(g_f, _mm256_set1_ps(0.587f))),
            _mm256_mul_ps(b_f, _mm256_set1_ps(0.114f)));
        __m256 u_vals = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(r_f, _mm256_set1_ps(-0.14713f)),
                          _mm256_mul_ps(g_f, _mm256_set1_ps(-0.28886f))),
            _mm256_add_ps(_mm256_mul_ps(b_f, _mm256_set1_ps(0.436f)), _mm256_set1_ps(128.0f)));
        __m256 v_vals = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(r_f, _mm256_set1_ps(0.615f)),
                          _mm256_mul_ps(g_f, _mm256_set1_ps(-0.51499f))),
            _mm256_add_ps(_mm256_mul_ps(b_f, _mm256_set1_ps(-0.10001f)), _mm256_set1_ps(128.0f)));

        // Convert back to integer and store results
        __m256i y_int = _mm256_cvtps_epi32(y_vals);
        __m256i u_int = _mm256_cvtps_epi32(u_vals);
        __m256i v_int = _mm256_cvtps_epi32(v_vals);

        // Store the results back into the buffer
        _mm256_store_si256((__m256i *)&yuvFrame->buf[pix_loc], y_int);
        _mm256_store_si256((__m256i *)&yuvFrame->buf[pix_loc + 1], u_int);
        _mm256_store_si256((__m256i *)&yuvFrame->buf[pix_loc + 2], v_int);
    }

    for (; i < frame_res; i++)
    {
        size_t pix_loc = i * rgbFrame->channels;
        unsigned char r = rgbFrame->buf[pix_loc];
        unsigned char g = rgbFrame->buf[pix_loc + 1];
        unsigned char b = rgbFrame->buf[pix_loc + 2];

        yuvFrame->buf[pix_loc] = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
        yuvFrame->buf[pix_loc + 1] = (unsigned char)(-0.14713 * r - 0.28886 * g + 0.436 * b + 128);
        yuvFrame->buf[pix_loc + 2] = (unsigned char)(0.615 * r - 0.51499 * g - 0.10001 * b + 128);
    }
}

void ldframe_rgb_to_yuv_sse(LDframe *rgbFrame, LDframe *yuvFrame)
{
    assert(SIMD_BATCH_SIZE == 4);
    size_t frame_res = rgbFrame->height * rgbFrame->width;

    size_t i = 0;
    for (size_t i = 0; i + SIMD_BATCH_SIZE <= frame_res; i += SIMD_BATCH_SIZE)
    {
        size_t pix_loc = i * rgbFrame->channels;

        // Load SIMD registers
        __m128i r = _mm_loadu_si128((__m128i *)&rgbFrame->buf[pix_loc]);
        __m128i g = _mm_loadu_si128((__m128i *)&rgbFrame->buf[pix_loc + 1]);
        __m128i b = _mm_loadu_si128((__m128i *)&rgbFrame->buf[pix_loc + 2]);

        // Convert to floats
        __m128 r_f = _mm_cvtepi32_ps(r);
        __m128 g_f = _mm_cvtepi32_ps(g);
        __m128 b_f = _mm_cvtepi32_ps(b);

        __m128 y_vals = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(r_f, _mm_set1_ps(0.299f)),
                       _mm_mul_ps(g_f, _mm_set1_ps(0.587f))),
            _mm_mul_ps(b_f, _mm_set1_ps(0.114f)));

        __m128 u_vals = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(r_f, _mm_set1_ps(-0.14713f)),
                       _mm_mul_ps(g_f, _mm_set1_ps(-0.28886f))),
            _mm_add_ps(_mm_mul_ps(b_f, _mm_set1_ps(0.436f)), _mm_set1_ps(128.0f)));

        __m128 v_vals = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(r_f, _mm_set1_ps(0.615f)),
                       _mm_mul_ps(g_f, _mm_set1_ps(-0.51499f))),
            _mm_add_ps(_mm_mul_ps(b_f, _mm_set1_ps(-0.10001f)), _mm_set1_ps(128.0f)));

        // Convert the float results back to integer
        __m128i y_int = _mm_cvtps_epi32(y_vals);
        __m128i u_int = _mm_cvtps_epi32(u_vals);
        __m128i v_int = _mm_cvtps_epi32(v_vals);

        // Store the YUV values back into the buffer (store in 3 separate channels)
        _mm_storeu_si128((__m128i *)&yuvFrame->buf[pix_loc], y_int);
        _mm_storeu_si128((__m128i *)&yuvFrame->buf[pix_loc + 1], u_int);
        _mm_storeu_si128((__m128i *)&yuvFrame->buf[pix_loc + 2], v_int);
    }

    // Handle the remaining pixels if the number of pixels is not divisible by SIMD_BATCH_SIZE (4)
    for (; i < frame_res; i++)
    {
        size_t pix_loc = i * rgbFrame->channels;
        unsigned char r = rgbFrame->buf[pix_loc];
        unsigned char g = rgbFrame->buf[pix_loc + 1];
        unsigned char b = rgbFrame->buf[pix_loc + 2];

        yuvFrame->buf[pix_loc] = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
        yuvFrame->buf[pix_loc + 1] = (unsigned char)(-0.14713 * r - 0.28886 * g + 0.436 * b + 128);
        yuvFrame->buf[pix_loc + 2] = (unsigned char)(0.615 * r - 0.51499 * g - 0.10001 * b + 128);
    }
}

LDframe *ldframe_rgb_to_gray(LDframe *rgbFrame)
{
    assert(rgbFrame->channels == 3);
    assert(rgbFrame->pix_fmt == PIXEL_FMT_RGB);
    LDframe *grayFrame = allocate_ldframe(rgbFrame->height, rgbFrame->width, 1, PIXEL_FMT_GRAY);
    size_t tot_pixels = rgbFrame->height * rgbFrame->width;
    convert_rgb_to_gray_cpu(rgbFrame->buf, grayFrame->buf, tot_pixels);
    return grayFrame;
}

void convert_rgb_to_gray_cpu(unsigned char *in_buf, unsigned char *out_buf, size_t tot_pixels)
{
    for (int i = 0; i < tot_pixels; i++)
    {
        int rgb_idx = i * 3;
        unsigned char r = in_buf[rgb_idx];
        unsigned char g = in_buf[rgb_idx + 1];
        unsigned char b = in_buf[rgb_idx + 2];
        if (i == 8294400 - 1)
        {
            int yyy = 3;
        }
        out_buf[i] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}
