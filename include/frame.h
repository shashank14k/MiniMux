#ifndef FRAME_H
#define FRAME_H
#include "pixel.h"
typedef struct
{
    unsigned char *buf;
    int height;
    int width;
    int channels;
    size_t bufsize;
    PIXEL_FMT pix_fmt;
} LDframe;

LDframe *allocate_ldframe(int height, int width, int channels, PIXEL_FMT pix_fmt);

// Convert rgb to yuv-> assert rgb->frame->pix_fmt == rgb
void ldframe_rgb_to_yuv(LDframe *rgbFrame, LDframe *yuvFrame);
void ldframe_rgb_to_yuv_cpu(LDframe *rgbFrame, LDframe *yuvFrame);
void ldframe_rgb_to_yuv_sse(LDframe *rgbFrame, LDframe *yuvFrame);
void ldframe_rgb_to_yuv_avx(LDframe *rgbFrame, LDframe *yuvFrame);
LDframe *ldframe_rgb_to_gray(LDframe *rgbFrame);
#endif // FRAME_H