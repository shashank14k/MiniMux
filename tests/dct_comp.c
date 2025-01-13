#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include "frame.h"
#include "dct.h"
#include "transforms.cuh"

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        fprintf(stderr, "Required input and output file paths.\n");
    }
    LDframe *frame = read_jpeg_img(argv[1]);
    if (frame == NULL)
    {
        fprintf(stderr, "Unable to read jpeg image.\n");
    }
    printf("Read JPEG image\n");
    size_t tot_pixels = frame->height * frame->width;
    unsigned char *dft_res = (unsigned char *)malloc(sizeof(unsigned char) * tot_pixels);
    unsigned char *dft_cnv = (unsigned char *)malloc(sizeof(unsigned char) * tot_pixels);
    LDframe *gray = ldframe_rgb_to_gray(frame);
    dct_ld_threaded(gray->buf, dft_res, frame->height, frame->width, 1, 16);
    idct_ld_threaded(dft_res, dft_cnv, frame->height, frame->width, 1, 16);
    printf("Done inverse\n");
    write_jpepg_img(argv[2], frame->height, frame->width, 1, dft_cnv);
}
