#include <stdio.h>
#include "util.h"
#include "frame.h"

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
    ldframe_rgb_to_yuv(frame, frame);
    printf("Converted rgb image to yuv\n");
    write_jpepg_img(argv[2], frame->height, frame->width, frame->channels, frame->buf);
}