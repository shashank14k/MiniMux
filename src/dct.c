#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include "dct.h"

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
int N_THREADS;

void dct_ld_1d(int vec[8])
{
    int vec_tmp[8];
    // STAGE 1 Transforms
    for (int j = 0; j < 4; j++)
    {
        vec_tmp[j] = vec[j] + vec[7 - j];
        vec_tmp[7 - j] = vec[j] - vec[7 - j];
    }

    // STAGE 2 Transforms
    vec[0] = vec_tmp[0] + vec_tmp[3];
    vec[1] = vec_tmp[1] + vec_tmp[2];
    vec[2] = vec_tmp[1] - vec_tmp[2];
    vec[3] = vec_tmp[0] - vec_tmp[3];
    vec[4] = vec_tmp[4];
    vec[5] = ((vec_tmp[6] - vec_tmp[5] * C0) + DELTA) >> SHIFT;
    vec[6] = ((vec_tmp[6] + vec_tmp[5] * C0) + DELTA) >> SHIFT;
    vec[7] = vec_tmp[7];

    int t0 = vec[0];
    int t4 = vec[4];
    int t6 = vec[6];
    // Final stage of even
    vec[0] = ((t0 + vec[1]) * C4 + DELTA) >> SHIFT;
    vec[4] = ((t0 - vec[1]) * C4 + DELTA) >> SHIFT;
    vec[2] = ((vec[2] * C6 + vec[3] * C2) + DELTA) >> SHIFT;
    vec[6] = ((vec[3] * C6 - vec[2] * C2) + DELTA) >> SHIFT;

    // STAGE 3 Transforms
    vec_tmp[4] = t4 + vec[5];
    vec_tmp[5] = t4 - vec[5];
    vec_tmp[6] = vec[7] - t6;
    vec_tmp[7] = vec[7] + t6;

    // Final stage of Odd
    vec[1] = ((vec_tmp[4] * C7 + vec_tmp[7] * C1) + DELTA) >> SHIFT;
    vec[7] = ((vec_tmp[7] * C7 - vec_tmp[4] * C1) + DELTA) >> SHIFT;
    vec[5] = ((vec_tmp[5] * C3 + vec_tmp[6] * C5) + DELTA) >> SHIFT;
    vec[3] = ((vec_tmp[6] * C4 - vec_tmp[5] * C5) + DELTA) >> SHIFT;
}

void transpose_ld(int data[8][8])
{
    int i, j;
    for (i = 0; i < 8; ++i)
    {
        for (j = i + 1; j < 8; ++j)
        {
            int tmp = data[i][j];
            data[i][j] = data[j][i];
            data[j][i] = tmp;
        }
    }
}

void *dct_thread(void *arg)
{
    DCTThreadData *data = (DCTThreadData *)arg;
    int p, q, k;
    int tmp[8][8];
    for (p = 0; p < 8; p++)
    {
        for (q = 0; q < 8; q++)
        {
            tmp[p][q] = (int)data->x[(data->block_y + p) * data->width * data->channels + (data->block_x + q) * data->channels];
        }
    }
    // 1D DCT for each row
    for (k = 0; k < 8; k++)
    {
        dct_ld_1d(tmp[k]);
    }

    transpose_ld(tmp);

    // 1D DCT for each col
    for (k = 0; k < 8; k++)
    {
        dct_ld_1d(tmp[k]);
    }

    transpose_ld(tmp);

    for (p = 0; p < 8; p++)
    {
        for (q = 0; q < 8; q++)
        {
            int pix_val = (int)(tmp[p][q] / quant_matrix_[p][q]);
            if (pix_val < 0)
                pix_val = 0;
            if (pix_val > 255)
                pix_val = 255;
            data->y[(data->block_y + p) * data->width + (data->block_x + q)] = (unsigned char)pix_val;
        }
    }

    pthread_mutex_lock(&mutex);
    N_THREADS--;
    pthread_cond_signal(&cond); // Signal that a thread has finished
    pthread_mutex_unlock(&mutex);
    return NULL;
}

void dct_ld_threaded(unsigned char *X, unsigned char *Y, int height, int width, int channel, int n_threads)
{
    for (int i = 0; i < height; i += 8)
    {
        for (int j = 0; j < width; j += 8)
        {
            // Allocate and fill thread data
            DCTThreadData *data = malloc(sizeof(DCTThreadData));
            data->x = X;
            data->y = Y;
            data->width = width;
            data->channels = channel;
            data->block_x = j;
            data->block_y = i;

            // Create the thread
            pthread_mutex_lock(&mutex);
            while (N_THREADS >= n_threads)
            {
                pthread_cond_wait(&cond, &mutex); // Wait for thread to finish
            }
            N_THREADS++;
            pthread_mutex_unlock(&mutex);

            pthread_t thread;
            pthread_create(&thread, NULL, dct_thread, (void *)data);
            pthread_detach(thread);
        }
    }
}

void dct_ld(unsigned char *X, unsigned char *Y, int heigth, int width, int channel)
{
    // Create temp matrix
    int tmp[8][8];
    int i, j, p, q, k;
    for (i = 0; i < heigth; i += 8)
    {
        for (j = 0; j < width; j += 8)
        {
            if (i == 1)
            {
                int xx = 2;
            }
            for (p = 0; p < 8; p++)
            {
                for (q = 0; q < 8; q++)
                {
                    tmp[p][q] = (int)X[(i + p) * width * channel + (j + q) * channel];
                }
            }
            // 1D DCT for each row
            for (k = 0; k < 8; k++)
            {
                dct_ld_1d(tmp[k]);
            }

            transpose_ld(tmp);

            // 1D DCT for each col
            for (k = 0; k < 8; k++)
            {
                dct_ld_1d(tmp[k]);
            }

            transpose_ld(tmp);

            for (p = 0; p < 8; p++)
            {
                for (q = 0; q < 8; q++)
                {
                    int pix_val = (int)(tmp[p][q] / quant_matrix_[p][q]);
                    if (pix_val < 0)
                        pix_val = 0;
                    if (pix_val > 255)
                        pix_val = 255;
                    Y[(i + p) * width + (j + q)] = (unsigned char)pix_val;
                }
            }
        }
    }
}

void idct_ld_1d(int vec[8])
{
    int vec_tmp[8];

    // Final stage of even
    int t0 = (vec[0] * C4 + DELTA) >> SHIFT;
    int t1 = (vec[4] * C4 + DELTA) >> SHIFT;
    vec[0] = t0 + t1;
    vec[4] = t0 - t1;

    // Final stage of odd
    vec_tmp[4] = (vec[1] * C7 + vec[7] * C1 + DELTA) >> SHIFT;
    vec_tmp[5] = (vec[5] * C3 + vec[6] * C5 + DELTA) >> SHIFT;
    vec_tmp[6] = (vec[7] - vec[6] * C6 + DELTA) >> SHIFT;
    vec_tmp[7] = (vec[7] + vec[6] * C6 + DELTA) >> SHIFT;

    // STAGE 3 Transforms
    vec[1] = vec_tmp[4] + vec_tmp[5];
    vec[3] = vec_tmp[5] - vec_tmp[4];
    vec[5] = vec_tmp[5];
    vec[7] = vec_tmp[7];

    // STAGE 2 Transforms
    vec_tmp[0] = vec[0] + vec[1];
    vec_tmp[1] = vec[2] + vec[3];
    vec_tmp[2] = vec[1] - vec[2];
    vec_tmp[3] = vec[0] - vec[3];
    vec[0] = vec_tmp[0] + vec_tmp[1];
    vec[1] = vec_tmp[0] - vec_tmp[1];
    vec[2] = vec_tmp[2];
    vec[3] = vec_tmp[3];

    // STAGE 1 Transforms
    for (int j = 0; j < 4; j++)
    {
        vec[j] += vec[7 - j];
        vec[7 - j] = vec[j] - vec[7 - j];
    }
}

void idct_ld(unsigned char *Y, unsigned char *X, int height, int width, int channel)
{
    int tmp[8][8];
    int i, j, p, q, k;

    for (i = 0; i < height; i += 8)
    {
        for (j = 0; j < width; j += 8)
        {
            // Load the block from Y
            for (p = 0; p < 8; p++)
            {
                for (q = 0; q < 8; q++)
                {
                    tmp[p][q] = (int)Y[(i + p) * width + (j + q)] * quant_matrix_[p][q];
                }
            }

            // 1D IDCT for each column
            transpose_ld(tmp);
            for (k = 0; k < 8; k++)
            {
                idct_ld_1d(tmp[k]);
            }

            // 1D IDCT for each row
            transpose_ld(tmp);
            for (k = 0; k < 8; k++)
            {
                idct_ld_1d(tmp[k]);
            }

            for (p = 0; p < 8; p++)
            {
                for (q = 0; q < 8; q++)
                {
                    int pix_val = tmp[p][q];
                    if (pix_val < 0)
                        pix_val = 0;
                    if (pix_val > 255)
                        pix_val = 255;
                    X[(i + p) * width + (j + q)] = (unsigned char)pix_val;
                }
            }
        }
    }
}

void *idct_thread(void *arg)
{
    DCTThreadData *data = (DCTThreadData *)arg;
    int p, q, k;
    int tmp[8][8];
    for (p = 0; p < 8; p++)
    {
        for (q = 0; q < 8; q++)
        {
            tmp[p][q] = (int)data->y[(data->block_y + p) * data->width + (data->block_x + q)] * quant_matrix_[p][q];
        }
    }

    // 1D IDCT for each column
    transpose_ld(tmp);
    for (k = 0; k < 8; k++)
    {
        idct_ld_1d(tmp[k]);
    }

    // 1D IDCT for each row
    transpose_ld(tmp);
    for (k = 0; k < 8; k++)
    {
        idct_ld_1d(tmp[k]);
    }

    for (p = 0; p < 8; p++)
    {
        for (q = 0; q < 8; q++)
        {
            int pix_val = tmp[p][q];
            if (pix_val < 0)
                pix_val = 0;
            if (pix_val > 255)
                pix_val = 255;
            data->x[(data->block_y + p) * data->width + (data->block_x + q)] = (unsigned char)pix_val;
        }
    }
    pthread_mutex_lock(&mutex);
    N_THREADS--;
    pthread_cond_signal(&cond); // Signal that a thread has finished
    pthread_mutex_unlock(&mutex);
    return NULL;
}

void idct_ld_threaded(unsigned char *Y, unsigned char *X, int height, int width, int channel, int n_threads)
{
    for (int i = 0; i < height; i += 8)
    {
        for (int j = 0; j < width; j += 8)
        {
            // Allocate and fill thread data
            DCTThreadData *data = malloc(sizeof(DCTThreadData));
            data->x = X;
            data->y = Y;
            data->width = width;
            data->channels = channel;
            data->block_x = j;
            data->block_y = i;

            // Create the thread
            pthread_mutex_lock(&mutex);
            while (N_THREADS >= n_threads)
            {
                pthread_cond_wait(&cond, &mutex); // Wait for thread to finish
            }
            N_THREADS++;
            pthread_mutex_unlock(&mutex);

            pthread_t thread;
            pthread_create(&thread, NULL, idct_thread, (void *)data);
            pthread_detach(thread);
        }
    }
}