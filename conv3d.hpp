#pragma once

#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <omp.h>

#ifdef LIB_COMPILATION
#define CONV3DDEF extern "C"
#else
#define CONV3DDEF static inline
#endif

// Declaration
CONV3DDEF float clampf(float val);
CONV3DDEF void conv3d(void* input_ptr, float* kernel, void* output_ptr, int width, int height, int NOT);

#ifdef CONV3D_IMPLEMENTATION
#undef CONV3D_IMPLEMENTATION

// Definitions
CONV3DDEF float clampf(float val)
{
    if (val < 0.0f) return 0.0f;
    if (val > 255.0f) return 255.0f;
    return val;
}

int KERNEL_WIDTH = 3;
int KERNEL_HEIGHT = 3;
int KERNEL_DEPTH = 3;
int CHANNELS = 3;
int padH = KERNEL_HEIGHT / 2;
int padW = KERNEL_WIDTH / 2;

CONV3DDEF void conv3d__(void* input_ptr, float* kernel, void* output_ptr, int width, int height, int NOT)
{
    int starty = padH;
    int endy = height - padH;
    int startx = padW;
    int endx = width - padW;
    float* input = static_cast<float*>(input_ptr);
    float* output = static_cast<float*>(output_ptr);

    int input_frame_stride = height * width * CHANNELS;
    int input_row_stride = width * CHANNELS;

    // #pragma omp parallel for num_threads(NOT)
    for (int y = starty; y < endy; ++y)
    {
        for (int x = startx; x < endx; ++x)
        {
            int out_idx_start = y * input_row_stride + x * CHANNELS;
            float sum[3] = {0.0f, 0.0f, 0.0f};
            for (int dz = 0; dz < KERNEL_DEPTH; ++dz)
            {
                for (int dy = 0; dy < KERNEL_HEIGHT; ++dy)
                {
                    for (int dx = 0; dx < KERNEL_WIDTH; ++dx)
                    {
                        int y_in = y + dy - padH;
                        int x_in = x + dx - padW;
                        
                        int idx_in_start = dz * input_frame_stride + y_in * input_row_stride + x_in * CHANNELS;
                        int kidx = dz * (KERNEL_WIDTH * KERNEL_HEIGHT) + dy * KERNEL_WIDTH + dx;
                        
                        float k = kernel[kidx];
                        for (int c = 0; c < CHANNELS; ++c)
                        {
                            sum[c] += input[idx_in_start + c] * k;
                        }
                    }
                }
            }
            
            for (int c = 0; c < CHANNELS; ++c)
            {
                output[out_idx_start + c] = clampf(sum[c]);
            }
        }
    }
}

CONV3DDEF void conv3d(void* input_ptr, float* kernel, void* output_ptr, int width, int height, int NOT)
{
    conv3d__(input_ptr, kernel, output_ptr, width, height, NOT);
}

#endif // CONV3D_IMPLEMENTATION
#undef CONV3DDEF
