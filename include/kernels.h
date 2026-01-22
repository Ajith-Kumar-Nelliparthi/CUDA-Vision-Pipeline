#ifndef KERNELS_H
#define KERNELS_H

// #include <cuda_runtime.h>

// Instead of declaring kernels, we declare "Launchers"
void launch_rgb2gray(const unsigned char* d_rgb, float* d_gray, int w, int h);
void launch_gaussian_blur(const float* d_in, float* d_out, int w, int h);
void launch_sobel(const float* d_in, float* d_out, int w, int h);
void launch_float2uchar(const float* d_in, unsigned char* d_out, int w, int h);

#endif