#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#define K_SIZE 3
#define R (K_SIZE / 2)
#define BLOCK_SIZE 16
#define TILE_SIZE (BLOCK_SIZE + 2 * R)

// constant memory for sobel kernels
__constant__ float Gx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ float Gy[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

// constant memory for Gaussian Blur
__constant__ float G_Blur[9] = {
    1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
    2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
    1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f
};

// kernel-1: RGB to Grayscale conversion
__global__ void rgb2gray(const unsigned char* rgb, float *gray, int width, int height){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height){
        int idx = (row * width + col) * 3; // RGB has 3 channels
        unsigned char r = rgb[idx];
        unsigned char g = rgb[idx + 1];
        unsigned char b = rgb[idx + 2];

        gray[row * width + col] = (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f;
    }
}

// kernel-2: Gaussian Blur (FIXED)
__global__ void gaussian_blur(const float* in, float *out, int width, int height){
    __shared__ float sdata[TILE_SIZE][TILE_SIZE + 1]; // Added padding to avoid bank conflicts

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;

    // Load data into shared memory with halo
    int tile_start_row = blockIdx.y * BLOCK_SIZE - R;
    int tile_start_col = blockIdx.x * BLOCK_SIZE - R;

    // Each thread loads one or more elements
    for (int i = ty; i < TILE_SIZE; i += BLOCK_SIZE){
        for (int j = tx; j < TILE_SIZE; j += BLOCK_SIZE){
            int global_row = tile_start_row + i;
            int global_col = tile_start_col + j;

            // Clamp to image boundaries
            global_row = max(0, min(height - 1, global_row));
            global_col = max(0, min(width - 1, global_col));

            sdata[i][j] = in[global_row * width + global_col];
        }
    }
    __syncthreads();

    // Apply convolution only for valid output pixels
    if (col < width && row < height){
        float sum = 0.0f;
        for (int i = -R; i <= R; i++){
            for (int j = -R; j <= R; j++){
                sum += sdata[ty + R + i][tx + R + j] * G_Blur[(i + R) * K_SIZE + (j + R)];
            }
        }
        out[row * width + col] = sum;
    }
}

// kernel-3: Sobel Edge Detection (FIXED)
__global__ void sobel_edge(const float* in, float *out, int width, int height){
    __shared__ float sdata[TILE_SIZE][TILE_SIZE + 1]; // Added padding

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;

    // Load data into shared memory with halo
    int tile_start_row = blockIdx.y * BLOCK_SIZE - R;
    int tile_start_col = blockIdx.x * BLOCK_SIZE - R;

    for (int i = ty; i < TILE_SIZE; i += BLOCK_SIZE){
        for (int j = tx; j < TILE_SIZE; j += BLOCK_SIZE){
            int global_row = tile_start_row + i;
            int global_col = tile_start_col + j;

            // Clamp to image boundaries
            global_row = max(0, min(height - 1, global_row));
            global_col = max(0, min(width - 1, global_col));

            sdata[i][j] = in[global_row * width + global_col];
        }
    }
    __syncthreads();

    if (col < width && row < height){
        float sumX = 0.0f, sumY = 0.0f;
        for (int i = -R; i <= R; i++){
            for (int j = -R; j <= R; j++){
                float pixel = sdata[ty + R + i][tx + R + j];
                sumX += pixel * Gx[(i + R) * K_SIZE + (j + R)];
                sumY += pixel * Gy[(i + R) * K_SIZE + (j + R)];
            }
        }
        float mag = sqrtf(sumX * sumX + sumY * sumY);
        out[row * width + col] = (mag > 0.1f) ? mag : 0.0f;
    }
}

// kernel-4: float to unsigned char conversion
__global__ void float2uchar(const float* in, unsigned char *out, int width, int height){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height){
        float val = in[row * width + col] * 255.0f;
        out[row * width + col] = (unsigned char)max(0.0f, min(255.0f, val));
    }
}

// Error checking helper
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err)); \
            return; \
        } \
    } while(0)

// wrapper for RGB to Grayscale conversion
void launch_rgb2gray(const unsigned char* d_rgb, float* d_gray, int w, int h) {
    dim3 threads(16, 16);
    dim3 blocks((w + 15) / 16, (h + 15) / 16);
    rgb2gray<<<blocks, threads>>>(d_rgb, d_gray, w, h);
    CUDA_CHECK_KERNEL();
}

// Wrapper for Gaussian Blur
void launch_gaussian_blur(const float* d_in, float* d_out, int w, int h) {
    dim3 threads(16, 16);
    dim3 blocks((w + 15) / 16, (h + 15) / 16);
    gaussian_blur<<<blocks, threads>>>(d_in, d_out, w, h);
    CUDA_CHECK_KERNEL();
}

// Wrapper for Sobel
void launch_sobel(const float* d_in, float* d_out, int w, int h) {
    dim3 threads(16, 16);
    dim3 blocks((w + 15) / 16, (h + 15) / 16);
    sobel_edge<<<blocks, threads>>>(d_in, d_out, w, h);
    CUDA_CHECK_KERNEL();
}

// Wrapper for Float to Uchar
void launch_float2uchar(const float* d_in, unsigned char* d_out, int w, int h) {
    dim3 threads(16, 16);
    dim3 blocks((w + 15) / 16, (h + 15) / 16);
    float2uchar<<<blocks, threads>>>(d_in, d_out, w, h);
    CUDA_CHECK_KERNEL();
}