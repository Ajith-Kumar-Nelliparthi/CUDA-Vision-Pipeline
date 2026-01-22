#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include "kernels.h"

// stb image for image loading and saving
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while (0)

int main(int argc, char** argv) {
    if (argc < 2){
        std::cout << "Usage: " << argv[0] << " <input_image>" << std::endl;
        return -1;
    }

    // load image from disk
    int width, height, channels;
    unsigned char* h_img = stbi_load(argv[1], &width, &height, &channels, 3);

    if (!h_img){
        fprintf(stderr, "Error loading image %s\n", argv[1]);
        return -1;
    }
    printf("Loaded image: %d x %d (%d channels)\n", width, height, channels);

    size_t img_size = width * height * 3;
    size_t gray_size = width * height * sizeof(float);
    size_t out_size = width * height;

    // allocate device memory
    unsigned char *d_rgb, *d_final_uchar;
    float *d_gray, *d_blur, *d_sobel;

    CUDA_CHECK(cudaMalloc(&d_rgb, img_size));
    CUDA_CHECK(cudaMalloc(&d_gray, gray_size));
    CUDA_CHECK(cudaMalloc(&d_blur, gray_size));
    CUDA_CHECK(cudaMalloc(&d_sobel, gray_size));
    CUDA_CHECK(cudaMalloc(&d_final_uchar, out_size));

    // copy image to device
    CUDA_CHECK(cudaMemcpy(d_rgb, h_img, img_size, cudaMemcpyHostToDevice));
    
    printf("Running vision pipeline...\n");
    // step-1: rgb -> grayscale
    launch_rgb2gray(d_rgb, d_gray, width, height);
    // step-2: gaussian blur
    launch_gaussian_blur(d_gray, d_blur, width, height);
    // step-3: sobel filter
    launch_sobel(d_blur, d_sobel, width, height);
    // convert float output to unsigned char
    launch_float2uchar(d_sobel, d_final_uchar, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned char* h_final = (unsigned char*)malloc(out_size);
    CUDA_CHECK(cudaMemcpy(h_final, d_final_uchar, out_size, cudaMemcpyDeviceToHost));

    // Save Result to Disk
    stbi_write_jpg("output_edges.jpg", width, height, 1, h_final, 100);
    std::cout << "Success! Result saved as 'output_edges.jpg'" << std::endl;

    // Cleanup
    stbi_image_free(h_img);
    free(h_final);
    cudaFree(d_rgb);
    cudaFree(d_gray);
    cudaFree(d_blur);
    cudaFree(d_sobel);
    cudaFree(d_final_uchar);

    return 0;
}
