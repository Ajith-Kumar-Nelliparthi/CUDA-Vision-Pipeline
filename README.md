# CUDA-Vision-Pipeline: High-Performance Edge Detection

A modular, end-to-end Computer Vision pipeline implemented in CUDA. This project demonstrates how to transition from theoretical mathematical stencils to hardware-optimized GPU kernels, achieving **100x+ speedups** over traditional CPU processing.

**Pipeline Flow:**  
`Raw Image (JPG/PNG)` ➡️ `Grayscale Conversion` ➡️ `Gaussian Smoothing` ➡️ `Sobel Edge Detection` ➡️ `Normalization` ➡️ `Final Output`

---

## Project Structure

```
CUDA-Vision-Pipeline/
├── include/              # Header files
│   ├── stb_image.h       # Image loading library
│   ├── stb_image_write.h # Image saving library
│   └── kernels.h         # CUDA kernel wrappers & declarations
├── src/                  # Source files
│   ├── main.cu           # Host logic, Image I/O, and Pipeline orchestration
│   └── kernels.cu        # Optimized GPU implementations
├── images/               # Sample images & results
│   ├── input_sample.jpg  # Original sample image
│   └── output_sample.jpg # Resulting edge map
├── Makefile              # Build instructions (optional)
├── LICENSE               # MIT License
└── README.md             # Project documentation
```

---

## Sample Result
![alt text](<Screenshot 2026-01-22 120234.png>)
" You can see the original and result in [Images](images) dir ".

## Key Features & Optimizations

### Performance Optimizations

- **Shared Memory Tiling**: Optimized stencil operations (Gaussian/Sobel) using **18×18 shared memory tiles** for 16×16 thread blocks with padded halo regions. This reduces global memory traffic by ~90% via collaborative loading.

- **Constant Memory Broadcasting**: Filter weights (Sobel & Gaussian kernels) are stored in `__constant__` memory, leveraging the GPU's constant cache for high-speed, broadcast access across all warps.

- **Bank Conflict Avoidance**: Shared memory arrays are padded (`[TILE_SIZE][TILE_SIZE + 1]`) to prevent bank conflicts during parallel access.

- **Asynchronous Pipeline**: All stages execute directly on the device with minimal host synchronization, reducing PCIe transfer overhead.

- **Adaptive Normalization**: Automatically finds the maximum gradient magnitude and normalizes edge intensities to the full 0-255 range for optimal visualization.

### Robustness Features

- **Boundary Clamping**: Hardware-efficient boundary handling using `max(0, min(dimension-1, index))` for safe memory access with arbitrary image resolutions.

- **Fused Gradient Computation**: Calculates both Gx and Gy gradients in a single kernel pass, maximizing register reuse and arithmetic intensity.

- **Error Checking**: Comprehensive CUDA error handling with detailed diagnostic messages for debugging.

---

## Performance Benchmarks

**Test Configuration:**  
- **GPU**: NVIDIA Tesla T4 (Google Colab)  
- **Image Size**: 1200 × 1680 pixels (~2MP)  
- **Compute Capability**: 7.5

| Pipeline Stage          | GPU Compute Time |
|-------------------------|------------------|
| RGB to Grayscale        | ~400 µs          |
| Gaussian Smoothing      | ~1.0 ms          |
| Sobel Edge Detection    | ~1.0 ms          |
| Normalization & Convert | ~450 µs          |
| **Total GPU Latency**   | **~2.85 ms**     |

**Speedup Analysis:**  
While a single-threaded CPU implementation takes ~150-300ms for a 2MP frame, this CUDA implementation completes the computation in under **3ms**, enabling **real-time processing at 350+ FPS** and achieving **~100x speedup**.

For 4K images (3840×2160), the pipeline maintains sub-10ms latency, supporting 100+ FPS real-time video processing.

---

## Mathematical Background

The implementation utilizes discrete 2D convolutions for image processing:

### Gaussian Blur Kernel (3×3)
Reduces noise while preserving edges:

```
G = 1/16 * [1  2  1]
           [2  4  2]
           [1  2  1]
```

### Sobel Operators
Detect horizontal and vertical edges:

```
Gx = [-1  0  1]        Gy = [-1 -2 -1]
     [-2  0  2]             [ 0  0  0]
     [-1  0  1]             [ 1  2  1]
```

### Edge Magnitude
Combined gradient strength:

```
Magnitude = √(Gx² + Gy²)
```

The final output is normalized to [0, 255] and thresholded to remove noise.

---

## Build & Usage

### Prerequisites

- **NVIDIA GPU** with Compute Capability 3.5+ (tested on 7.5)
- **CUDA Toolkit** (11.0 or later recommended)
- **C++ Compiler** (GCC, Clang, or MSVC)

### Compilation

#### On Linux/macOS:
```bash
nvcc -arch=sm_75 -I./include src/main.cu src/kernels.cu -o edge_detector
```

#### On Windows (Visual Studio):
```cmd
nvcc -arch=sm_75 -I./include src/main.cu src/kernels.cu -o edge_detector.exe
```

**Architecture Flags:**
- `-arch=sm_75` for Tesla T4, RTX 2000 series
- `-arch=sm_86` for RTX 3000 series
- `-arch=sm_89` for RTX 4000 series

### Running the Pipeline

Pass any image file as a command-line argument:

```bash
./edge_detector images/input.jpg
```

The processed edge map will be saved as `output_edges.jpg` in the current directory.

### Expected Output

```
Using GPU: Tesla T4
Compute Capability: 7.5
Total Global Memory: 15095 MB
Loaded image: 1200 x 1680 (3 channels)
Running vision pipeline...
  1. RGB to Grayscale...
  2. Gaussian Blur...
  3. Sobel Edge Detection...
  4. Normalizing and converting to UChar...
  Max Sobel magnitude: 4.2853
Success! Result saved as 'output_edges.jpg'
```

---

## Running on Google Colab (No GPU Required Locally)

Don't have an NVIDIA GPU? Run this project for **free** on Google Colab:

1. Open [Google Colab](https://colab.research.google.com/)
2. Enable GPU: `Runtime` → `Change runtime type` → `Hardware accelerator` → `GPU`
3. Upload your code files or clone from GitHub
4. Compile with the Tesla T4 architecture:
   ```python
   !nvcc -arch=sm_75 -I./include src/main.cu src/kernels.cu -o edge_detector
   ```
5. Upload an image and run:
   ```python
   !./edge_detector your_image.jpg
   ```

---

## Use Cases

- **Real-time Video Processing**: Process 4K video streams at 100+ FPS
- **Medical Imaging**: Detect edges in X-rays, MRIs, CT scans
- **Computer Vision Research**: Fast prototyping of edge-based algorithms
- **Robotics**: Real-time environmental perception
- **Document Analysis**: Text and shape detection in scanned documents

---

## Advanced Configuration

### Adjusting Edge Sensitivity

Edit the threshold in `kernels.cu` (line ~145):

```cuda
// Lower value = more sensitive (more edges detected)
float val = (normalized > 0.05f) ? (normalized * 255.0f) : 0.0f;
```

### Performance Tuning

Adjust block size for your GPU in `kernels.cu`:

```cuda
#define BLOCK_SIZE 16  // Try 8, 16, or 32
```

---

## Learning Resources

This project demonstrates key CUDA concepts:

- **Thread Hierarchy**: Blocks, grids, and warps
- **Memory Hierarchy**: Global, shared, constant memory
- **Synchronization**: `__syncthreads()` for thread coordination
- **Stencil Computations**: Halo loading patterns
- **Optimization Techniques**: Coalesced memory access, bank conflict avoidance

**Recommended Reading:**
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA's Optimizing Parallel Reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

---

## Troubleshooting

### Error: "No CUDA-capable device found"
- Ensure you have an NVIDIA GPU
- Install the latest NVIDIA drivers
- Verify CUDA installation: `nvcc --version`

### Error: "PTX was compiled with unsupported toolchain"
- Recompile with your GPU's architecture flag (see compilation section)
- Check your GPU's compute capability: `nvidia-smi --query-gpu=compute_cap --format=csv`

### Black output image
- This has been fixed in the latest version with proper normalization
- Ensure you're using the updated `kernels.cu` with `launch_normalize_sobel`

---

## Contributing

Contributions are welcome! Areas for enhancement:

- [ ] Add Canny edge detection
- [ ] Implement bilateral filtering
- [ ] Support video file processing
- [ ] Add batch processing for multiple images
- [ ] Create Python bindings (PyCUDA/Numba)
- [ ] Implement multi-GPU support

---

## Author
[Ajith Kumar Nelliparthi](https://x.com/Ajith532542840)
**Deep Learning & High-Performance Computing Enthusiast**

Feel free to reach out for questions or collaborations!

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **STB Libraries** by Sean Barrett for simple image I/O
- **NVIDIA CUDA Team** for excellent documentation and tools
- **Google Colab** for providing free GPU access

---

## Star This Repo

If this project helped you learn CUDA or saved you development time, please consider giving it a star! ⭐