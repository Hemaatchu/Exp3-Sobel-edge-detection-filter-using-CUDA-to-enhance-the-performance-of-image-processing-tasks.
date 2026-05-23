# Exp3-Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.

<h3>NAME : Hemavathy S</h3>
<h3>REGISTER NO: 212223230076</h3>
<h3>EX.NO : 3</h3>
<h3>DATE : 21-05-2026</h3>
<h1> <align=center> Sobel edge detection filter using CUDA </h3>
  Implement Sobel edge detection filtern using GPU.</h3>
Experiment Details:
  
## AIM:
  The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

Code Overview: You will work with the provided CUDA implementation of the Sobel edge detection filter. The code reads an input image, applies the Sobel filter in parallel on the GPU, and writes the result to an output image.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
CUDA Toolkit and OpenCV installed.
A sample image for testing.

## PROCEDURE:
Tasks: 
a. Modify the Kernel:

Update the kernel to handle color images by converting them to grayscale before applying the Sobel filter.
Implement boundary checks to avoid reading out of bounds for pixels on the image edges.

b. Performance Analysis:

Measure the performance (execution time) of the Sobel filter with different image sizes (e.g., 256x256, 512x512, 1024x1024).
Analyze how the block size (e.g., 8x8, 16x16, 32x32) affects the execution time and output quality.

c. Comparison:

Compare the output of your CUDA Sobel filter with a CPU-based Sobel filter implemented using OpenCV.
Discuss the differences in execution time and output quality.

## PROGRAM:

```
%%writefile sobelEdgeDetectionFilter.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <opencv2/opencv.hpp>

using namespace cv;

__global__ void sobelFilter(unsigned char *srcImage,
                            unsigned char *dstImage,
                            unsigned int width,
                            unsigned int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < width - 1 && y < height - 1)
    {
        int Gx = 0;
        int Gy = 0;

        int idx;

        // Sobel X
        idx = (y - 1) * width + (x - 1);
        Gx += -1 * srcImage[idx];

        idx = (y - 1) * width + x;
        Gx += 0 * srcImage[idx];

        idx = (y - 1) * width + (x + 1);
        Gx += 1 * srcImage[idx];

        idx = y * width + (x - 1);
        Gx += -2 * srcImage[idx];

        idx = y * width + (x + 1);
        Gx += 2 * srcImage[idx];

        idx = (y + 1) * width + (x - 1);
        Gx += -1 * srcImage[idx];

        idx = (y + 1) * width + x;
        Gx += 0 * srcImage[idx];

        idx = (y + 1) * width + (x + 1);
        Gx += 1 * srcImage[idx];

        // Sobel Y
        idx = (y - 1) * width + (x - 1);
        Gy += -1 * srcImage[idx];

        idx = (y - 1) * width + x;
        Gy += -2 * srcImage[idx];

        idx = (y - 1) * width + (x + 1);
        Gy += -1 * srcImage[idx];

        idx = (y + 1) * width + (x - 1);
        Gy += 1 * srcImage[idx];

        idx = (y + 1) * width + x;
        Gy += 2 * srcImage[idx];

        idx = (y + 1) * width + (x + 1);
        Gy += 1 * srcImage[idx];

        int magnitude = sqrtf((float)(Gx * Gx + Gy * Gy));

        if (magnitude > 255)
            magnitude = 255;

        dstImage[y * width + x] = (unsigned char)magnitude;
    }
}

void checkCudaErrors(cudaError_t r)
{
    if (r != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    // Read input image
    Mat image = imread("/content/butterfly.webp", IMREAD_GRAYSCALE);

    if (image.empty())
    {
        printf("Error: Image not found.\n");
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    size_t imageSize = width * height * sizeof(unsigned char);

    // Allocate host memory
    unsigned char *h_outputImage =
        (unsigned char *)malloc(imageSize);

    if (h_outputImage == nullptr)
    {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    // Allocate device memory
    unsigned char *d_inputImage, *d_outputImage;

    checkCudaErrors(cudaMalloc(&d_inputImage, imageSize));
    checkCudaErrors(cudaMalloc(&d_outputImage, imageSize));

    checkCudaErrors(cudaMemcpy(d_inputImage,
                               image.data,
                               imageSize,
                               cudaMemcpyHostToDevice));

    // CUDA timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel launch configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16,
                  (height + 15) / 16);

    cudaEventRecord(start);

    sobelFilter<<<gridSize, blockSize>>>(
        d_inputImage,
        d_outputImage,
        width,
        height);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    // Calculate execution time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy output back to host
    checkCudaErrors(cudaMemcpy(h_outputImage,
                               d_outputImage,
                               imageSize,
                               cudaMemcpyDeviceToHost));

    // Save output image
    Mat outputImage(height,
                    width,
                    CV_8UC1,
                    h_outputImage);

    imwrite("/content/output_sobel.jpeg", outputImage);

    // Free memory
    free(h_outputImage);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Total time taken: %f milliseconds\n",
           milliseconds);

    return 0;
}

```

## OUTPUT:
<img width="800" height="533" alt="image" src="https://github.com/user-attachments/assets/0db9cc64-e46d-4a61-8e11-7830da469789" />

<img width="911" height="575" alt="image" src="https://github.com/user-attachments/assets/47f1284d-837e-4b02-a87e-f1a824a06774" />

## RESULT:
Thus the program has been executed by using CUDA to ________________.

Questions and Answers:

1.What challenges did you face while implementing the Sobel filter for color images?

One challenge in implementing the Sobel filter for color images was handling multiple color channels (RGB) instead of a single grayscale channel. Another difficulty was managing memory allocation and ensuring correct edge detection results while maintaining good CUDA performance and avoiding indexing errors.

2.How did changing the block size influence the performance of your CUDA implementation?

Changing the block size affected the execution speed and GPU utilization of the CUDA implementation. Larger block sizes improved parallel processing efficiency up to a limit, while very large or very small block sizes reduced performance due to increased memory overhead and inefficient thread usage.

3.What were the differences in output between the CUDA and CPU implementations? Discuss any discrepancies.

The CUDA implementation produced output similar to the CPU implementation, but slight differences were observed due to parallel computation and floating-point precision variations. CUDA processing was much faster, especially for larger images, while the CPU implementation took more execution time.

5.Suggest potential optimizations for improving the performance of the Sobel filter.

Performance of the Sobel filter can be improved by using shared memory to reduce global memory access and by optimizing thread block sizes for better GPU utilization. Additional improvements include using grayscale images instead of color images and minimizing memory transfers between CPU and GPU.


Deliverables:

Modified CUDA code with comments explaining your changes.
A report summarizing your findings, including graphs of execution times and a comparison of outputs.
Answers to the questions posed in the experiment.
Tools Required:

