#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <cmath>
#include <iomanip>
#define BLOCK_SIZE 16
#define TILE_SIZE 8
#define dim_kernel 5
#define width 10000
#define height 10000

__global__ void
tiled_convolution(const unsigned int *__restrict__ input, const unsigned int *__restrict__ kernel_in, unsigned int *__restrict__ output)
{

    int TILE_WIDTH = blockDim.x;
    int halfkernel = dim_kernel / 2;
    int totalSize = TILE_WIDTH + dim_kernel - 1;
    extern __shared__ unsigned int shared_input[TILE_SIZE * TILE_SIZE + (dim_kernel - 1) * (dim_kernel - 1) - TILE_SIZE * TILE_SIZE];
    extern __shared__ unsigned int kernel[dim_kernel * dim_kernel];
    int th_idx_x = threadIdx.x;
    int th_idx_y = threadIdx.y;
    int block_idx_x = blockIdx.x;
    int block_idx_y = blockIdx.y;

    int threadId = th_idx_y * blockDim.x + th_idx_x;
    int totalThreads = blockDim.x * blockDim.y;

    // Total number of elements to load
    int totalElements = totalSize * totalSize;
    int elementsPerThread = (totalElements + totalThreads - 1) / totalThreads; // rounding up

    // shared memory load
    for (int i = 0; i < elementsPerThread; i++)
    {
        int idx = i * totalThreads + threadId;
        if (idx < totalElements)
        {
            int row = idx / totalSize;
            int col = idx % totalSize;

            int global_matrix_col = block_idx_x * TILE_WIDTH + col - halfkernel;
            int global_matrix_row = block_idx_y * TILE_WIDTH + row - halfkernel;
            if (global_matrix_col < 0)
                global_matrix_col = 0;
            if (global_matrix_col >= width)
                global_matrix_col = width - 1;
            if (global_matrix_row < 0)
                global_matrix_row = 0;
            if (global_matrix_row >= height)
                global_matrix_row = height - 1;

            shared_input[row * totalSize + col] = input[global_matrix_col * width + global_matrix_row];
        }
    }

    if (th_idx_x < dim_kernel && th_idx_y < dim_kernel)
    {
        kernel[th_idx_x + th_idx_y * dim_kernel] = kernel_in[th_idx_x + th_idx_y * dim_kernel];
    }

    __syncthreads();

    int out_x = block_idx_x * TILE_WIDTH + th_idx_x;
    int out_y = block_idx_y * TILE_WIDTH + th_idx_y;
    if (out_x < width && out_y < height)
    {
        unsigned int sum = 0;
        int base_x = th_idx_x + halfkernel;
        int base_y = th_idx_y + halfkernel;

        // Convolution
        for (int i = -halfkernel; i <= halfkernel; i++)
        {
            for (int j = -halfkernel; j <= halfkernel; j++)
            {
                unsigned int val = shared_input[(base_y + i) * totalSize + (base_x + j)];
                unsigned int coeff = kernel[(i + halfkernel) * dim_kernel + (j + halfkernel)];
                sum += val * coeff;
            }
        }

        output[out_y * width + out_x] = sum;
    }
}

// No tiling
__global__ void convolution(const unsigned int *__restrict__ input,
                            const unsigned int *__restrict__ kernel,
                            unsigned int *__restrict__ output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        unsigned int sum = 0;
        int halfkernel = dim_kernel / 2;

        for (int m = -halfkernel; m <= halfkernel; ++m)
        {
            for (int n = -halfkernel; n <= halfkernel; ++n)
            {
                int ix = min(max(x + n, 0), width - 1);
                int iy = min(max(y + m, 0), height - 1);
                unsigned int val = input[iy * width + ix];
                unsigned int coeff = kernel[(m + halfkernel) * dim_kernel + (n + halfkernel)];
                sum += val * coeff;
            }
        }
        output[y * width + x] = sum;
    }
}

int main(int argc, char *argv[])
{

    size_t matrix_size = width * height * sizeof(unsigned int);
    size_t kernel_size = dim_kernel * dim_kernel * sizeof(unsigned int);

    unsigned int *matrix, *kernel_matrix, *output_matrix, *output_tiled;
    (cudaMallocManaged(&matrix, matrix_size));
    (cudaMallocManaged(&kernel_matrix, kernel_size));
    (cudaMallocManaged(&output_matrix, matrix_size));
    (cudaMallocManaged(&output_tiled, matrix_size));

    // Initialize matrix
    for (int i = 0; i < width; i++)
        for (int j = 0; j < height; j++)
            matrix[i * width + j] = 1;
    // Initialize kernel
    for (int i = 0; i < dim_kernel; i++)
        for (int j = 0; j < dim_kernel; j++)
            kernel_matrix[i * dim_kernel + j] = 1;

    // Timing Events
    cudaEvent_t start, stop;
    (cudaEventCreate(&start));
    (cudaEventCreate(&stop));
    //-------------------------------------------------------------
    // Basic Convolution
    (cudaEventRecord(start, 0));
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    convolution<<<gridSize, blockSize>>>(matrix, kernel_matrix, output_matrix);
    (cudaGetLastError());
    (cudaEventRecord(stop, 0));
    (cudaEventSynchronize(stop));

    float time_basic;
    (cudaEventElapsedTime(&time_basic, start, stop));
    printf("Basic time BLOCK_SIZE=%d: %.4f ms\n", BLOCK_SIZE, time_basic);

    //-------------------------------------------------------------
    // Tiled version
    for (unsigned int i = 0; i < 2; i++)
    {
        (cudaEventRecord(start, 0));
        dim3 dimBlock(TILE_SIZE, TILE_SIZE);
        dim3 dimGrid((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);
        size_t sharedMemSize = (TILE_SIZE + dim_kernel - 1) * (TILE_SIZE + dim_kernel - 1) * sizeof(unsigned int) + dim_kernel * dim_kernel;
        tiled_convolution<<<dimGrid, dimBlock, sharedMemSize>>>(matrix, kernel_matrix, output_tiled);
        (cudaGetLastError());
        (cudaEventRecord(stop, 0));
        (cudaEventSynchronize(stop));
        float time_tiled;
        (cudaEventElapsedTime(&time_tiled, start, stop));
        if (i > 0)
            printf("Tiled TILE_SIZE=%d: %.4f ms\n", TILE_SIZE, time_tiled);
    }

    (cudaFree(matrix));
    (cudaFree(kernel_matrix));
    (cudaFree(output_matrix));
    (cudaFree(output_tiled));
    (cudaEventDestroy(start));
    (cudaEventDestroy(stop));

    return 0;
}