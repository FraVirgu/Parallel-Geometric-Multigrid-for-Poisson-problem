#include "globals.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
void initialize_random_vector(double *x)
{

    for (int i = 0; i < L; i++)
    {
        x[i] = i;
    }
}
void prolungator(double *input, double *output, int input_H, int input_W, int output_H, int output_W)
{
    double weight[9] = {0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25};

    int index_output_x;
    int index_output_y;

    for (int i = 0; i < input_H; i++)
    {
        for (int j = 0; j < input_W; j++)
        {
            index_output_x = (2 * j) - 1;
            index_output_y = (2 * i) - 1;

            for (int k = 0; k < 3; k++)
            {
                for (int l = 0; l < 3; l++)
                {
                    int x = index_output_x + l;
                    int y = index_output_y + k;

                    if (x >= 0 && x < output_W && y >= 0 && y < output_H)
                    {
                        if (x == 0 || x == output_W - 1 || y == 0 || y == output_H - 1)
                        {
                            output[y * output_W + x] = 0.0; // Enforce boundary condition
                        }
                        else
                        {
                            output[y * output_W + x] += weight[k * 3 + l] * input[i * input_W + j];
                        }
                    }
                }
            }
        }
    }
}
__global__ void prolungator_kernel(const double *__restrict__ input, double *__restrict__ output,
                                   int input_H, int input_W, int output_H, int output_W)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y >= output_H || x >= output_W)
        return;
    if (y == 0 || y == output_H - 1 || x == 0 || x == output_W - 1)
    {
        output[y * output_W + x] = 0.0;
        return;
    }

    const double weight[3][3] = {
        {0.25, 0.5, 0.25},
        {0.5, 1.0, 0.5},
        {0.25, 0.5, 0.25}};

    double value = 0.0;
    double weight_sum = 0.0;

    // Loop over coarse-grid points that could influence this fine-grid point
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {

            int ci = (y / 2) + dy;
            int cj = (x / 2) + dx;

            if (ci >= 0 && ci < input_H && cj >= 0 && cj < input_W)
            {
                int fy = y - (2 * ci);
                int fx = x - (2 * cj);

                // Check if (fx, fy) is within [-1, 1]
                if (fx >= -1 && fx <= 1 && fy >= -1 && fy <= 1)
                {
                    double w = weight[fy + 1][fx + 1];
                    value += w * input[ci * input_W + cj];
                    weight_sum += w;
                }
            }
        }
    }

    output[y * output_W + x] = (weight_sum > 0.0) ? (value / weight_sum) : 0.0;
}

int main()
{
    double *matrix_coarse, *matrix_fine_seq, *matrix_fine_cuda;
    double *d_matrix_coarse, *d_matrix_fine_cuda;

    int n_coarse = N / 2;
    int l_coarse = n_coarse * n_coarse;
    int l_fine = N * N;

    size_t bytes_coarse = l_coarse * sizeof(double);
    size_t bytes_fine = l_fine * sizeof(double);

    // Allocate host memory
    matrix_coarse = new double[l_coarse];
    matrix_fine_seq = new double[l_fine];
    matrix_fine_cuda = new double[l_fine];

    // Allocate unified memory (accessible by both CPU and GPU)
    cudaMallocManaged(&d_matrix_coarse, bytes_coarse);
    cudaMallocManaged(&d_matrix_fine_cuda, bytes_fine);

    // Initialize input coarse matrix
    initialize_random_vector(matrix_coarse);
    initialize_random_vector(d_matrix_coarse);

    // Print coarse matrix (n_coarse x n_coarse)
    std::cout << "Coarse Matrix input:" << std::endl;
    for (int y = 0; y < n_coarse; y++)
    {
        for (int x = 0; x < n_coarse; x++)
        {
            std::cout << matrix_coarse[y * n_coarse + x] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Sequential prolungator
    prolungator(matrix_coarse, matrix_fine_seq, n_coarse, n_coarse, N, N);

    // Print sequential fine matrix (N x N)
    std::cout << "Fine Matrix output SEQUENTIAL:" << std::endl;
    for (int y = 0; y < N; y++)
    {
        for (int x = 0; x < N; x++)
        {
            std::cout << matrix_fine_seq[y * N + x] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // GPU prolungator
    cudaMemset(d_matrix_fine_cuda, 0, bytes_fine); // Clear before accumulation

    int block_size = 16;
    dim3 threadsPerBlock(block_size, block_size);
    dim3 numBlocks((n_coarse + block_size - 1) / block_size,
                   (n_coarse + block_size - 1) / block_size);

    prolungator_kernel<<<numBlocks, threadsPerBlock>>>(
        d_matrix_coarse, d_matrix_fine_cuda,
        n_coarse, n_coarse,
        N, N);

    cudaDeviceSynchronize();

    // Print GPU fine matrix output (N x N)
    std::cout << "Fine Matrix output DEVICE (BASIC CUDA):" << std::endl;
    for (int y = 0; y < N; y++)
    {
        for (int x = 0; x < N; x++)
        {
            std::cout << d_matrix_fine_cuda[y * N + x] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
