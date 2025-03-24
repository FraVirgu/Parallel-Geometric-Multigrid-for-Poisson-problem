#include "globals.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
void initialize_random_vector(double *x, int l)
{

    for (int i = 0; i < l; i++)
    {
        x[i] = i;
    }
}
void dynamic_initialize_zeros_vector(double *x, int l)
{
    for (int i = 0; i < l; i++)
    {
        x[i] = 0.0;
    }
}

bool arrays_are_equal(const double *a1, const double *a2, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (a1[i] != a2[i])
        {
            return false;
        }
    }
    return true;
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
    int index_x_coarse, index_y_coarse;
    int index_x_coarse_tmp, index_y_coarse_tmp;
    double weight_value;
    if (x % 2 == 0 && y % 2 == 0)
    {
        index_x_coarse = x / 2;
        index_y_coarse = y / 2;
        value = 1.0 * input[index_y_coarse * input_W + index_x_coarse];
        output[y * output_W + x] = value;
        return;
    }

    if (x % 2 != 0 && y % 2 != 0)
    {
        // the comment refers to the element of the coarse grid that influences the fine grid
        weight_value = 0.25;

        // top left corner
        index_x_coarse = (x - 1) / 2;
        index_y_coarse = (y - 1) / 2;
        index_x_coarse_tmp = index_x_coarse;
        index_y_coarse_tmp = index_y_coarse;
        value += weight_value * input[index_y_coarse * input_W + index_x_coarse];

        // top right corner
        index_y_coarse = index_y_coarse;
        index_x_coarse = index_x_coarse + 1;
        value += weight_value * input[index_y_coarse * input_W + index_x_coarse];

        // bottom left corner
        index_y_coarse = index_y_coarse_tmp + 1;
        index_x_coarse = index_x_coarse_tmp;
        value += weight_value * input[index_y_coarse * input_W + index_x_coarse];

        // bottom right corner
        index_y_coarse = index_y_coarse;
        index_x_coarse = index_x_coarse + 1;
        value += weight_value * input[index_y_coarse * input_W + index_x_coarse];

        output[y * output_W + x] = value;
        return;
    }

    if (x % 2 == 0 && y % 2 != 0)
    {
        weight_value = 0.5;
        index_x_coarse = x / 2;
        index_y_coarse = (y - 1) / 2;
        value += weight_value * input[index_y_coarse * input_W + index_x_coarse];
        index_y_coarse = (y + 1) / 2;
        value += weight_value * input[index_y_coarse * input_W + index_x_coarse];
        output[y * output_W + x] = value;
        return;
    }

    if (x % 2 != 0 && y % 2 == 0)
    {
        weight_value = 0.5;
        index_x_coarse = (x - 1) / 2;
        index_y_coarse = y / 2;
        value += weight_value * input[index_y_coarse * input_W + index_x_coarse];
        index_x_coarse = (x + 1) / 2;
        value += weight_value * input[index_y_coarse * input_W + index_x_coarse];
        output[y * output_W + x] = value;
        return;
    }
}

int main()
{
    double *matrix_coarse, *matrix_fine_seq;
    double *d_matrix_coarse, *d_matrix_fine_cuda;

    int n_coarse = N / 2;
    int l_coarse = n_coarse * n_coarse;
    int l_fine = N * N;

    size_t bytes_coarse = l_coarse * sizeof(double);
    size_t bytes_fine = l_fine * sizeof(double);

    // Allocate host memory
    matrix_coarse = new double[l_coarse];
    matrix_fine_seq = new double[l_fine];

    // Allocate unified memory (accessible by both CPU and GPU)
    cudaMallocManaged(&d_matrix_coarse, bytes_coarse);
    cudaMallocManaged(&d_matrix_fine_cuda, bytes_fine);

    // Initialize input coarse matrix
    initialize_random_vector(matrix_coarse, l_coarse);
    initialize_random_vector(d_matrix_coarse, l_coarse);

    // Initialize output fine matrix
    dynamic_initialize_zeros_vector(matrix_fine_seq, l_fine);
    dynamic_initialize_zeros_vector(d_matrix_fine_cuda, l_fine);

    auto start = std::chrono::high_resolution_clock::now();
    // Sequential prolungator
    prolungator(matrix_coarse, matrix_fine_seq, n_coarse, n_coarse, N, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Sequential prolungator elapsed time: " << elapsed.count() << " s\n";

    int TILE_SIZE = 64;
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    auto start_cuda = std::chrono::high_resolution_clock::now();
    prolungator_kernel<<<numBlocks, threadsPerBlock>>>(
        d_matrix_coarse, d_matrix_fine_cuda,
        n_coarse, n_coarse,
        N, N);

    cudaDeviceSynchronize();
    auto end_cuda = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cuda = end_cuda - start_cuda;
    std::cout << "CUDA prolungator elapsed time: " << elapsed_cuda.count() << " s\n";
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    // Compare results
    if (arrays_are_equal(matrix_fine_seq, d_matrix_fine_cuda, l_fine))
    {
        std::cout << "Results are equal!" << std::endl;
    }
    else
    {
        std::cout << "Results are NOT equal!" << std::endl;
    }

    return 0;
}
