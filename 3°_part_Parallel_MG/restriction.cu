#include "globals.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#define BLOCK_SIZE 8
void restriction(double *input, double *output, int input_H, int input_W, int output_H, int output_W)
{
    double weight[9] = {0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25};
    for (int i = 0; i < output_H; i++)
    {
        for (int j = 0; j < output_W; j++)
        {
            if (i == 0 || i == output_H - 1 || j == 0 || j == output_W - 1) // Enforce boundary condition
            {
                output[i * output_W + j] = 0.0;
            }
            else
            {
                double sum = 0.0;
                double weight_sum = 0.0;
                int index_input_x = 2 * j;
                int index_input_y = 2 * i;

                for (int k = -1; k <= 1; k++)
                {
                    for (int l = -1; l <= 1; l++)
                    {
                        int x = index_input_x + l;
                        int y = index_input_y + k;

                        if (x >= 0 && x < input_W && y >= 0 && y < input_H)
                        {
                            sum += weight[(k + 1) * 3 + (l + 1)] * input[y * input_W + x];
                            weight_sum += weight[(k + 1) * 3 + (l + 1)];
                        }
                    }
                }

                output[i * output_W + j] = sum / weight_sum;
            }
        }
    }
}
void initialize_random_vector(double *x)
{

    for (int i = 0; i < L; i++)
    {
        x[i] = i;
    }
}

__global__ void restriction_kernel_tiled(const double *__restrict__ input, double *__restrict__ output,
                                         int input_H, int input_W, int output_H, int output_W)
{
    __shared__ double tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2]; // Padding for 3x3 access

    const double weight[3][3] = {
        {0.25, 0.5, 0.25},
        {0.5, 1.0, 0.5},
        {0.25, 0.5, 0.25}};

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    int in_x = out_x * 2;
    int in_y = out_y * 2;

    int local_x = threadIdx.x + 1;
    int local_y = threadIdx.y + 1;

    // Primary point (center)
    if (in_x < input_W && in_y < input_H)
        tile[local_y][local_x] = input[in_y * input_W + in_x];

    // Halo above
    if (threadIdx.y == 0 && in_y - 1 >= 0)
        tile[local_y - 1][local_x] = input[(in_y - 1) * input_W + in_x];

    // Halo below
    if (threadIdx.y == blockDim.y - 1 && in_y + 1 < input_H)
        tile[local_y + 1][local_x] = input[(in_y + 1) * input_W + in_x];

    // Halo left
    if (threadIdx.x == 0 && in_x - 1 >= 0)
        tile[local_y][local_x - 1] = input[in_y * input_W + in_x - 1];

    // Halo right
    if (threadIdx.x == blockDim.x - 1 && in_x + 1 < input_W)
        tile[local_y][local_x + 1] = input[in_y * input_W + in_x + 1];

    // Diagonals if needed for 3x3 stencil
    if (threadIdx.x == 0 && threadIdx.y == 0 && in_x - 1 >= 0 && in_y - 1 >= 0)
        tile[local_y - 1][local_x - 1] = input[(in_y - 1) * input_W + in_x - 1];

    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0 && in_x + 1 < input_W && in_y - 1 >= 0)
        tile[local_y - 1][local_x + 1] = input[(in_y - 1) * input_W + in_x + 1];

    if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1 && in_x - 1 >= 0 && in_y + 1 < input_H)
        tile[local_y + 1][local_x - 1] = input[(in_y + 1) * input_W + in_x - 1];

    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1 && in_x + 1 < input_W && in_y + 1 < input_H)
        tile[local_y + 1][local_x + 1] = input[(in_y + 1) * input_W + in_x + 1];

    __syncthreads();

    // Only compute valid output points
    if (out_x < output_W && out_y < output_H)
    {
        if (out_x == 0 || out_y == 0 || out_x == output_W - 1 || out_y == output_H - 1)
        {
            output[out_y * output_W + out_x] = 0.0;
        }
        else
        {
            double sum = 0.0, wsum = 0.0;
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    double w = weight[dy + 1][dx + 1];
                    double val = tile[threadIdx.y + dy + 1][threadIdx.x + dx + 1];
                    sum += w * val;
                    wsum += w;
                }
            }
            output[out_y * output_W + out_x] = sum / wsum;
        }
    }
}

__global__ void restriction_kernel_basic(const double *__restrict__ input, double *__restrict__ output,
                                         int input_H, int input_W, int output_H, int output_W)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= output_W || out_y >= output_H)
        return;

    const double weight[3][3] = {
        {0.25, 0.5, 0.25},
        {0.5, 1.0, 0.5},
        {0.25, 0.5, 0.25}};

    if (out_x == 0 || out_y == 0 || out_x == output_W - 1 || out_y == output_H - 1)
    {
        output[out_y * output_W + out_x] = 0.0;
        return;
    }

    int in_x = out_x * 2;
    int in_y = out_y * 2;

    double sum = 0.0;
    double wsum = 0.0;

    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            int src_x = in_x + dx;
            int src_y = in_y + dy;

            if (src_x >= 0 && src_x < input_W && src_y >= 0 && src_y < input_H)
            {
                double val = input[src_y * input_W + src_x];
                double w = weight[dy + 1][dx + 1];
                sum += w * val;
                wsum += w;
            }
        }
    }

    output[out_y * output_W + out_x] = sum / wsum;
}

int main()
{
    double *matrix_one, *matrix_restr, *matrix_restr_basic;
    double *d_matrix_one, *d_matrix_restr, *d_matrix_restr_basic;

    int n_restr = N / 2;
    int l_restr = n_restr * n_restr;
    size_t bytes = L * sizeof(double);
    size_t bytes_restr = l_restr * sizeof(double);

    // Allocate host memory
    matrix_one = new double[L];
    matrix_restr = new double[l_restr];
    matrix_restr_basic = new double[l_restr];

    // Allocate unified memory accessible by both CPU and GPU
    cudaMallocManaged(&d_matrix_one, bytes);
    cudaMallocManaged(&d_matrix_restr, bytes_restr);
    cudaMallocManaged(&d_matrix_restr_basic, bytes_restr);

    // Initialize input matrix
    initialize_random_vector(matrix_one);
    initialize_random_vector(d_matrix_one);

    // Print input matrix (N x N)
    std::cout << "Matrix input (Fine):" << std::endl;
    for (int y = 0; y < N; y++)
    {
        for (int x = 0; x < N; x++)
        {
            std::cout << matrix_one[y * N + x] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Sequential restriction
    restriction(matrix_one, matrix_restr, H, W, n_restr, n_restr);

    // Print sequential output matrix (n_restr x n_restr)
    std::cout << "Matrix output SEQUENTIAL (COARSE):" << std::endl;
    for (int y = 0; y < n_restr; y++)
    {
        for (int x = 0; x < n_restr; x++)
        {
            std::cout << matrix_restr[y * n_restr + x] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // GPU restriction using tiled kernel
    int block_size_restr = N / 2; // Block size for GPU kernel
    dim3 threadsPerBlock_restr(block_size_restr, block_size_restr);
    dim3 numBlocks_restr((n_restr + block_size_restr - 1) / block_size_restr,
                         (n_restr + block_size_restr - 1) / block_size_restr);

    restriction_kernel_tiled<<<numBlocks_restr, threadsPerBlock_restr>>>(
        d_matrix_one, d_matrix_restr, H, W, n_restr, n_restr);

    cudaDeviceSynchronize();

    // Print GPU output matrix (n_restr x n_restr) from tiled kernel
    std::cout << "Matrix output DEVICE (TILED) (COARSE):" << std::endl;
    for (int y = 0; y < n_restr; y++)
    {
        for (int x = 0; x < n_restr; x++)
        {
            std::cout << d_matrix_restr[y * n_restr + x] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // GPU restriction using basic kernel
    restriction_kernel_basic<<<numBlocks_restr, threadsPerBlock_restr>>>(
        d_matrix_one, d_matrix_restr_basic, H, W, n_restr, n_restr);

    cudaDeviceSynchronize();

    // Print GPU output matrix (n_restr x n_restr) from basic kernel
    std::cout << "Matrix output DEVICE (BASIC) (COARSE) :" << std::endl;
    for (int y = 0; y < n_restr; y++)
    {
        for (int x = 0; x < n_restr; x++)
        {
            std::cout << d_matrix_restr_basic[y * n_restr + x] << " ";
        }
        std::cout << std::endl;
    }

    // Free allocated memory
    delete[] matrix_one;
    delete[] matrix_restr;
    delete[] matrix_restr_basic;
    cudaFree(d_matrix_one);
    cudaFree(d_matrix_restr);
    cudaFree(d_matrix_restr_basic);

    return 0;
}
