#include "globals.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// #define BLOCK_SIZE (N / 2)
#define BLOCK_SIZE 8

//  Initialize helper functions
void initialize_zeros_vector(double *x)
{
    for (int i = 0; i < L; i++)
    {
        x[i] = 0.0;
    }
}
void initialize_random_vector(double *x)
{

    for (int i = 0; i < L; i++)
    {
        x[i] = i;
    }
}
void compute_rhs(double *f)
{
    double dx = a / W;
    double dy = a / H;
    double factor = (M_PI * M_PI / (a * a)) * (p * p + q * q);

    for (int y = 0; y < H; y++)
    {
        for (int x = 0; x < W; x++)
        {
            double x_val = x * dx;
            double y_val = y * dy;

            // Apply Dirichlet boundary condition: f = 0 at the boundaries
            if (x == 0 || x == W - 1 || y == 0 || y == H - 1)
            {
                f[y * W + x] = 0.0;
            }
            else
            {
                f[y * W + x] = factor * sin(p * M_PI * x_val / a) * sin(q * M_PI * y_val / a);
            }
        }
    }
}
double vector_norm(double *f)
{
    double sum = 0.0;
    sum = 0.0;
    for (int i = 0; i < L; i++)
    {
        sum += f[i] * f[i]; // Sum of squares
    }

    return sqrt(sum); // Square root of sum
}
void compute_residual(double *r, double *x, double *f)
{
    for (int y = 1; y < H - 1; y++)
    {
        for (int x_pos = 1; x_pos < W - 1; x_pos++)
        {
            int index = y * W + x_pos;
            // return the normalized residual
            r[index] = ((h * h) * f[index] - 4 * x[index] + x[index - 1] + x[index + 1] + x[index - W] + x[index + W]);
        }
    }
}
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
bool JacobiSequential(double *x, double *x_new, double *f, int v, int height, int weight, double h_act, int l)
{
    for (int i = 0; i < v; i++)
    {
        // Perform Jacobi iteration
        for (int y = 1; y < height - 1; y++)
        {
            for (int x_pos = 1; x_pos < weight - 1; x_pos++)
            {
                int index = y * weight + x_pos;
                x_new[index] = 0.25 * ((h_act * h_act * f[index]) + x[index - 1] + x[index + 1] + x[index - weight] + x[index + weight]);
            }
        }

        // Copy x_new to x for next iteration
        for (int j = 0; j < l; j++)
        {
            x[j] = x_new[j];
        }
    }
    return false;
}

__global__ void JacobiKernel(double *x, double *x_new, double *f, int height, int width, double h_act)
{
    int x_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int y_pos = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y_pos * width + x_pos;
    double x_out = 0.0;

    if (x_pos != 0 && y_pos != 0 && x_pos < width - 1 && y_pos < height - 1)
    {
        int index = y_pos * width + x_pos;
        x_out = 0.25 * ((h_act * h_act * f[index]) + x[index - 1] + x[index + 1] + x[index - width] + x[index + width]);
    }
    __syncthreads();

    if (x_pos != 0 && y_pos != 0 && x_pos < width - 1 && y_pos < height - 1)
    {
        x_new[index] = x_out;
    }
    __syncthreads();
}
__global__ void device_compute_residual(double *r, double *x, double *f, int height, int width, double h_act)
{
    int x_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int y_pos = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y_pos * width + x_pos;
    double r_out = 0.0;

    if (x_pos != 0 && y_pos != 0 && x_pos < width - 1 && y_pos < height - 1)
    {
        r_out = ((h_act * h_act) * f[index] - 4 * x[index] + x[index - 1] + x[index + 1] + x[index - width] + x[index + width]);
    }

    __syncthreads();

    if (x_pos != 0 && y_pos != 0 && x_pos < width - 1 && y_pos < height - 1)
    {
        r[index] = r_out;
    }

    __syncthreads();
}

int main()
{
    double *x, *output, *smoother_output, *f, *f_restr, *res, *matrix_one, *matrix_restr;
    double *d_x, *d_output, *d_smoother_output, *d_f, *d_f_restr, *d_res, *d_matrix_one, *d_matrix_restr;

    int n_restr = N / 2;
    int l_restr = n_restr * n_restr;
    size_t bytes = L * sizeof(double);
    size_t bytes_restr = l_restr * sizeof(double);

    // Allocate host memory
    x = new double[L];
    output = new double[L];
    smoother_output = new double[L];
    f = new double[L];
    matrix_one = new double[L];
    matrix_restr = new double[l_restr];
    f_restr = new double[l_restr];
    res = new double[L];

    int v = 30000; // Number of iterations

    // Initialize host-side data
    initialize_zeros_vector(x);
    compute_rhs(f);

    // Allocate unified memory accessible by both CPU and GPU
    cudaMallocManaged(&d_x, bytes);
    cudaMallocManaged(&d_output, bytes);
    cudaMallocManaged(&d_smoother_output, bytes);
    cudaMallocManaged(&d_f, bytes);
    cudaMallocManaged(&d_f_restr, bytes_restr);
    cudaMallocManaged(&d_res, bytes);
    cudaMallocManaged(&d_matrix_one, bytes);
    cudaMallocManaged(&d_matrix_restr, bytes_restr);

    // Initialize data (no cudaMemcpy needed)
    initialize_zeros_vector(x);
    initialize_zeros_vector(output);
    compute_rhs(f);

    initialize_zeros_vector(d_x);
    initialize_zeros_vector(d_output);
    compute_rhs(d_f);

    int block_size = N / 4;
    int num_blocks = N / block_size;
    dim3 threadsPerBlock(block_size, block_size);
    dim3 numBlocks(num_blocks, num_blocks);
    int tmp;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < v; i++)
    {
        if (i % 2 == 0)
        {
            JacobiKernel<<<numBlocks, threadsPerBlock>>>(d_x, d_output, d_f, H, W, h);
            cudaDeviceSynchronize();
            tmp = i;
        }
        else
        {
            JacobiKernel<<<numBlocks, threadsPerBlock>>>(d_output, d_x, d_f, H, W, h);
            cudaDeviceSynchronize();
            tmp = i;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "CUDA Elapsed time: " << elapsed.count() << " s\n";
    if (tmp % 2 == 0)
    {
        compute_residual(res, d_output, f);
        device_compute_residual<<<numBlocks, threadsPerBlock>>>(d_res, d_output, d_f, H, W, h);
        cudaDeviceSynchronize();
        std::cout << "sequential Residual norm: " << vector_norm(res) << std::endl;
        std::cout << "device Residual norm: " << vector_norm(d_res) << std::endl;
    }
    else
    {
        compute_residual(res, d_x, f);
        device_compute_residual<<<numBlocks, threadsPerBlock>>>(d_res, d_x, d_f, H, W, h);
        cudaDeviceSynchronize();
        std::cout << "Residual norm: " << vector_norm(res) << std::endl;
        std::cout << "device Residual norm: " << vector_norm(d_res) << std::endl;
    }
    // sequential Jacoby
    /*

    initialize_zeros_vector(res);
    start = std::chrono::high_resolution_clock::now();
    JacobiSequential(x, output, f, v, H, W, h, L);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Sequential Elapsed time: " << elapsed.count() << " s\n";
    compute_residual(res, output, f);
    std::cout << "Residual norm: " << vector_norm(res) << std::endl;

    */

    return 0;
}
