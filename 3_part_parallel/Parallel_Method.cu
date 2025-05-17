#include "iostream"
#include <cuda_runtime.h>
#include "../globals.hpp"
#include "../2_part_MG/MultiGrid.hpp"

__global__ void jacobi_kernel(double *x, double *f, int height, int width, double h_act)
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

    if (x_pos != 0 && y_pos != 0 && x_pos < width - 1 && y_pos < height - 1)
    {
        x[index] = x_out;
    }
}

__global__ void device_compute_residual(double *r, double *x, double *f, int height, int width, double h_act)
{
    int x_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int y_pos = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y_pos * width + x_pos;
    double r_out = 0.0;

    if (x_pos != 0 && y_pos != 0 && x_pos < width - 1 && y_pos < height - 1)
    {
        r_out = f[index] - (1.0 / (h_act * h_act)) * (4 * x[index] - x[index - 1] - x[index + 1] - x[index - width] - x[index + width]);
    }

    __syncthreads();

    if (x_pos != 0 && y_pos != 0 && x_pos < width - 1 && y_pos < height - 1)
    {
        r[index] = r_out;
    }

    __syncthreads();
}

__global__ void restriction_kernel_full_weighting(const double *__restrict__ input, double *__restrict__ output,
                                                  int input_H, int input_W, int output_H, int output_W)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x <= 0 || out_x >= output_W - 1 || out_y <= 0 || out_y >= output_H - 1)
        return;

    int in_x = 2 * out_x;
    int in_y = 2 * out_y;

    int idx_out = out_y * output_W + out_x;

    // Index helpers
    int idx_center = in_y * input_W + in_x;
    int idx_left = idx_center - 1;
    int idx_right = idx_center + 1;
    int idx_top = idx_center - input_W;
    int idx_bottom = idx_center + input_W;
    int idx_tl = idx_top - 1;
    int idx_tr = idx_top + 1;
    int idx_bl = idx_bottom - 1;
    int idx_br = idx_bottom + 1;

    // Apply full-weighting stencil
    output[idx_out] =
        0.25 * input[idx_center] +
        0.125 * (input[idx_left] + input[idx_right] + input[idx_top] + input[idx_bottom]) +
        0.0625 * (input[idx_tl] + input[idx_tr] + input[idx_bl] + input[idx_br]);
}
__global__ void prolungator_kernel(const double *__restrict__ input, double *__restrict__ output,
                                   int input_H, int input_W, int output_H, int output_W)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y >= output_H || x >= output_W)
        return;

    // Don't touch boundary — Dirichlet condition
    if (y == 0 || y == output_H - 1 || x == 0 || x == output_W - 1)
    {
        output[y * output_W + x] = 0.0;
        return;
    }

    double value = 0.0;
    int cx = x / 2;
    int cy = y / 2;

    // Case 1: even-even (direct injection)
    if (x % 2 == 0 && y % 2 == 0)
    {
        value = input[cy * input_W + cx];
    }

    // Case 2: odd-odd (bilinear from 4 corners)
    else if (x % 2 == 1 && y % 2 == 1)
    {
        if (cx + 1 < input_W && cy + 1 < input_H)
        {
            value = 0.25 * (input[cy * input_W + cx] +
                            input[cy * input_W + (cx + 1)] +
                            input[(cy + 1) * input_W + cx] +
                            input[(cy + 1) * input_W + (cx + 1)]);
        }
    }

    // Case 3: even row, odd column (horizontal interpolation)
    else if (x % 2 == 1 && y % 2 == 0)
    {
        if (cx + 1 < input_W)
        {
            value = 0.5 * (input[cy * input_W + cx] +
                           input[cy * input_W + (cx + 1)]);
        }
    }

    // Case 4: odd row, even column (vertical interpolation)
    else if (x % 2 == 0 && y % 2 == 1)
    {
        if (cy + 1 < input_H)
        {
            value = 0.5 * (input[cy * input_W + cx] +
                           input[(cy + 1) * input_W + cx]);
        }
    }

    output[y * output_W + x] += value;
}

class Parallel
{

public:
    static void
    ComputeJacobi(double *d_x, double *d_f, int height, int weight, double h_act, int v)
    {
        int block_size = num_thread;
        int num_blocks = height / block_size;
        if (num_blocks == 0)
            num_blocks++;
        dim3 threadsPerBlock(block_size, block_size);
        dim3 numBlocks(num_blocks, num_blocks);
        for (int i = 0; i <= v; i++)
        {
            jacobi_kernel<<<numBlocks, threadsPerBlock>>>(d_x, d_f, height, weight, h_act);
            cudaDeviceSynchronize();
        }

        return;
    }

    static void
    ComputeResidual(double *d_r, double *d_x, double *d_f, int height, int width, double h_act)
    {
        int block_size = num_thread;
        int num_blocks = height / block_size;
        if (num_blocks == 0)
            num_blocks++;
        dim3 threadsPerBlock(block_size, block_size);
        dim3 numBlocks(num_blocks, num_blocks);
        device_compute_residual<<<numBlocks, threadsPerBlock>>>(d_r, d_x, d_f, height, width, h_act);
        cudaDeviceSynchronize();
    }

    static void
    ComputeRestriction(double *fine, double *coarse, int fine_N, int coarse_N)
    {
        int block_size = num_thread;
        int num_blocks = (coarse_N + block_size - 1) / block_size;
        if (num_blocks == 0)
            num_blocks++;
        dim3 threadsPerBlock(block_size, block_size);
        dim3 numBlocks(num_blocks, num_blocks);
        restriction_kernel_full_weighting<<<numBlocks, threadsPerBlock>>>(fine, coarse, fine_N, fine_N, coarse_N, coarse_N);
        cudaDeviceSynchronize();
    }

    static void
    ComputeProlungator(double *coarse, double *fine, int coarse_N, int fine_N)
    {
        int block_size = num_thread;
        int num_blocks = fine_N / block_size;
        if (num_blocks == 0)
            num_blocks++;
        dim3 threadsPerBlock(block_size, block_size);
        dim3 numBlocks(num_blocks, num_blocks);
        prolungator_kernel<<<numBlocks, threadsPerBlock>>>(coarse, fine, coarse_N, coarse_N, fine_N, fine_N);
        cudaDeviceSynchronize();
    }
};
