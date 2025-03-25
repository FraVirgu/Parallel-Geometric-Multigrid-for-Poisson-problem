#include "globals.hpp"
#include "cpu_mg.hpp"

__global__ void jacobi_kernel(double *x, double *x_new, double *f, int height, int width, double h_act)
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

int JacobiCall(double *d_x, double *d_output, double *d_f, int height, int weight, double h_act, int v)
{
    int block_size = N / 4;
    int num_blocks = N / block_size;
    dim3 threadsPerBlock(block_size, block_size);
    dim3 numBlocks(num_blocks, num_blocks);
    int tmp;
    for (int i = 0; i < v; i++)
    {
        if (i % 2 == 0)
        {
            jacobi_kernel<<<numBlocks, threadsPerBlock>>>(d_x, d_output, d_f, H, W, h);
            cudaDeviceSynchronize();
            tmp = i;
        }
        else
        {
            jacobi_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_x, d_f, H, W, h);
            cudaDeviceSynchronize();
            tmp = i;
        }
    }

    return tmp;
}

void ResidualCall(double *d_r, double *d_x, double *d_f, int height, int weight, double h_act)
{
    int block_size = N / 4;
    int num_blocks = N / block_size;
    dim3 threadsPerBlock(block_size, block_size);
    dim3 numBlocks(num_blocks, num_blocks);
    device_compute_residual<<<numBlocks, threadsPerBlock>>>(d_r, d_x, d_f, H, W, h);
    cudaDeviceSynchronize();
}

void RestrictionCall(double *input, double *output, int input_H, int input_W, int output_H, int output_W, int n_restr)
{
    // GPU restriction using tiled kernel
    int block_size_restr = N / 2; // Block size for GPU kernel
    dim3 threadsPerBlock_restr(block_size_restr, block_size_restr);
    dim3 numBlocks_restr((n_restr + block_size_restr - 1) / block_size_restr,
                         (n_restr + block_size_restr - 1) / block_size_restr);

    // GPU restriction using basic kernel
    restriction_kernel_basic<<<numBlocks_restr, threadsPerBlock_restr>>>(
        input, output, H, W, n_restr, n_restr);

    cudaDeviceSynchronize();
}

void ProlungatorCall(double *input, double *output, int input_H, int input_W, int output_H, int output_W)
{
    int block_size_prol = N / 2; // Block size for GPU kernel
    dim3 threadsPerBlock_prol(block_size_prol, block_size_prol);
    dim3 numBlocks_prol((output_W + block_size_prol - 1) / block_size_prol,
                        (output_H + block_size_prol - 1) / block_size_prol);

    prolungator_kernel<<<numBlocks_prol, threadsPerBlock_prol>>>(
        input, output, input_H, input_W, output_H, output_W);

    cudaDeviceSynchronize();
}

void MG(double *output, double *initial_solution, double *smoother_output, double *f, double *smoother_residual, int v1, int v2, int level, int n, int l, int weight, int height, double h_actual, int alfa = 1)
{
    int tmp = JacobiCall(initial_solution, smoother_output, f, height, weight, h_actual, v1);
    if (tmp % 2 == 0)
    {
        ResidualCall(smoother_residual, smoother_output, f, height, weight, h_actual);
    }
    else
    {
        ResidualCall(smoother_residual, initial_solution, f, height, weight, h_actual);
    }

    // Restriction
    int n_succ, l_succ, weight_succ, height_succ;
    double h_succ;
    n_succ = n / 2;
    l_succ = n_succ * n_succ;
    weight_succ = n_succ;
    height_succ = n_succ;
    h_succ = 1.0 / (n_succ - 1);
    double *r_H;
    cudaMallocManaged(&r_H, l_succ * sizeof(double));
    RestrictionCall(smoother_residual, r_H, height, weight, height_succ, weight_succ, n_succ);

    // Initialize vectors for coarse grid
    double *initial_solution_H;
    double *delta_H;
    double *smoother_output_H;
    double *smoother_residual_H;

    cudaMallocManaged(&initial_solution_H, l_succ * sizeof(double));
    cudaMallocManaged(&delta_H, l_succ * sizeof(double));
    cudaMallocManaged(&smoother_output_H, l_succ * sizeof(double));
    cudaMallocManaged(&smoother_residual_H, l_succ * sizeof(double));
    dynamic_initialize_zeros_vector(initial_solution_H, l_succ);
    dynamic_initialize_zeros_vector(delta_H, l_succ);
    dynamic_initialize_zeros_vector(smoother_output_H, l_succ);
    dynamic_initialize_zeros_vector(smoother_residual_H, l_succ);

    if (n <= 2)
    {
        cpu_jacobi(initial_solution_H, delta_H, r_H, 1, height_succ, weight_succ, h_succ, l_succ);
    }
    else
    {
        for (int i = 0; i < alfa; i++)
        {

            if (n_succ <= 16)
            {
                cpu_MG(delta_H, initial_solution_H, smoother_output_H, r_H, smoother_residual_H, v1, v2, level + 1, n_succ, l_succ, weight_succ, height_succ, h_succ);
                for (int j = 0; j < l_succ; j++)
                {
                    initial_solution_H[j] = delta_H[j];
                }
            }
            else
            {
                MG(delta_H, initial_solution_H, smoother_output_H, r_H, smoother_residual_H, v1, v2, level + 1, n_succ, l_succ, weight_succ, height_succ, h_succ);
                for (int j = 0; j < l_succ; j++)
                {
                    initial_solution_H[j] = delta_H[j];
                }
            }
        }
    }

    // Prolongation
    double *delta_h;
    cudaMallocManaged(&delta_h, l * sizeof(double));
    ProlungatorCall(delta_H, delta_h, height_succ, weight_succ, height, weight);
    for (int i = 0; i < l; i++)
    {
        smoother_output[i] += delta_h[i];
    }

    JacobiCall(smoother_output, output, f, v2, height, weight, h_actual);
}

void MG_CALL()
{

    double *d_output, *d_initial_solution, *d_smoother_output, *d_f, *d_smoother_residual;
    int v1 = 100, v2 = 200;
    size_t bytes = L * sizeof(double);

    auto start = chrono::high_resolution_clock::now();

    cudaMallocManaged(&d_output, bytes);
    cudaMallocManaged(&d_initial_solution, bytes);
    cudaMallocManaged(&d_smoother_output, bytes);
    cudaMallocManaged(&d_f, bytes);
    cudaMallocManaged(&d_smoother_residual, bytes);

    initialize_zeros_vector(d_initial_solution);
    initialize_zeros_vector(d_output);
    initialize_zeros_vector(d_smoother_output);
    initialize_zeros_vector(d_smoother_residual);
    compute_rhs(d_f);

    MG(d_output, d_initial_solution, d_smoother_output, d_f, d_smoother_residual, v1, v2, 0, N, L, N, N, H);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Multigrid time: " << elapsed.count() * 1000 << " ms" << endl;
    double *res = new double[L];
    compute_residual(res, d_output, d_f);
    cout << "Residual norm: " << vector_norm(res) << endl;

    // Free allocated memory
    cudaFree(d_output);
    cudaFree(d_initial_solution);
    cudaFree(d_smoother_output);
    cudaFree(d_f);
    cudaFree(d_smoother_residual);
}

int main()
{
    MG_CALL();

    return 0;
}