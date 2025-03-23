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
