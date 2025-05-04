#include "main.hpp"
#include "cg.hpp"

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
                            // output[y * output_W + x] = 0.0; // Enforce boundary condition
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

void restriction(double *input, double *output, int input_H, int input_W, int output_H, int output_W)
{
    double weight[9] = {0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25};
    for (int i = 0; i < output_H; i++)
    {
        for (int j = 0; j < output_W; j++)
        {
            if (i == 0 || i == output_H - 1 || j == 0 || j == output_W - 1) // Enforce boundary condition
            {
                // output[i * output_W + j] = 0.0;
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

// Multigrid solver
void MG(double *output, double *initial_solution, double *smoother_output, double *f, double *smoother_residual, int v1, int v2, int level, int n, int l, int weight, int height, double h_actual, int alfa = 1)
{
    // !!! NB :
    // In MG the output is the first parameter, in Jacibi2 the output is the second parameter ( not well written )

    // Pre-smoothing
    Jacobi(initial_solution, smoother_output, f, v1, height, weight, h_actual, l);
    dynamic_compute_residual(smoother_residual, smoother_output, f, weight, height, h_actual);

    // Restriction
    int n_succ, l_succ, weight_succ, height_succ;
    double h_succ;
    n_succ = n / 2;
    l_succ = n_succ * n_succ;
    weight_succ = n_succ;
    height_succ = n_succ;
    h_succ = 1.0 / (n_succ - 1);
    double *r_H = new double[l_succ];
    // dynamic_initialize_zeros_vector(r_H, l_succ);
    restriction(smoother_residual, r_H, height, weight, height_succ, weight_succ);

    // Initialize vectors for coarse grid
    double *initial_solution_H = new double[l_succ];
    double *delta_H = new double[l_succ];
    double *smoother_output_H = new double[l_succ];
    double *smoother_residual_H = new double[l_succ];
    // dynamic_initialize_zeros_vector(initial_solution_H, l_succ);
    // dynamic_initialize_zeros_vector(delta_H, l_succ);
    // dynamic_initialize_zeros_vector(smoother_output_H, l_succ);
    // dynamic_initialize_zeros_vector(smoother_residual_H, l_succ);

    if (n <= 32)
    {
        Jacobi(initial_solution_H, delta_H, r_H, 10, height_succ, weight_succ, h_succ, l_succ);
    }
    else
    {
        for (int i = 0; i < alfa; i++)
        {
            MG(delta_H, initial_solution_H, smoother_output_H, r_H, smoother_residual_H, v1, v2, level + 1, n_succ, l_succ, weight_succ, height_succ, h_succ);
            for (int j = 0; j < l_succ; j++)
            {
                initial_solution_H[j] = delta_H[j];
            }
        }
    }

    double *delta_h = new double[l];
    prolungator(delta_H, delta_h, height_succ, weight_succ, height, weight);

    for (int i = 0; i < l; i++)
    {
        smoother_output[i] += delta_h[i];
    }

    // Post-smoothing
    Jacobi(smoother_output, output, f, v2, height, weight, h_actual, l);
}

void update_global_parameter(int n)
{
    N = n;
    W = n;
    H = n;
    L = n * n;
    h = 1.0 / (n - 1);
}

void initialize_FG(int initial_N, double **x, double **output, double **smoother_output, double **f, double **res, int *n, int *l, int *weight, int *height, double *h_act)
{

    int count = 2;
    for (int i = 0; i < log2(initial_N); i++)
    {
        n[i] = count;
        l[i] = count * count;
        weight[i] = count;
        height[i] = count;
        h_act[i] = 1.0 / (count - 1);
        update_global_parameter(count);
        x[i] = new double[L];
        output[i] = new double[L];
        smoother_output[i] = new double[L];
        f[i] = new double[L];
        res[i] = new double[L];
        initialize_zeros_vector(x[i]);
        initialize_zeros_vector(output[i]);
        initialize_zeros_vector(smoother_output[i]);
        initialize_zeros_vector(res[i]);
        compute_rhs(f[i]);
        count = count * 2;
    }
}

void FMG(int initial_N, double **output, double **x, double **smoother_output, double **f, double **res, int *n, int *l, int *weight, int *height, double *h_act, int v1, int v2)
{
    Jacobi(x[0], output[0], f[0], 2, height[0], weight[0], h_act[0], l[0]);
    for (int i = 0; i < log2(initial_N) - 1; i++)
    {
        prolungator(output[i], x[i + 1], height[i], weight[i], height[i + 1], weight[i + 1]);
        MG(output[i + 1], x[i + 1], smoother_output[i + 1], f[i + 1], res[i + 1], v1, v2, 0, n[i + 1], l[i + 1], weight[i + 1], height[i + 1], h_act[i + 1]);
    }
}
