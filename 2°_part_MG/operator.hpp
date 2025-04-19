#include "main.hpp"
#include "cg.hpp"
#include "jacobian.hpp"

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

// Multigrid solver
void MG(double *output, double *initial_solution, double *smoother_output, double *f, double *smoother_residual, int v1, int v2, int level, int n, int l, int weight, int height, double h_actual, int alfa = 1)
{
    // !!! NB :
    // In MG the output is the first parameter, in Jacibi2 the output is the second parameter ( not well written )

    // Pre-smoothing
    Jacobian(initial_solution, smoother_output, f, smoother_residual, v1, height, weight, h_actual, l);
    double residual_norm = dynamic_compute_vector_norm(smoother_residual, l);
    double f_residual_norm = dynamic_compute_vector_norm(f, l);
    double first_norm_residual = residual_norm / f_residual_norm;
    // cout << "level : " << level << "residual norm: " << norm_residual << endl;
    //  Restriction
    int n_succ,
        l_succ, weight_succ, height_succ;
    double h_succ;
    n_succ = n / 2;
    l_succ = n_succ * n_succ;
    weight_succ = n_succ;
    height_succ = n_succ;
    h_succ = 1.0 / (n_succ - 1);
    double *r_H = new double[l_succ];
    dynamic_initialize_zeros_vector(r_H, l_succ);
    restriction(smoother_residual, r_H, height, weight, height_succ, weight_succ);
    /*
cout << "smoother_residual_norm: " << dynamic_compute_vector_norm(smoother_residual, l) << endl;
    cout << "r_H_norm: " << dynamic_compute_vector_norm(r_H, l_succ) << endl;
    */

    // Initialize vectors for coarse grid
    double *initial_solution_H = new double[l_succ];
    double *delta_H = new double[l_succ];
    double *smoother_output_H = new double[l_succ];
    double *smoother_residual_H = new double[l_succ];
    dynamic_initialize_zeros_vector(initial_solution_H, l_succ);
    dynamic_initialize_zeros_vector(delta_H, l_succ);
    dynamic_initialize_zeros_vector(smoother_output_H, l_succ);
    dynamic_initialize_zeros_vector(smoother_residual_H, l_succ);

    if (n <= 8)
    {
        Jacobian(initial_solution_H, delta_H, r_H, smoother_residual_H, 50, height_succ, weight_succ, h_succ, l_succ);
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

    /*
    cout << endl
         << "level : " << level << endl;
    cout << "delta_H_norm: " << dynamic_compute_vector_norm(delta_H, l_succ) << endl;

    cout << "smoother_output_norm: " << dynamic_compute_vector_norm(smoother_output, l) << endl;
    cout << "delta_h_norm: " << dynamic_compute_vector_norm(delta_h, l) << endl;

    */
    dynamic_initialize_zeros_vector(delta_h, l);
    prolungator(delta_H, delta_h, height_succ, weight_succ, height, weight);

    for (int i = 0; i < l; i++)
    {
        smoother_output[i] += delta_h[i];
    }

    // Post-smoothing
    Jacobian(smoother_output, output, f, smoother_residual, v2, height, weight, h_actual, l);
    double second_norm_residual = dynamic_compute_vector_norm(smoother_residual, l);
    double norm_residual_now = second_norm_residual / f_residual_norm;

    //  cout << "level : " << level << "first norm residual: " << first_norm_residual << " second norm residual: " << norm_residual_now << endl;
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
        dynamic_initialize_zeros_vector(x[i], L);
        dynamic_initialize_zeros_vector(output[i], L);
        dynamic_initialize_zeros_vector(smoother_output[i], L);
        dynamic_initialize_zeros_vector(res[i], L);
        dynamic_compute_rhs(f[i], weight[i], height[i], h_act[i]);
        count = count * 2;
    }
}

void FMG(int initial_N, double **output, double **x, double **smoother_output, double **f, double **res, int *n, int *l, int *weight, int *height, double *h_act, int v1, int v2)
{
    Jacobian(x[0], output[0], f[0], res[0], 5, height[0], weight[0], h_act[0], l[0]);
    for (int i = 0; i < log2(initial_N) - 1; i++)
    {
        prolungator(output[i], x[i + 1], height[i], weight[i], height[i + 1], weight[i + 1]);
        MG(output[i + 1], x[i + 1], smoother_output[i + 1], f[i + 1], res[i + 1], v1, v2, 0, n[i + 1], l[i + 1], weight[i + 1], height[i + 1], h_act[i + 1], 1);
    }
}
