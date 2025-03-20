#include <iostream>
#include <vector>
#include <cmath>
#include "globals.hpp"
#include <chrono>

using namespace std;

// Perform Jacobi iterations
bool Jacobian_2(double *x, double *x_new, double *f, int v, int height, int weight, double h_act, int l)
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
void dynamic_initialize_zeros_vector(double *x, int l)
{
    for (int i = 0; i < l; i++)
    {
        x[i] = 0.0;
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
void initialize_zeros_vector(double *x)
{
    for (int i = 0; i < L; i++)
    {
        x[i] = 0.0;
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

// Compute residual r = f - Ax
void dynamic_compute_residual(double *r, double *x, double *f, int weight, int height, double h_actual)
{
    for (int y = 1; y < height - 1; y++)
    {
        for (int x_pos = 1; x_pos < weight - 1; x_pos++)
        {
            int index = y * weight + x_pos;
            r[index] = f[index] - (4 * x[index] - x[index - 1] - x[index + 1] - x[index - weight] - x[index + weight]) / (h_actual * h_actual);
        }
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
void MG(double *output, double *initial_solution, double *smoother_output, double *f, double *smoother_residual, int v1, int v2, int level, int n, int l, int weight, int height, double h_actual, int alfa = 2)
{
    // !!! NB :
    // In MG the output is the first parameter, in Jacibi2 the output is the second parameter ( not well written )

    // Pre-smoothing
    Jacobian_2(initial_solution, smoother_output, f, v1, height, weight, h_actual, l);
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
    dynamic_initialize_zeros_vector(r_H, l_succ);
    restriction(smoother_residual, r_H, height, weight, height_succ, weight_succ);

    // Initialize vectors for coarse grid
    double *initial_solution_H = new double[l_succ];
    double *delta_H = new double[l_succ];
    double *smoother_output_H = new double[l_succ];
    double *smoother_residual_H = new double[l_succ];
    dynamic_initialize_zeros_vector(initial_solution_H, l_succ);
    dynamic_initialize_zeros_vector(delta_H, l_succ);
    dynamic_initialize_zeros_vector(smoother_output_H, l_succ);
    dynamic_initialize_zeros_vector(smoother_residual_H, l_succ);

    if (n <= 2)
    {
        Jacobian_2(initial_solution_H, delta_H, r_H, 1, height_succ, weight_succ, h_succ, l_succ);
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
    Jacobian_2(smoother_output, output, f, v2, height, weight, h_actual, l);
}

int main()
{
    // Save initial parameters
    int initial_N = N;
    int initial_W = W, initial_H = H, initial_L = L;
    double initial_h = h;

    double *x = new double[L];
    double *output = new double[L];
    double *smoother_output = new double[L];
    double *f = new double[L];
    double *res = new double[L];
    initialize_zeros_vector(x);
    compute_rhs(f);
    cout << "JACOBI METHOD" << endl;
    auto start = chrono::high_resolution_clock::now();
    Jacobian_2(x, output, f, 10000, H, W, h, L);
    auto end = chrono::high_resolution_clock::now();
    compute_residual(res, output, f);
    cout << "Jacobi time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    cout << "Residual norm: " << vector_norm(res) << endl;

    int v1 = 400, v2 = 500, level = 0;
    cout << "MULTIGRID METHOD" << endl;
    initialize_zeros_vector(x);
    initialize_zeros_vector(output);
    initialize_zeros_vector(smoother_output);
    initialize_zeros_vector(res);
    auto start_MG = chrono::high_resolution_clock::now();
    MG(output, x, smoother_output, f, res, v1, v2, level, N, L, W, H, h);
    auto end_MG = chrono::high_resolution_clock::now();
    cout << "Multigrid time: " << chrono::duration_cast<chrono::milliseconds>(end_MG - start_MG).count() << "ms" << endl;
    initialize_zeros_vector(res);
    compute_residual(res, output, f);
    cout << "Residual norm: " << vector_norm(res) << endl;
    return 0;
}