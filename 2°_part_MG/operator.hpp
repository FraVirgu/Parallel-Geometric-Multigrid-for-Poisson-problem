#include "main.hpp"
#include "cg.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
#include <Eigen/Dense>
using namespace Eigen;

void enforce_dirichlet_boundary(double *u, int n)
{
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            if (i == 0 || i == n - 1 || j == 0 || j == n - 1)
                u[j * n + i] = 0.0;
        }
    }
}

void poisson_direct_solver(int n, double a, double *f_func, double *solution_out)
{
    int N = n * n;
    double h = 1.0 / (n - 1);
    MatrixXd A = MatrixXd::Zero(N, N);
    VectorXd b = VectorXd::Zero(N);

    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            int idx = j * n + i;

            if (i == 0 || i == n - 1 || j == 0 || j == n - 1)
            {
                A(idx, idx) = 1.0;
                b(idx) = 0.0;
            }
            else
            {
                A(idx, idx) = 4.0 * a / (h * h);
                A(idx, idx - 1) = -a / (h * h);
                A(idx, idx + 1) = -a / (h * h);
                A(idx, idx - n) = -a / (h * h);
                A(idx, idx + n) = -a / (h * h);
                b(idx) = f_func[idx];
            }
        }
    }

    VectorXd phi = A.colPivHouseholderQr().solve(b);
    for (int i = 0; i < N; ++i)
        solution_out[i] = phi(i);
}

void prolungator(double *input, double *output, int input_H, int input_W, int output_H, int output_W)
{
    for (int i = 0; i < output_H * output_W; ++i)
        output[i] = 0.0;

    for (int i = 0; i < input_H - 1; ++i)
    {
        for (int j = 0; j < input_W - 1; ++j)
        {
            double val = input[i * input_W + j];

            output[(2 * i) * output_W + (2 * j)] += val;
            output[(2 * i + 1) * output_W + (2 * j)] += 0.5 * (val + input[(i + 1) * input_W + j]);
            output[(2 * i) * output_W + (2 * j + 1)] += 0.5 * (val + input[i * input_W + (j + 1)]);
            output[(2 * i + 1) * output_W + (2 * j + 1)] += 0.25 * (val + input[(i + 1) * input_W + j] +
                                                                    input[i * input_W + (j + 1)] +
                                                                    input[(i + 1) * input_W + (j + 1)]);
        }
    }

    enforce_dirichlet_boundary(output, output_W);
}

void restriction(double *input, double *output, int input_H, int input_W, int output_H, int output_W)
{
    for (int i = 0; i < output_H * output_W; ++i)
        output[i] = 0.0;

    for (int i = 1; i < output_H - 1; ++i)
    {
        for (int j = 1; j < output_W - 1; ++j)
        {
            int fine_y = 2 * i;
            int fine_x = 2 * j;

            double result =
                (1.0 / 16.0) * (input[(fine_y - 1) * input_W + (fine_x - 1)] +
                                input[(fine_y - 1) * input_W + (fine_x + 1)] +
                                input[(fine_y + 1) * input_W + (fine_x - 1)] +
                                input[(fine_y + 1) * input_W + (fine_x + 1)]) +
                (1.0 / 8.0) * (input[(fine_y - 1) * input_W + (fine_x)] +
                               input[(fine_y + 1) * input_W + (fine_x)] +
                               input[(fine_y)*input_W + (fine_x - 1)] +
                               input[(fine_y)*input_W + (fine_x + 1)]) +
                (1.0 / 4.0) * input[fine_y * input_W + fine_x];

            output[i * output_W + j] = result;
        }
    }

    enforce_dirichlet_boundary(output, output_W);
}

void MG(double *output, double *initial_solution, double *smoother_output, double *f, double *smoother_residual,
        int v1, int v2, int level, int n, int l, int width, int height, double h_actual, int alfa = 1)
{
    Jacobi(initial_solution, smoother_output, f, v1, height, width, h_actual, l);
    iteration_performed += v1;
    enforce_dirichlet_boundary(smoother_output, n);
    dynamic_compute_residual(smoother_residual, smoother_output, f, width, height, h_actual);

    cout << "level : " << level << " Residual norm: " << dynamic_vector_norm(smoother_residual, l) << endl;

    int n_succ = n / 2;
    int l_succ = n_succ * n_succ;
    double h_succ = 1.0 / (n_succ - 1);

    double *r_H = new double[l_succ];
    restriction(smoother_residual, r_H, height, width, n_succ, n_succ);

    double *initial_solution_H = new double[l_succ];
    double *delta_H = new double[l_succ];
    double *smoother_output_H = new double[l_succ];
    double *smoother_residual_H = new double[l_succ];

    if (n_succ == 8)
    {
        poisson_direct_solver(n_succ, a, r_H, delta_H);
    }

    else
        for (int i = 0; i < alfa; ++i)
        {

            MG(delta_H, initial_solution_H, smoother_output_H, r_H, smoother_residual_H,
               v1, v2, level + 1, n_succ, l_succ, n_succ, n_succ, h_succ, a);
        }

    double *delta_h = new double[l];
    prolungator(delta_H, delta_h, n_succ, n_succ, height, width);

    cout << "level : " << level << "   ||delta_H|| = " << dynamic_vector_norm(delta_H, l_succ)
         << ", ||delta_h|| = " << dynamic_vector_norm(delta_h, l) << endl;

    double omega = 1; // Try values between 0.5 and 1.0
    for (int i = 0; i < l; ++i)
        smoother_output[i] += omega * delta_h[i];

    enforce_dirichlet_boundary(smoother_output, n);

    dynamic_compute_residual(smoother_residual, smoother_output, f, width, height, h_actual);
    cout << "level : " << level << "  Residual norm (after correction): " << dynamic_vector_norm(smoother_residual, l) << endl;

    Jacobi(smoother_output, output, f, v2, height, width, h_actual, l);
    iteration_performed += v2;
    enforce_dirichlet_boundary(output, n);

    dynamic_compute_residual(smoother_residual, output, f, width, height, h_actual);
    cout << "level : " << level << " Residual norm (after post-smooth): " << dynamic_vector_norm(smoother_residual, l) << endl;

    delete[] r_H;
    delete[] initial_solution_H;
    delete[] delta_H;
    delete[] smoother_output_H;
    delete[] smoother_residual_H;
    delete[] delta_h;
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
