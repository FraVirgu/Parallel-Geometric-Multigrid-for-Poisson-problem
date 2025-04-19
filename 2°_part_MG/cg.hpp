#include "main.hpp"

void compute_difference(double *error, double *x, double *x_true)
{
    for (int i = 0; i < L; i++)
    {
        error[i] = x[i] - x_true[i];
    }
}

void enforce_dirichlet_boundary(double *v)
{
    for (int y = 0; y < H; y++)
    {
        for (int x_pos = 0; x_pos < W; x_pos++)
        {
            if (y == 0 || y == H - 1 || x_pos == 0 || x_pos == W - 1)
            {
                v[y * W + x_pos] = 0.0;
            }
        }
    }
}

// Function to compute the standard inner product
double compute_inner_product(const double *v1, const double *v2)
{
    double result = 0.0;
    for (int i = 0; i < L; i++)
    {
        result += v1[i] * v2[i];
    }
    return result;
}

// Function to compute the inner product with A where A * r is calculated using the Laplacian
void compute_inner_product_with_A(double *r, double *result)
{
    for (int y = 1; y < H - 1; y++)
    {
        for (int x_pos = 1; x_pos < W - 1; x_pos++)
        {
            int index = y * W + x_pos;

            // Apply Laplacian operator (A * r) using the same stencil
            result[index] = (4 * r[index] - r[index - 1] - r[index + 1] - r[index - W] - r[index + W]) / (h * h);
        }
    }
}

bool conjugate_gradient(double *x, double *f, double *r, double *p_d, double *Ap_d, int *number_iteration_performed, double *residual_reached)
{
    double alpha, beta, res_tmp, norm_residual, err_tmp, norm_error;
    double *err = new double[L];

    dynamic_initialize_zeros_vector(x, L);
    // Compute initial residual: r = f - A * x
    dynamic_compute_residual(r, x, f, W, H, h);
    norm_residual = dynamic_compute_vector_norm(r, L); // divide by the norm of the right-hand side
    res_tmp = norm_residual;

    // Initialize search direction: p = r
    for (int j = 0; j < L; j++)
    {
        p_d[j] = r[j];
    }

    for (int i = 0; i < MAX_ITERATION; i++)
    {

        // Compute p_d^(k)^Tr^(k)
        double rTr = compute_inner_product(r, r); // Store r^T * r
        // Compute A * p
        compute_inner_product_with_A(p_d, Ap_d);

        // Compute step size: alpha = (r^T * r) / (p^T * A * p)
        double pAp = compute_inner_product(p_d, Ap_d);
        alpha = rTr / pAp;

        // Update solution: x[k+1] = x[k] + alpha * p_d
        for (int j = 0; j < L; j++)
        {
            x[j] += alpha * p_d[j];
        }

        // Update residual: r[k+1] = r[k] - alpha * A * p
        for (int j = 0; j < L; j++)
        {
            r[j] -= alpha * Ap_d[j];
        }

        // Compute new residual
        norm_residual = dynamic_compute_vector_norm(r, L); // divide by the norm of the right-hand side
        cout << "Iteration " << i << ": Residual norm = " << norm_residual << endl;
        enforce_dirichlet_boundary(x);

        // Convergence check
        if (norm_residual < EPSILON)
        {
            *residual_reached = norm_residual;
            *number_iteration_performed = i;

            return true;
        }

        // Compute new beta: beta = (r[k+1]^T * r[k+1]) / (r[k]^T * r[k])
        double rTr_new = compute_inner_product(r, r);
        beta = rTr_new / rTr;

        // Update search direction: p[k+1] = r[k+1] + beta * p[k]
        for (int j = 0; j < L; j++)
        {
            p_d[j] = r[j] + beta * p_d[j];
        }
    }

    return false;
}
