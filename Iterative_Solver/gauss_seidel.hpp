#include "head.hpp"

// Perform Gauss-Seidel iterations
bool GaussSeidel(double *x, double *f, double *r, double *residual_reached, int *number_iteration_performed, vector<double> *residuals, vector<double> *errors, double *x_true)
{
    double norm_residual;
    double *err = new double[L];
    double norm_error;

    //  Compute initial residual
    compute_residual(r, x, f);
    norm_residual = vector_norm(r);
    residuals->push_back(norm_residual);

    //  Compute initial error
    compute_difference(err, x, x_true);
    norm_error = vector_norm(err) / vector_norm(x_true);
    errors->push_back(norm_error);

    for (int i = 0; i < MAX_ITERATION; i++)
    {
        // Perform Gauss-Seidel iteration (update x in place)
        for (int y = 1; y < H - 1; y++)
        {
            for (int x_pos = 1; x_pos < W - 1; x_pos++)
            {
                int index = y * W + x_pos;
                x[index] = 0.25 * ((h * h * f[index]) + x[index - 1] + x[index + 1] + x[index - W] + x[index + W]);
            }
        }

        // Compute new residual
        compute_residual(r, x, f);
        norm_residual = vector_norm(r);
        residuals->push_back(norm_residual);
        *residual_reached = norm_residual;

        // Compute the error
        compute_difference(err, x, x_true);
        norm_error = vector_norm(err) / vector_norm(x_true);
        errors->push_back(norm_error);

        //  Convergence check (residual)
        if (norm_residual < EPSILON)
        {
            *number_iteration_performed = i;
            return true;
        }

        //  Convergence check (error)
        if (norm_error < EPSILON)
        {
            *number_iteration_performed = i;
            return true;
        }
    }

    return false;
}
