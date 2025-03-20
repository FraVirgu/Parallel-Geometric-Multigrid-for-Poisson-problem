#include "head.hpp"

// Perform Jacobi iterations
bool Jacobian(double *x, double *x_new, double *f, double *r, double *residual_reached, int *number_iteration_performed, vector<double> *residuals, vector<double> *errors, double *x_true)
{
    double norm_residual;
    double norm_error;
    double *err = new double[L];

    //  Compute initial residual
    compute_residual(r, x, f);
    norm_residual = vector_norm(r); // divide by the norm of the right-hand side
    residuals->push_back(norm_residual);

    //  Compute initial error
    compute_difference(err, x, x_true);
    norm_error = vector_norm(err) / vector_norm(x_true);
    errors->push_back(norm_error);

    if (!fix_iteration)
    {

        for (int i = 0; i < MAX_ITERATION; i++)
        {
            // Perform Jacobi iteration
            for (int y = 1; y < H - 1; y++)
            {
                for (int x_pos = 1; x_pos < W - 1; x_pos++)
                {
                    int index = y * W + x_pos;
                    x_new[index] = 0.25 * ((h * h * f[index]) + x[index - 1] + x[index + 1] + x[index - W] + x[index + W]);
                }
            }

            // Compute new residual
            compute_residual(r, x_new, f);
            norm_residual = vector_norm(r);
            residuals->push_back(norm_residual);
            *residual_reached = norm_residual;

            //  Compute the error
            compute_difference(err, x_new, x_true);
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

            // Copy x_new to x for next iteration
            for (int j = 0; j < L; j++)
            {
                x[j] = x_new[j];
            }
        }
        return false;
    }
    else
    {
        for (int i = 0; i < number_fixed_iteration; i++)
        {
            // Perform Jacobi iteration
            for (int y = 1; y < H - 1; y++)
            {
                for (int x_pos = 1; x_pos < W - 1; x_pos++)
                {
                    int index = y * W + x_pos;
                    x_new[index] = 0.25 * ((h * h * f[index]) + x[index - 1] + x[index + 1] + x[index - W] + x[index + W]);
                }
            }

            // Compute new residual
            compute_residual(r, x_new, f);
            norm_residual = vector_norm(r);
            residuals->push_back(norm_residual);
            *residual_reached = norm_residual;

            //  Compute the error
            compute_difference(err, x_new, x_true);
            norm_error = vector_norm(err) / vector_norm(x_true);

            errors->push_back(norm_error);

            // Copy x_new to x for next iteration
            for (int j = 0; j < L; j++)
            {
                x[j] = x_new[j];
            }
        }
        return true;
    }
}
