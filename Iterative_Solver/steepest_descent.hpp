#include "head.hpp"
// Compute optimal alpha (alpha_opt = (r^T r) / (r^T A r)), using the stencil
double compute_alpha_opt(double *r)
{
    double numerator = 0.0;
    double denominator = 0.0;
    double temp_r;

    for (int y = 1; y < H - 1; y++)
    {
        for (int x_pos = 1; x_pos < W - 1; x_pos++)
        {
            int index = y * W + x_pos;
            numerator += r[index] * r[index];

            // Apply Laplacian operator (A * r) using the same stencil
            temp_r = -4 * r[index] + r[index - 1] + r[index + 1] + r[index - W] + r[index + W];
            denominator += r[index] * temp_r;
        }
    }
    return numerator / denominator;
}

// THIS IS NOT THE CG IS JUST THE GRADIENT DESCENT !!!

// Perform Conjugate Gradient iterations
bool Steepest_Descent(double *x, double *f, double *r, int *number_iteration_performed, double *residual_reached, vector<double> *residuals, vector<double> *errors, double *x_true)
{

    double alpha_opt;
    double norm_residual;
    double res_tmp;
    double err_tmp;
    double norm_error;
    double *err = new double[L];
    // Compute initial residual
    compute_residual(r, x, f);
    norm_residual = vector_norm(r);
    res_tmp = norm_residual;
    residuals->push_back(norm_residual);

    //  Compute initial error
    compute_difference(err, x, x_true);
    norm_error = vector_norm(err) / vector_norm(x_true);
    err_tmp = norm_error;
    errors->push_back(norm_error);

    for (int i = 0; i < MAX_ITERATION; i++)
    {
        alpha_opt = compute_alpha_opt(r);

        // Update solution: x[k+1] = x[k] - alpha_opt * r[k]
        for (int j = 0; j < L; j++)
        {
            x[j] -= alpha_opt * r[j];
        }

        // Compute new residual
        compute_residual(r, x, f);
        norm_residual = vector_norm(r);

        // Compute the error
        compute_difference(err, x, x_true);
        norm_error = vector_norm(err) / vector_norm(x_true);

        // Update residual reached
        if (norm_residual <= res_tmp)
        {
            res_tmp = norm_residual;
            residuals->push_back(norm_residual);
            *residual_reached = norm_residual;
        }
        // update error reached
        if (norm_error <= err_tmp)
        {
            err_tmp = norm_error;
            errors->push_back(norm_error);
        }

        // Convergence check (residual)
        if (norm_residual < EPSILON)
        {
            *number_iteration_performed = i;
            return true;
        }

        // Convergence check (error)
        if (norm_error < EPSILON)
        {
            *number_iteration_performed = i;
            return true;
        }
    }

    return false;
}
