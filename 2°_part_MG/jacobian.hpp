#ifndef JACOBIAN_HPP
#define JACOBIAN_HPP
#include "main.hpp"
// THIS METHOD HAS BEEN TESTED
void Jacobian(double *x, double *x_new, double *f, double *r, int v, int height, int weight, double h_act, int l)
{
    double norm_residual;
    double norm_error;

    int n_iteration = v;

    for (int i = 0; i < n_iteration; i++)
    {
        // Perform Jacobi iteration
        for (int y = 1; y < height - 1; y++)
        {
            for (int x_pos = 1; x_pos < weight - 1; x_pos++)
            {
                int index = y * weight + x_pos;
                x_new[index] = 0.25 * ((h_act * h_act) * (f[index]) + x[index - 1] + x[index + 1] + x[index - weight] + x[index + weight]);
            }
        }

        // Copy x_new to x for next iteration
        for (int j = 0; j < l; j++)
        {
            x[j] = x_new[j];
        }
    }

    // Compute new residual
    dynamic_compute_residual(r, x_new, f, weight, height, h_act);
}

#endif
