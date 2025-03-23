#ifndef HEAD_HPP
#define HEAD_HPP
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include "globals.hpp"

using namespace std;

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

// Compute residual norm
double compute_residual_norm(double *x, double *f)
{
    double norm = 0.0;
    norm = 0.0;
    for (int y = 1; y < H - 1; y++)
    {
        for (int x_pos = 1; x_pos < W - 1; x_pos++)
        {
            int index = y * W + x_pos;
            norm += pow((h * h) * f[index] - 4 * x[index] + x[index - 1] + x[index + 1] + x[index - W] + x[index + W], 2);
        }
    }
    return sqrt(norm);
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

void compute_difference(double *error, double *x, double *x_true)
{
    for (int i = 0; i < L; i++)
    {
        error[i] = x[i] - x_true[i];
    }
}

// Initialize vector x to zero
void initialize_zeros_vector(double *x)
{
    for (int i = 0; i < L; i++)
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

// Compute Laplacian using finite differences
void compute_laplacian(double *f, double (*func)(double, double))
{
    double dx = a / W;
    double dy = a / H;
    double dx2 = dx * dx;
    double dy2 = dy * dy;

    for (int y = 1; y < H - 1; y++)
    {
        for (int x = 1; x < W - 1; x++)
        {
            double x_val = x * dx;
            double y_val = y * dy;

            // Discrete Laplacian: -1/h^2 * (u[i-1] + u[i+1] + u[j-1] + u[j+1] - 4*u[i,j])
            f[y * W + x] = -(func(x_val - dx, y_val) + func(x_val + dx, y_val) +
                             func(x_val, y_val - dy) + func(x_val, y_val + dy) -
                             4 * func(x_val, y_val)) /
                           dx2;
        }
    }

    // Set boundary conditions to zero
    for (int x = 0; x < W; x++)
    {
        f[x] = 0.0;               // Top boundary
        f[(H - 1) * W + x] = 0.0; // Bottom boundary
    }
    for (int y = 0; y < H; y++)
    {
        f[y * W] = 0.0;           // Left boundary
        f[y * W + (W - 1)] = 0.0; // Right boundary
    }
}

// Add the missing #endif directive
#endif