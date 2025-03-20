#ifndef MAIN_HPP
#define MAIN_HPP
#include <iostream>
#include <vector>
#include <cmath>
#include "globals.hpp"
#include <chrono>
using namespace std;

// Jacobi solver
bool Jacobi(double *x, double *x_new, double *f, int v, int height, int weight, double h_act, int l)
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

// Initialize helper functions
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

// Multigrid solver helper functions
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
void dynamic_initialize_zeros_vector(double *x, int l)
{
    for (int i = 0; i < l; i++)
    {
        x[i] = 0.0;
    }
}
void dynamic_compute_rhs(double *f, int weight, int height)
{
    double dx = a / weight;
    double dy = a / height;
    double factor = (M_PI * M_PI / (a * a)) * (p * p + q * q);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < weight; x++)
        {
            double x_val = x * dx;
            double y_val = y * dy;

            // Apply Dirichlet boundary condition: f = 0 at the boundaries
            if (x == 0 || x == weight - 1 || y == 0 || y == height; -1)
            {
                f[y * weight + x] = 0.0;
            }
            else
            {
                f[y * weight + x] = factor * sin(p * M_PI * x_val / a) * sin(q * M_PI * y_val / a);
            }
        }
    }
}

#endif