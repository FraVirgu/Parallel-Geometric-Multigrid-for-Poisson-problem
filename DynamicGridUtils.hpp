#ifndef DYNAMIC_GRID_UTILS_HPP
#define DYNAMIC_GRID_UTILS_HPP

#include <cmath>
#include "globals.hpp"
#include "stdio.h"
#include "iostream"
class DynamicGridUtils
{
public:
    static constexpr double epsilon = 1e-3;

    // Initialize vector with zeros
    static void initialize_zeros(double *x, int l)
    {
        for (int i = 0; i < l; ++i)
            x[i] = 0.0;
    }

    // Compute the L2 norm
    static double norm(const double *f, int l)
    {
        double sum = 0.0;
        for (int i = 0; i < l; ++i)
            sum += f[i] * f[i];
        return std::sqrt(sum);
    }

    // Compute difference: err = x1 - x2
    static void compute_error(double *err, const double *x1, const double *x2, int l)
    {
        for (int i = 0; i < l; ++i)
            err[i] = x1[i] - x2[i];
    }

    // Dot product of two vectors
    static double dot(const double *v1, const double *v2, int l)
    {
        double result = 0.0;
        for (int i = 0; i < l; ++i)
            result += v1[i] * v2[i];
        return result;
    }

    // Apply 5-point Laplacian stencil: A * u -> result
    static void apply_laplacian(const double *u, double *result, int W, int H, double h)
    {
        for (int y = 1; y < H - 1; ++y)
        {
            for (int x = 1; x < W - 1; ++x)
            {
                int idx = y * W + x;
                result[idx] = (4.0 * u[idx] - u[idx - 1] - u[idx + 1] - u[idx - W] - u[idx + W]) / (h * h);
            }
        }
    }

    // Compute residual: r = f - A * x
    static void compute_residual(double *r, const double *x, const double *f, int W, int H, double h)
    {
        for (int y = 1; y < H - 1; ++y)
        {
            for (int x_pos = 1; x_pos < W - 1; ++x_pos)
            {
                int idx = y * W + x_pos;
                r[idx] = f[idx] - (1.0 / (h * h)) * (4 * x[idx] - x[idx - 1] - x[idx + 1] - x[idx - W] - x[idx + W]);
            }
        }
    }

    // Compute optimal alpha for Steepest Descent
    static double compute_alpha_opt(const double *r, int W, int H, double h)
    {
        double numerator = 0.0;
        double denominator = 0.0;

        for (int y = 1; y < H - 1; ++y)
        {
            for (int x_pos = 1; x_pos < W - 1; ++x_pos)
            {
                int idx = y * W + x_pos;
                double lap_r = (4.0 * r[idx] - r[idx - 1] - r[idx + 1] - r[idx - W] - r[idx + W]) / (h * h);
                numerator += r[idx] * r[idx];
                denominator += r[idx] * lap_r;
            }
        }

        return numerator / denominator;
    }

    // Compute exact solution u(x,y) = sin(p pi x / a) * sin(q pi y / a)
    static double compute_function(double x, double y)
    {
        return std::sin(p * M_PI * x / a) * std::sin(q * M_PI * y / a);
    }

    static void compute_exact_solution(double *u, double h, int width, int height)
    {
        for (int j = 0; j < height; ++j)
        {
            for (int i = 0; i < width; ++i)
            {
                double x = i * h;
                double y = j * h;
                u[j * width + i] = compute_function(x, y);
            }
        }
    }

    // Compute RHS f(x,y) = -Δu(x,y)
    static void compute_rhs(double *f, int width, int height, double h)
    {
        double factor = (M_PI * M_PI / (a * a)) * (p * p + q * q);

        for (int j = 0; j < height; ++j)
        {
            for (int i = 0; i < width; ++i)
            {
                double x = i * h;
                double y = j * h;
                f[j * width + i] = factor * std::sin(p * M_PI * x / a) * std::sin(q * M_PI * y / a);
            }
        }
    }

    static void copy_vector(double *source, double *destination, int size)
    {
        for (int i = 0; i < size; i++)
        {
            destination[i] = source[i];
        }
    }

    static bool compare_vector(double *x_1, double *x_2, int l)
    {
        for (int i = 0; i < l; i++)
        {
            double error = std::abs(x_1[i] - x_2[i]);
            if (error > epsilon)
            {
                std::cout << "i block : " << i << " x_1[i]: " << x_1[i] << " x_2[i]: " << x_2[i] << " error: " << error << std::endl;
                return false;
            }
        }
        return true;
    }
};

#endif // DYNAMIC_GRID_UTILS_HPP
