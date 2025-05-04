#ifndef MAIN_HPP
#define MAIN_HPP
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include "globals.hpp"
#include <chrono>
#include <fstream>
#include <sys/stat.h>
using namespace std;

// Jacobi solver
bool Jacobi(double *x, double *x_new, double *f, int v, int height, int weight, double h_act, int l)
{
    int index = 0;
    for (int i = 0; i <= v; i++)
    {
        // Perform Jacobi iteration
        for (int y = 1; y < height - 1; y++)
        {
            for (int x_pos = 1; x_pos < weight - 1; x_pos++)
            {
                index = y * weight + x_pos;
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

// function prototype
double compute_function(double x, double y)
{
    return sin(p * M_PI * x / a) * sin(q * M_PI * y / a);
}
void compute_exact_solution(double *u, double (*func)(double, double))
{
    double dx = h;
    double dy = h;

    for (int y = 0; y < H; y++)
    {
        for (int x = 0; x < W; x++)
        {
            double x_val = x * dx;
            double y_val = y * dy;
            u[y * W + x] = func(x_val, y_val);
        }
    }
}

void compute_rhs(double *f)
{
    double dx = h;
    double dy = h;
    double factor = (M_PI * M_PI / (a * a)) * (p * p + q * q);

    for (int y = 0; y < H; y++)
    {
        for (int x = 0; x < W; x++)
        {
            double x_val = x * dx;
            double y_val = y * dy;
            f[y * W + x] = factor * sin(p * M_PI * x_val / a) * sin(q * M_PI * y_val / a);
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
            r[index] = (f[index] - (1 / (h * h)) * (4 * x[index] - x[index - 1] - x[index + 1] - x[index - W] - x[index + W]));
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
double dynamic_vector_norm(double *f, int l)
{
    double sum = 0.0;
    sum = 0.0;
    for (int i = 0; i < l; i++)
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
            r[index] = (f[index] - (1 / (h_actual * h_actual)) * (4 * x[index] - x[index - 1] - x[index + 1] - x[index - weight] - x[index + weight]));
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
void dynamic_compute_rhs(double *f, int weight, int height, double h_act)
{
    double dx = h_act;
    double dy = h_act;
    double factor = (M_PI * M_PI / (a * a)) * (p * p + q * q);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < weight; x++)
        {
            double x_val = x * dx;
            double y_val = y * dy;

            f[y * weight + x] = factor * sin(p * M_PI * x_val / a) * sin(q * M_PI * y_val / a);
        }
    }
}

void create_directory_if_not_exists(const std::string &path)
{
    struct stat info;
    if (stat(path.c_str(), &info) != 0)
    {
        mkdir(path.c_str(), 0777);
    }
}
void save_timings_to_file(std::vector<std::pair<int, double>> &timings_cg, std::vector<std::pair<int, double>> &timings_MG, std::vector<std::pair<int, double>> &timings_FMG)
{
    create_directory_if_not_exists("OUTPUT_RESULT");

    std::ofstream file_cg("OUTPUT_RESULT/timings_cg.txt");
    if (file_cg.is_open())
    {
        for (const auto &timing : timings_cg)
        {
            file_cg << timing.first << " " << timing.second << "\n";
        }
        file_cg.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Conjugate Gradient timings.\n";
    }

    std::ofstream file_MG("OUTPUT_RESULT/timings_MG.txt");
    if (file_MG.is_open())
    {
        for (const auto &timing : timings_MG)
        {
            file_MG << timing.first << " " << timing.second << "\n";
        }
        file_MG.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Multigrid timings.\n";
    }

    std::ofstream file_FMG("OUTPUT_RESULT/timings_FMG.txt");
    if (file_FMG.is_open())
    {
        for (const auto &timing : timings_FMG)
        {
            file_FMG << timing.first << " " << timing.second << "\n";
        }
        file_FMG.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Full Multigrid timings.\n";
    }
}

#endif