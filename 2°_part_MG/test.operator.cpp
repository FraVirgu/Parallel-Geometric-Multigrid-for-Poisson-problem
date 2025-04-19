#include <iostream>
#include <vector>
#include <cmath>

#include <chrono>
#include <fstream>
#include <sys/stat.h>

void dynamic_initialize_zeros_vector(double *x, int l)
{
    for (int i = 0; i < l; i++)
    {
        x[i] = 0.0;
    }
}

double dynamic_compute_vector_norm(double *r, int l)
{
    double sum = 0.0;
    for (int i = 0; i < l; i++)
    {
        sum += r[i] * r[i]; // Sum of squares
    }
    return sqrt(sum); // Square root of sum
}
void prolungator(double *input, double *output, int input_H, int input_W, int output_H, int output_W)
{
    double weight[9] = {0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25};

    int index_output_x;
    int index_output_y;

    for (int i = 0; i < input_H; i++)
    {
        for (int j = 0; j < input_W; j++)
        {
            index_output_x = (2 * j) - 1;
            index_output_y = (2 * i) - 1;

            for (int k = 0; k < 3; k++)
            {
                for (int l = 0; l < 3; l++)
                {
                    int x = index_output_x + l;
                    int y = index_output_y + k;

                    if (x >= 0 && x < output_W && y >= 0 && y < output_H)
                    {
                        if (x == 0 || x == output_W - 1 || y == 0 || y == output_H - 1)
                        {
                            output[y * output_W + x] = 0.0; // Enforce boundary condition
                        }
                        else
                        {
                            output[y * output_W + x] += weight[k * 3 + l] * input[i * input_W + j];
                        }
                    }
                }
            }
        }
    }
}

void restriction(double *input, double *output, int input_H, int input_W, int output_H, int output_W)
{
    double weight[9] = {0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25};
    for (int i = 0; i < output_H; i++)
    {
        for (int j = 0; j < output_W; j++)
        {
            if (i == 0 || i == output_H - 1 || j == 0 || j == output_W - 1) // Enforce boundary condition
            {
                output[i * output_W + j] = 0.0;
            }
            else
            {
                double sum = 0.0;
                double weight_sum = 0.0;
                int index_input_x = 2 * j;
                int index_input_y = 2 * i;

                for (int k = -1; k <= 1; k++)
                {
                    for (int l = -1; l <= 1; l++)
                    {
                        int x = index_input_x + l;
                        int y = index_input_y + k;

                        if (x >= 0 && x < input_W && y >= 0 && y < input_H)
                        {
                            sum += weight[(k + 1) * 3 + (l + 1)] * input[y * input_W + x];
                            weight_sum += weight[(k + 1) * 3 + (l + 1)];
                        }
                    }
                }

                output[i * output_W + j] = sum / weight_sum;
            }
        }
    }
}

int main()
{
    int coarse_N = 4;
    int fine_N = 8;
    int coarse_size = coarse_N * coarse_N;
    int fine_size = fine_N * fine_N;

    double *coarse = new double[coarse_size]{
        0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 1.0, 0.0,
        0.0, 1.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0};

    double *fine = new double[fine_size];
    double *restricted = new double[coarse_size];
    dynamic_initialize_zeros_vector(fine, fine_size);
    dynamic_initialize_zeros_vector(restricted, coarse_size);

    // Prolongation: 4x4 → 8x8
    prolungator(coarse, fine, coarse_N, coarse_N, fine_N, fine_N);

    // Restriction: 8x8 → 4x4
    restriction(fine, restricted, fine_N, fine_N, coarse_N, coarse_N);

    // Norms
    double norm_coarse = dynamic_compute_vector_norm(coarse, coarse_size);
    double norm_fine = dynamic_compute_vector_norm(fine, fine_size);
    double norm_restricted = dynamic_compute_vector_norm(restricted, coarse_size);

    std::cout << "||u_H||         = " << norm_coarse << std::endl;
    std::cout << "||P u_H||       = " << norm_fine << std::endl;
    std::cout << "||R P u_H||     = " << norm_restricted << std::endl;

    return 0;
}