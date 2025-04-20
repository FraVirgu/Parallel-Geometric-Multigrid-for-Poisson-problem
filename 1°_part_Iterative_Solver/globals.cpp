#include "globals.hpp"
int N = 64;
int L = N * N;
int W = N;
int H = N;
double h = 1.0 / (N - 1);
int MAX_ITERATION = 1000000;
double EPSILON = 1e-8;
double a = 1.0;
double p = 1.0;
double q = 1.0;
bool fix_iteration = false;
int number_fixed_iteration = 1000000;
bool SAVE_ERROR_VERCTOR = false;
// Update grid parameters when N changes
void update_grid_parameters(int new_N)
{
    N = new_N;
    W = N;
    H = N;
    L = W * H;
    h = 1.0 / (N - 1);
}