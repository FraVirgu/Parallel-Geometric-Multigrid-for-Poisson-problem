#include "globals.hpp"
int N = 256; // Default value for N
int L = N * N;
int W = N;
int H = N;
double h = 1.0 / (N - 1);
int MAX_ITERATION = 1000000;
double EPSILON = 1e-5;
double a = 1.0;
double p = 1.0;
double q = 1.0;
bool fix_iteration = false;
int number_fixed_iteration = 20;
int iteration_performed = 0;
int v1 = 3, v2 = 3;
