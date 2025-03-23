#include "globals.hpp"
int N = 100;
int L = N * N;
int W = N;
int H = N;
double h = 1.0 / (N - 1);
int MAX_ITERATION = 1000000;
double EPSILON = 1e-4;
double a = 1.0;
double p = 1.0;
double q = 1.0;
bool fix_iteration = false;
int number_fixed_iteration = 10;
