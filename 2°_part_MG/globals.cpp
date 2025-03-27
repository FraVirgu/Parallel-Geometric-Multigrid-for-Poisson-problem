#include "globals.hpp"
int N = 128;
int L = N * N;
int W = N;
int H = N;

int MAX_ITERATION = 1000000;
double EPSILON = 1e-7;
double a = 1.0;
double h = a / (N - 1);
double p = 2.0;
double q = 3.0;
bool fix_iteration = false;
int number_fixed_iteration = 20;
int v1 = 2, v2 = 2;
