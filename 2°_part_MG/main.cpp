#include "operator.hpp"

void JacobiCall()
{
    cout << "JACOBI METHOD" << endl;
    double *x = new double[L];
    double *output = new double[L];
    double *smoother_output = new double[L];
    double *f = new double[L];
    double *res = new double[L];
    initialize_zeros_vector(x);
    compute_rhs(f);

    auto start = chrono::high_resolution_clock::now();
    Jacobi(x, output, f, 10000, H, W, h, L);
    auto end = chrono::high_resolution_clock::now();
    compute_residual(res, output, f);
    cout << "Jacobi time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    cout << "Residual norm: " << vector_norm(res) << endl;
}

void MGCall()
{
    cout << "\nMULTIGRID METHOD" << endl;
    double *x = new double[L];
    double *output = new double[L];
    double *smoother_output = new double[L];
    double *f = new double[L];
    double *res = new double[L];
    int v1 = 50, v2 = 100, level = 0;
    initialize_zeros_vector(x);
    initialize_zeros_vector(output);
    initialize_zeros_vector(smoother_output);
    initialize_zeros_vector(res);
    compute_rhs(f);

    auto start_MG = chrono::high_resolution_clock::now();
    MG(output, x, smoother_output, f, res, v1, v2, level, N, L, W, H, h);
    auto end_MG = chrono::high_resolution_clock::now();
    cout << "Multigrid time: " << chrono::duration_cast<chrono::milliseconds>(end_MG - start_MG).count() << "ms" << endl;
    initialize_zeros_vector(res);
    compute_residual(res, output, f);
    cout << "Residual norm: " << vector_norm(res) << endl;
}

void FMgCall()
{
    cout << "\nFULL MULTIGRID METHOD" << endl;
    double **x = new double *[static_cast<int>(log2(N))];
    double **output = new double *[static_cast<int>(log2(N))];
    double **smoother_output = new double *[static_cast<int>(log2(N))];
    double **f = new double *[static_cast<int>(log2(N))];
    double **res = new double *[static_cast<int>(log2(N))];
    int *n = new int[static_cast<int>(log2(N))];
    int *l = new int[static_cast<int>(log2(N))];
    int *weight = new int[static_cast<int>(log2(N))];
    int *height = new int[static_cast<int>(log2(N))];
    double *h_act = new double[static_cast<int>(log2(N))];
    int initial_N = N;

    auto start_FMG = chrono::high_resolution_clock::now();
    initialize_FG(initial_N, x, output, smoother_output, f, res, n, l, weight, height, h_act);
    FMG(initial_N, output, x, smoother_output, f, res, n, l, weight, height, h_act);
    auto end_FMG = chrono::high_resolution_clock::now();
    cout << "FMG time: " << chrono::duration_cast<chrono::milliseconds>(end_FMG - start_FMG).count() << "ms" << endl;
    compute_residual(res[static_cast<int>(log2(N)) - 1], output[static_cast<int>(log2(N)) - 1], f[static_cast<int>(log2(N)) - 1]);
    cout << "Residual norm: " << vector_norm(res[static_cast<int>(log2(N)) - 1]) << endl;
}

void ConiugateGradientCall()
{
    double *x = new double[L];
    double *x_tmp = new double[L];
    double *x_true = new double[L];
    double *f = new double[L];
    double *res = new double[L];
    double *p_d = new double[L];
    double *Ap_d = new double[L];
    int *number_iteration_performed = new int;
    double *residual_reached = new double;
    initialize_zeros_vector(x);
    compute_rhs(f);
    initialize_zeros_vector(x);
    initialize_zeros_vector(res);
    initialize_zeros_vector(p_d);
    initialize_zeros_vector(Ap_d);
    cout << "\nCG:" << endl;
    auto start_CG = chrono::high_resolution_clock::now();
    bool result = conjugate_gradient(x, f, res, p_d, Ap_d, number_iteration_performed, residual_reached);
    auto end_CG = chrono::high_resolution_clock::now();
    cout << "CG time: " << chrono::duration_cast<chrono::milliseconds>(end_CG - start_CG).count() << "ms" << endl;
    std::cout << "Residual norm: " << *residual_reached << std::endl;
}

int main()
{
    // JacobiCall();
    ConiugateGradientCall();
    MGCall();
    FMgCall();
    return 0;
}