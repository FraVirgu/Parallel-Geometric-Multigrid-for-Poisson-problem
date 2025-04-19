#include "operator.hpp"
bool save_solution_MG = false;
void save_solution_to_file(double *x, int height, int length)
{
    std::ofstream file("solution.txt");
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file solution.txt for writing." << std::endl;
        return;
    }
    file << height << std::endl;
    for (int i = 0; i < length; ++i)
    {
        file << x[i] << std::endl;
    }
    file.close();
    std::cout << "Solution saved to solution.txt" << std::endl;
}

void JacobiCall()
{
    cout << "JACOBI METHOD" << endl;
    double *x = new double[L];
    double *output = new double[L];
    double *smoother_output = new double[L];
    double *f = new double[L];
    double *res = new double[L];
    dynamic_initialize_zeros_vector(x, L);
    dynamic_compute_rhs(f, W, H, h);

    auto start = chrono::high_resolution_clock::now();
    Jacobi(x, output, f, 10000, H, W, h, L);
    auto end = chrono::high_resolution_clock::now();
    dynamic_compute_residual(res, output, f, W, H, h);
    cout << "Jacobi time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

    cout << "Residual norm: " << dynamic_compute_vector_norm(res, L) << endl;
}

auto MGCall()
{
    cout << "\nMULTIGRID METHOD" << endl;
    double *x = new double[L];
    double *output = new double[L];
    double *smoother_output = new double[L];
    double *f = new double[L];
    double *res = new double[L];
    dynamic_initialize_zeros_vector(x, L);
    dynamic_initialize_zeros_vector(output, L);
    dynamic_initialize_zeros_vector(smoother_output, L);
    dynamic_initialize_zeros_vector(res, L);
    dynamic_compute_rhs(f, W, H, h);
    int level = 0;
    auto start_MG = chrono::high_resolution_clock::now();
    MG(output, x, smoother_output, f, res, v1, v2, level, N, L, W, H, h);
    auto end_MG = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_MG - start_MG).count();
    cout << "Multigrid time: " << chrono::duration_cast<chrono::milliseconds>(end_MG - start_MG).count() << "ms" << endl;
    dynamic_initialize_zeros_vector(res, L);
    dynamic_compute_residual(res, output, f, W, H, h);
    cout << "Residual norm: " << dynamic_compute_vector_norm(res, L) << endl;

    return duration;
}

auto FMgCall()
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

    initialize_FG(initial_N, x, output, smoother_output, f, res, n, l, weight, height, h_act);
    auto start_FMG = chrono::high_resolution_clock::now();
    FMG(initial_N, output, x, smoother_output, f, res, n, l, weight, height, h_act, v1, v2);
    auto end_FMG = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_FMG - start_FMG).count();
    if (save_solution_MG)
        save_solution_to_file(output[static_cast<int>(log2(N)) - 1], height[static_cast<int>(log2(N)) - 1], l[static_cast<int>(log2(N)) - 1]);
    cout
        << "FMG time: " << chrono::duration_cast<chrono::milliseconds>(end_FMG - start_FMG).count() << "ms" << endl;
    dynamic_compute_residual(res[static_cast<int>(log2(N)) - 1], output[static_cast<int>(log2(N)) - 1], f[static_cast<int>(log2(N)) - 1], W, H, h);
    cout << "Residual norm: " << dynamic_compute_vector_norm(res[static_cast<int>(log2(N)) - 1], l[static_cast<int>(log2(N)) - 1]) << endl;
    return duration;
}

auto ConiugateGradientCall()
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
    dynamic_initialize_zeros_vector(x, L);
    dynamic_compute_rhs(f, W, H, h);
    dynamic_initialize_zeros_vector(res, L);
    dynamic_initialize_zeros_vector(p_d, L);
    dynamic_initialize_zeros_vector(Ap_d, L);

    cout << "\nCG:" << endl;
    auto start_CG = chrono::high_resolution_clock::now();
    bool result = conjugate_gradient(x, f, res, p_d, Ap_d, number_iteration_performed, residual_reached);
    auto end_CG = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_CG - start_CG).count();
    cout << "CG time: " << chrono::duration_cast<chrono::milliseconds>(end_CG - start_CG).count() << "ms" << endl;
    std::cout << "Residual norm: " << *residual_reached << std::endl;
    return duration;
}

vector<int> n_initialization()
{
    vector<int> n;
    for (int i = 4; i <= N; i = i * 2)
    {
        n.push_back(i);
    }
    return n;
}

int main()
{
    // vector<int> n = n_initialization();

    vector<int> n;
    n.push_back(N);

    save_solution_MG = false;

    std::vector<std::pair<int, double>> timings_CG;
    std::vector<std::pair<int, double>> timings_MG;
    std::vector<std::pair<int, double>> timings_FMG;

    for (int i = 0; i < n.size(); i++)
    {
        update_global_parameter(n[i]);
        cout << "\n----------------\tN: " << N << endl;
        auto duration_CG = ConiugateGradientCall();
        auto duration_MG = MGCall();
        auto duration_FMG = FMgCall();

        timings_CG.push_back(std::make_pair(N, duration_CG));
        timings_MG.push_back(std::make_pair(N, duration_MG));
        timings_FMG.push_back(std::make_pair(N, duration_FMG));
    }

    save_timings_to_file(timings_CG, timings_MG, timings_FMG);

    return 0;
}