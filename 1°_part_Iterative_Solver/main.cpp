#include "main.hpp"

/**
 * @brief Executes a single run of iterative solvers for a linear system.
 * Initializes data, computes RHS and exact solution, runs solvers (Jacobi, Steepest Descent,
 * Gauss-Seidel, Conjugate Gradient), and saves residuals/errors to files.
 */
void singleRun()
{
    std::vector<double> *residuals_jacobian = new std::vector<double>();
    std::vector<double> *residuals_steepest = new std::vector<double>();
    std::vector<double> *residuals_gs = new std::vector<double>();
    std::vector<double> *residuals_cg = new std::vector<double>();

    std::vector<double> *error_jacobian = new std::vector<double>();
    std::vector<double> *error_steepest = new std::vector<double>();
    std::vector<double> *error_gs = new std::vector<double>();
    std::vector<double> *error_cg = new std::vector<double>();

    std::vector<double> *final_error_norm = new std::vector<double>();

    double *x = new double[L];
    double *x_tmp = new double[L];
    double *x_true = new double[L];
    double *f = new double[L];
    double *res = new double[L];
    double *p_d = new double[L];
    double *Ap_d = new double[L];
    int *number_iteration_performed = new int;
    double *residual_reached = new double;

    compute_rhs(f);
    compute_exact_solution(x_true, compute_function);

    JacobiCall(x, x_tmp, res, f, residual_reached, number_iteration_performed, residuals_jacobian, error_jacobian, x_true);
    SteepestDescentCall(x, f, res, number_iteration_performed, residual_reached, residuals_steepest, error_steepest, x_true);
    GaussSeidelCall(x, f, res, residual_reached, number_iteration_performed, residuals_gs, error_gs, x_true);
    ConiugateGradientCall(x, f, res, p_d, Ap_d, residual_reached, number_iteration_performed, residuals_cg, error_cg, x_true);

    free(x);
    free(x_tmp);
    free(f);
    free(number_iteration_performed);
    free(residual_reached);
    save_residuals_to_file(residuals_jacobian, residuals_steepest, residuals_gs, residuals_cg);
    save_error_to_file(error_jacobian, error_steepest, error_gs, error_cg);

    delete residuals_jacobian;
    delete residuals_steepest;
    delete residuals_gs;
}

/**
 * @brief Same as singleRun : runs iterative solvers and records their residuals and errors.
 */
void timeSingleRun(std::vector<std::pair<int, double>> &timings_jacobi, std::vector<std::pair<int, double>> &timings_gs, std::vector<std::pair<int, double>> &timings_steepest, std::vector<std::pair<int, double>> &timings_cg, std::vector<std::pair<int, double>> &error_grid_jacobian, std::vector<std::pair<int, double>> &error_grid_steepest, std::vector<std::pair<int, double>> &error_grid_gs, std::vector<std::pair<int, double>> &error_grid_cg)
{
    std::vector<double> *residuals_jacobian = new std::vector<double>();
    std::vector<double> *residuals_steepest = new std::vector<double>();
    std::vector<double> *residuals_gs = new std::vector<double>();
    std::vector<double> *residuals_cg = new std::vector<double>();

    std::vector<double> *error_jacobian = new std::vector<double>();
    std::vector<double> *error_steepest = new std::vector<double>();
    std::vector<double> *error_gs = new std::vector<double>();
    std::vector<double> *error_cg = new std::vector<double>();

    double *x = new double[L];
    double *x_tmp = new double[L];
    double *x_true = new double[L];
    double *f = new double[L];
    double *res = new double[L];
    double *p_d = new double[L];
    double *Ap_d = new double[L];
    int *number_iteration_performed = new int;
    double *residual_reached = new double;

    compute_rhs(f);
    compute_exact_solution(x_true, compute_function);
    // Jacobi
    auto start_jacobi = std::chrono::high_resolution_clock::now();
    JacobiCall(x, x_tmp, res, f, residual_reached, number_iteration_performed, residuals_jacobian, error_jacobian, x_true);
    auto end_jacobi = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_jacobi = end_jacobi - start_jacobi;
    timings_jacobi.push_back(std::make_pair(N, elapsed_jacobi.count()));
    error_grid_jacobian.push_back(std::make_pair(N, error_jacobian->back()));

    // Steepest Descent
    auto start_steepest = std::chrono::high_resolution_clock::now();
    SteepestDescentCall(x, f, res, number_iteration_performed, residual_reached, residuals_steepest, error_steepest, x_true);
    auto end_steepest = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_steepest = end_steepest - start_steepest;
    timings_steepest.push_back(std::make_pair(N, elapsed_steepest.count()));
    error_grid_steepest.push_back(std::make_pair(N, error_steepest->back()));

    // Gauss Seidel
    auto start_gs = std::chrono::high_resolution_clock::now();
    GaussSeidelCall(x, f, res, residual_reached, number_iteration_performed, residuals_gs, error_gs, x_true);
    auto end_gs = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_gs = end_gs - start_gs;
    timings_gs.push_back(std::make_pair(N, elapsed_gs.count()));
    error_grid_gs.push_back(std::make_pair(N, error_gs->back()));

    // Conjugate Gradient
    auto start_cg = std::chrono::high_resolution_clock::now();
    ConiugateGradientCall(x, f, res, p_d, Ap_d, residual_reached, number_iteration_performed, residuals_cg, error_cg, x_true);
    auto end_cg = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cg = end_cg - start_cg;
    timings_cg.push_back(std::make_pair(N, elapsed_cg.count()));
    error_grid_cg.push_back(std::make_pair(N, error_cg->back()));

    free(x);
    free(x_tmp);
    free(f);
    free(number_iteration_performed);
    free(residual_reached);

    delete residuals_jacobian;
    delete residuals_steepest;
    delete residuals_gs;

    delete error_jacobian;
}

/**
 * @brief Executes multiple runs of iterative solvers for varying problem sizes.
 *
 * This function initializes parameters for different problem sizes, runs iterative
 * solvers (Jacobi, Gauss-Seidel, Steepest Descent, and Conjugate Gradient), and
 * collects timing and error data for each solver. The results are then saved to files.
 */
void multipleRun()
{
    vector<int> n = n_initialization();
    std::vector<std::pair<int, double>> timings_jacobi;
    std::vector<std::pair<int, double>> timings_gs;
    std::vector<std::pair<int, double>> timings_steepest;
    std::vector<std::pair<int, double>> timings_cg;

    std::vector<std::pair<int, double>> error_j;
    std::vector<std::pair<int, double>> error_gs;
    std::vector<std::pair<int, double>> error_steepest;
    std::vector<std::pair<int, double>> error_cg;

    for (int i = 0; i < n.size(); i++)
    {
        parameter_initialization(n[i], 1000000, 1e-4, 1.0, 1.0, 1.0);
        cout << "\t\t\t\t\t\t\t\t\t   N: " << N << endl;
        timeSingleRun(timings_jacobi, timings_gs, timings_steepest, timings_cg, error_j, error_gs, error_steepest, error_cg);
    }

    save_timings_to_file(timings_jacobi, timings_gs, timings_steepest, timings_cg);
    save_error_h_to_file(error_j, error_gs, error_steepest, error_cg);
}

int main()
{
    singleRun();
    // multipleRun();
    return 0;
}