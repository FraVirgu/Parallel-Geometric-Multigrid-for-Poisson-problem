#include "head.hpp"
#include "gauss_seidel.hpp"
#include "jacobian.hpp"
#include "steepest_descent.hpp"
#include "conjugate_gradient.hpp"
#include <chrono>

#include <sys/stat.h>
#include <sys/types.h>

// function prototype
double compute_function(double x, double y)
{
    return sin(p * M_PI * x / a) * sin(q * M_PI * y / a);
}

void JacobiCall(double *x, double *x_new, double *r, double *f, double *residual_reached, int *number_iteration_performed, std::vector<double> *residuals_jacobian, std::vector<double> *error_jacobian, double *x_true)
{
    initialize_zeros_vector(x);
    initialize_zeros_vector(x_new); // Ensure x_tmp is also initialized
    initialize_zeros_vector(r);
    cout << "_____  Jacobian:" << endl;
    bool result = Jacobian(x, x_new, f, r, residual_reached, number_iteration_performed, residuals_jacobian, error_jacobian, x_true);

    if (result)
    {
        std::cout << "Number of iteration performed: " << *number_iteration_performed << std::endl;
        std::cout << "Residual reached: " << *residual_reached << std::endl;
    }
    else
    {
        std::cout << "Did not converge within the maximum number of iterations." << std::endl;
        std::cout << "Residual reached: " << *residual_reached << std::endl;
    }
}

void SteepestDescentCall(double *x, double *f, double *r, int *number_iteration_performed, double *residual_reached, std::vector<double> *residuals_cg, std::vector<double> *error_cg, double *x_true)
{
    initialize_zeros_vector(x);
    initialize_zeros_vector(r);
    cout << "_____  Steepest-Descent :" << endl;
    bool result = Steepest_Descent(x, f, r, number_iteration_performed, residual_reached, residuals_cg, error_cg, x_true);

    if (result)
    {
        std::cout << "Number of iteration performed: " << *number_iteration_performed << std::endl;
        std::cout << "Residual reached: " << *residual_reached << std::endl;
    }
    else
    {
        std::cout << "Did not converge within the maximum number of iterations." << std::endl;
        std::cout << "Residual reached: " << *residual_reached << std::endl;
    }
}

void GaussSeidelCall(double *x, double *f, double *r, double *residual_reached, int *number_iteration_performed, std::vector<double> *residuals_gs, std::vector<double> *error_gs, double *x_true)
{
    initialize_zeros_vector(x);
    initialize_zeros_vector(r);
    cout << "_____  Gauss-Seidel:" << endl;
    bool result = GaussSeidel(x, f, r, residual_reached, number_iteration_performed, residuals_gs, error_gs, x_true);
    if (result)
    {
        std::cout << "Number of iteration performed: " << *number_iteration_performed << std::endl;
        std::cout << "Residual reached: " << *residual_reached << std::endl;
    }
    else
    {
        std::cout << "Did not converge within the maximum number of iterations." << std::endl;
        std::cout << "Residual reached: " << *residual_reached << std::endl;
    }
}

void ConiugateGradientCall(double *x, double *f, double *r, double *p_d, double *Ap_d, double *residual_reached, int *number_iteration_performed, std::vector<double> *residuals_cg, std::vector<double> *error_cg, double *x_true)
{
    initialize_zeros_vector(x);
    initialize_zeros_vector(r);
    initialize_zeros_vector(p_d);
    initialize_zeros_vector(Ap_d);
    cout << "_____  Conjugate Gradient:" << endl;
    bool result = conjugate_gradient(x, f, r, p_d, Ap_d, number_iteration_performed, residual_reached, residuals_cg, error_cg, x_true);

    if (result)
    {
        std::cout << "Number of iteration performed: " << *number_iteration_performed << std::endl;
        std::cout << "Residual reached: " << *residual_reached << std::endl;
    }
    else
    {
        std::cout << "Did not converge within the maximum number of iterations." << std::endl;
        std::cout << "Residual reached: " << *residual_reached << std::endl;
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

void save_residuals_to_file(std::vector<double> *residuals_jacobian, std::vector<double> *residuals_steepest, std::vector<double> *residuals_gs, std::vector<double> *residuals_cg)
{
    create_directory_if_not_exists("OUTPUT_RESULT");

    std::ofstream file_jacobian("OUTPUT_RESULT/residuals_jacobian.txt");
    if (file_jacobian.is_open())
    {
        for (const auto &residual : *residuals_jacobian)
        {
            file_jacobian << residual << "\n";
        }
        file_jacobian.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Jacobian residuals.\n";
    }

    std::ofstream file_steepest("OUTPUT_RESULT/residuals_steepest_descent.txt");
    if (file_steepest.is_open())
    {
        for (const auto &residual : *residuals_steepest)
        {
            file_steepest << residual << "\n";
        }
        file_steepest.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Steepest Descent residuals.\n";
    }

    std::ofstream file_gs("OUTPUT_RESULT/residuals_gs.txt");
    if (file_gs.is_open())
    {
        for (const auto &residual : *residuals_gs)
        {
            file_gs << residual << "\n";
        }
        file_gs.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Gauss-Seidel residuals.\n";
    }

    std::ofstream file_cg("OUTPUT_RESULT/residuals_cg.txt");
    if (file_cg.is_open())
    {
        for (const auto &residual : *residuals_cg)
        {
            file_cg << residual << "\n";
        }
        file_cg.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Conjugate Gradient residuals.\n";
    }
}

void save_error_to_file(std::vector<double> *error_jacobian, std::vector<double> *error_steepest, std::vector<double> *error_gs, std::vector<double> *error_cg)
{
    create_directory_if_not_exists("OUTPUT_RESULT");

    std::ofstream file_jacobian("OUTPUT_RESULT/error_jacobian.txt");
    if (file_jacobian.is_open())
    {
        for (const auto &error : *error_jacobian)
        {
            file_jacobian << error << "\n";
        }
        file_jacobian.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Jacobian errors.\n";
    }

    std::ofstream file_steepest("OUTPUT_RESULT/error_steepest_descent.txt");
    if (file_steepest.is_open())
    {
        for (const auto &error : *error_steepest)
        {
            file_steepest << error << "\n";
        }
        file_steepest.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Steepest Descent errors.\n";
    }

    std::ofstream file_gs("OUTPUT_RESULT/error_gs.txt");
    if (file_gs.is_open())
    {
        for (const auto &error : *error_gs)
        {
            file_gs << error << "\n";
        }
        file_gs.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Gauss-Seidel errors.\n";
    }

    std::ofstream file_cg("OUTPUT_RESULT/error_cg.txt");
    if (file_cg.is_open())
    {
        for (const auto &error : *error_cg)
        {
            file_cg << error << "\n";
        }
        file_cg.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Conjugate Gradient errors.\n";
    }
}

void save_timings_to_file(std::vector<std::pair<int, double>> &timings_jacobi, std::vector<std::pair<int, double>> &timings_gs, std::vector<std::pair<int, double>> &timings_steepest, std::vector<std::pair<int, double>> &timings_cg)
{
    create_directory_if_not_exists("OUTPUT_RESULT");

    std::ofstream file_jacobian("OUTPUT_RESULT/timings_jacobian.txt");
    if (file_jacobian.is_open())
    {
        for (const auto &timing : timings_jacobi)
        {
            file_jacobian << timing.first << " " << timing.second << "\n";
        }
        file_jacobian.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Jacobian timings.\n";
    }

    std::ofstream file_steepest("OUTPUT_RESULT/timings_steepest_descent.txt");
    if (file_steepest.is_open())
    {
        for (const auto &timing : timings_steepest)
        {
            file_steepest << timing.first << " " << timing.second << "\n";
        }
        file_steepest.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Steepest Descent timings.\n";
    }

    std::ofstream file_gs("OUTPUT_RESULT/timings_gs.txt");
    if (file_gs.is_open())
    {
        for (const auto &timing : timings_gs)
        {
            file_gs << timing.first << " " << timing.second << "\n";
        }
        file_gs.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Gauss-Seidel timings.\n";
    }

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
}

void save_error_h_to_file(std::vector<std::pair<int, double>> &error_j, std::vector<std::pair<int, double>> &error_gs, std::vector<std::pair<int, double>> &error_steepest, std::vector<std::pair<int, double>> &error_cg)
{
    create_directory_if_not_exists("OUTPUT_RESULT");

    std::ofstream file_jacobian("OUTPUT_RESULT/h_errors_jacobian.txt");
    if (file_jacobian.is_open())
    {
        for (const auto &error : error_j)
        {
            file_jacobian << error.first << " " << error.second << "\n";
        }
        file_jacobian.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Jacobian errors.\n";
    }

    std::ofstream file_steepest("OUTPUT_RESULT/h_errors_steepest_descent.txt");
    if (file_steepest.is_open())
    {
        for (const auto &error : error_steepest)
        {
            file_steepest << error.first << " " << error.second << "\n";
        }
        file_steepest.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Steepest Descent errors.\n";
    }

    std::ofstream file_gs("OUTPUT_RESULT/h_errors_gs.txt");
    if (file_gs.is_open())
    {
        for (const auto &error : error_gs)
        {
            file_gs << error.first << " " << error.second << "\n";
        }
        file_gs.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Gauss-Seidel errors.\n";
    }

    std::ofstream file_cg("OUTPUT_RESULT/h_errors_cg.txt");
    if (file_cg.is_open())
    {
        for (const auto &error : error_cg)
        {
            file_cg << error.first << " " << error.second << "\n";
        }
        file_cg.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Conjugate Gradient errors.\n";
    }
}

vector<int> n_initialization()
{
    vector<int> n;
    for (int i = 0; i < 140; i = i + 20)
    {
        n.push_back(i);
    }
    return n;
}

void parameter_initialization(int n, int max_iter, double epsilon, double a_val, double p_val, double q_val)
{
    N = n;
    L = n * n;
    W = n;
    H = n;
    h = 1.0 / (n - 1);
    MAX_ITERATION = max_iter;
    EPSILON = epsilon;
    a = a_val;
    p = p_val;
    q = q_val;
}

// Run the simulation once with the starting parameters and save the results to file for each method
// OUTPUT FILE (residual) : residuals_jacobian.txt, residuals_steepest_descent.txt, residuals_gs.txt
// OUTPUT FILE (error) : error_jacobian.txt, error_steepest_descent.txt, error_gs.txt
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
    compute_laplacian(x_true, compute_function);

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

void timeSingleRun(std::vector<std::pair<int, double>> &timings_jacobi, std::vector<std::pair<int, double>> &timings_gs, std::vector<std::pair<int, double>> &timings_steepest, std::vector<std::pair<int, double>> &timings_cg)
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
    compute_laplacian(x_true, compute_function);
    // Jacobi
    auto start_jacobi = std::chrono::high_resolution_clock::now();
    JacobiCall(x, x_tmp, res, f, residual_reached, number_iteration_performed, residuals_jacobian, error_jacobian, x_true);
    auto end_jacobi = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_jacobi = end_jacobi - start_jacobi;
    timings_jacobi.push_back(std::make_pair(N, elapsed_jacobi.count()));

    // Steepest Descent
    auto start_steepest = std::chrono::high_resolution_clock::now();
    SteepestDescentCall(x, f, res, number_iteration_performed, residual_reached, residuals_steepest, error_steepest, x_true);
    auto end_steepest = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_steepest = end_steepest - start_steepest;
    timings_steepest.push_back(std::make_pair(N, elapsed_steepest.count()));

    // Gauss Seidel
    auto start_gs = std::chrono::high_resolution_clock::now();
    GaussSeidelCall(x, f, res, residual_reached, number_iteration_performed, residuals_gs, error_gs, x_true);
    auto end_gs = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_gs = end_gs - start_gs;
    timings_gs.push_back(std::make_pair(N, elapsed_gs.count()));

    // Conjugate Gradient
    auto start_cg = std::chrono::high_resolution_clock::now();
    ConiugateGradientCall(x, f, res, p_d, Ap_d, residual_reached, number_iteration_performed, residuals_cg, error_cg, x_true);
    auto end_cg = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cg = end_cg - start_cg;
    timings_cg.push_back(std::make_pair(N, elapsed_cg.count()));

    free(x);
    free(x_tmp);
    free(f);
    free(number_iteration_performed);
    free(residual_reached);

    delete residuals_jacobian;
    delete residuals_steepest;
    delete residuals_gs;
}

void timeSingleRun_H(std::vector<std::pair<int, double>> &error_h_j)
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
    compute_laplacian(x_true, compute_function);
    // Jacobi
    JacobiCall(x, x_tmp, res, f, residual_reached, number_iteration_performed, residuals_jacobian, error_jacobian, x_true);
    error_h_j.push_back(std::make_pair(N, error_jacobian->back()));

    free(x);
    free(x_tmp);
    free(f);
    free(number_iteration_performed);
    free(residual_reached);

    delete residuals_jacobian;
    delete residuals_steepest;
    delete residuals_gs;
}

// Run the simulation for multiple N values and save the time execution for each method
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
        parameter_initialization(n[i], 100000, 1e-4, 1.0, 1.0, 1.0);
        cout << "\t\t\t\t\t\t\t\t\t   N: " << N << endl;
        timeSingleRun(timings_jacobi, timings_gs, timings_steepest, timings_cg);
    }

    save_timings_to_file(timings_jacobi, timings_gs, timings_steepest, timings_cg);
}

// Run the simulation for multiple N values and save the time execution for each method
void multipleRun_h()
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

    fix_iteration = true;

    for (int i = 0; i < n.size(); i++)
    {
        parameter_initialization(n[i], 100000, 1e-4, 1.0, 1.0, 1.0);
        cout << "\t\t\t\t\t\t\t\t\t   N: " << N << endl;
        timeSingleRun_H(error_j);
    }

    save_error_h_to_file(error_j, error_gs, error_steepest, error_cg);
}
