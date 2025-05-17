#include <iostream>
#include <vector>
#include <cmath>
#include "SolverRunner.hpp"

int main()
{
    const double epsilon = 1e-6;
    int smoothing_iterations = 100000;

    std::vector<int> n_list = {4, 8, 16, 32, 64, 128, 256};

    std::vector<std::pair<int, double>> timings_jacobi, timings_gs, timings_steepest, timings_cg;
    std::vector<std::pair<int, double>> error_j, error_gs, error_steepest, error_cg;

    for (int N : n_list)
    {
        std::cout << "\n=============================" << std::endl;
        std::cout << "Solving for N = " << N << " (" << N * N << " grid points)" << std::endl;
        std::cout << "=============================" << std::endl;

        SolverRunner runner(N, epsilon, smoothing_iterations);
        runner.run_all();

        // Save the residuals and error for a single N, if N is more than one value comment next line
        // runner.save_results();

        // Record timings
        timings_jacobi.emplace_back(N, runner.get_final_time("Jacobi"));
        timings_gs.emplace_back(N, runner.get_final_time("Gauss-Seidel"));
        timings_steepest.emplace_back(N, runner.get_final_time("Steepest Descent"));
        timings_cg.emplace_back(N, runner.get_final_time("Conjugate Gradient"));

        // Record errors
        error_j.emplace_back(N, runner.get_final_error("Jacobi"));
        error_gs.emplace_back(N, runner.get_final_error("Gauss-Seidel"));
        error_steepest.emplace_back(N, runner.get_final_error("Steepest Descent"));
        error_cg.emplace_back(N, runner.get_final_error("Conjugate Gradient"));
    }

    save_timings_to_file(timings_jacobi, timings_gs, timings_steepest, timings_cg);
    save_error_h_to_file(error_j, error_gs, error_steepest, error_cg);

    // JACOBI ERR VECTOR ITERATION
    /*
    n_list = {64};
    smoothing_iterations = 50;
    for (int N : n_list)
    {
        std::cout << "\n=============================" << std::endl;
        std::cout << "Solving for N = " << N << " (" << N * N << " grid points)" << std::endl;
        std::cout << "=============================" << std::endl;

        SolverRunner runner(N, epsilon, smoothing_iterations);
        runner.run_jacobi_err_vector_iteration();
    }


    */

    return 0;
}
