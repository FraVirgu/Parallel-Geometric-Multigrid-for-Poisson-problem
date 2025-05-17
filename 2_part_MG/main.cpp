#include "MultiGridTestRunner.hpp"
#include "../1_part_Smoother_Comparison/SolverRunner.hpp"

int main()
{
    // Define grid sizes (must be odd for multigrid coarsening to work properly)
    std::vector<int> N_list = {33, 65, 129, 257, 513, 1025};

    // std::vector<int> N_list = {129};

    // Set convergence tolerance and max iterations
    const double epsilon = 1e-7;
    const int mg_max_iterations = 1;
    int smoothing_iterations = 50000;
    int alpha = 3;
    // Instantiate and run test runner
    MultigridTestRunner runner(epsilon, mg_max_iterations, alpha);

    // NORMAL RUN
    runner.run_all_cycles(N_list);

    // PLOT ERR_H NORM FOR EACH N and FOR EACH mg_iter_list
    /*
    std::vector<int> mg_iter_list = {1, 2, 3};

    for (int mg_iter : mg_iter_list)
    {
        // Instantiate and run test runner
        MultigridTestRunner runner(epsilon, mg_iter);
        runner.run_all_cycles_err_h(N_list, mg_iter);
    }

    std::vector<std::pair<int, double>> timings_jacobi, timings_gs, timings_steepest, timings_cg;
    std::vector<std::pair<int, double>> error_j, error_gs, error_steepest, error_cg;
    for (int N : N_list)
    {
        SolverRunner runner(N, epsilon, smoothing_iterations);
        runner.run_cg();

        // Record timings
        // timings_jacobi.emplace_back(N, runner.get_final_time("Jacobi"));
        // timings_gs.emplace_back(N, runner.get_final_time("Gauss-Seidel"));
        // timings_steepest.emplace_back(N, runner.get_final_time("Steepest Descent"));
        timings_cg.emplace_back(N, runner.get_final_time("Conjugate Gradient"));

        // Record errors
        // error_j.emplace_back(N, runner.get_final_error("Jacobi"));
        // error_gs.emplace_back(N, runner.get_final_error("Gauss-Seidel"));
        // error_steepest.emplace_back(N, runner.get_final_error("Steepest Descent"));
        error_cg.emplace_back(N, runner.get_final_error("Conjugate Gradient"));
    }
    save_timings_to_file(timings_jacobi, timings_gs, timings_steepest, timings_cg);
    save_error_h_to_file(error_j, error_gs, error_steepest, error_cg);


    */

    // PLOT ERR VECTOR ITERATION

    // std::vector<int> N_list = {65};
    // runner.run_v_cycles_err_vector_iteration(N_list);

    // PLOT ERR NORM ITERATION
    /*
      std::vector<int> mg_iter_list = {1, 2, 3};
    std::vector<int> N_list = {513};
    for (int mg_iter : mg_iter_list)
    {
        // Instantiate and run test runner
        MultigridTestRunner runner(epsilon, mg_iter);
        runner.run_all_cycles_err_norm_iteration(mg_iter, N_list);
    }

    */

    // PLOT TIMES
    // runner.run_all_cycles_time_h(N_list);

    return 0;
}
