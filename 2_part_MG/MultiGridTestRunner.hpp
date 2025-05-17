#ifndef MULTIGRID_TEST_RUNNER_HPP
#define MULTIGRID_TEST_RUNNER_HPP
#include "save_to_file.hpp"
#include "../save_vector_err_file.hpp"
#include "MultiGrid.hpp"
#include "../globals.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono>

class MultigridTestRunner
{
private:
    const double epsilon;
    const int max_outer_iterations;

public:
    int alpha = 1;
    std::vector<double> v_cycle_err_norm_iteration;
    std::vector<double> w_cycle_err_norm_iteration;
    std::vector<double> f_cycle_err_norm_iteration;

    MultigridTestRunner(double eps, int max_iter, int alp)
        : epsilon(eps), max_outer_iterations(max_iter), alpha(alp) {}

    void run_all_cycles(const std::vector<int> &N_list)
    {
        for (int N : N_list)
        {
            std::cout << "\n=== Multigrid Solution for N = " << N << " ===\n";
            double *final_error;
            run_cycle("V-cycle", N, &MultigridSolver::v_cycle, false, false);
            run_cycle("W-cycle", N, &MultigridSolver::w_cycle, false, false);
            run_cycle("F-cycle", N, &MultigridSolver::f_cycle, false, false); // Wrapper handles alpha and n_final
        }
    }

    void run_all_cycles_err_h(const std::vector<int> &N_list, int iteration_performed)
    {
        std::vector<std::pair<int, double>> err_V_cycle, err_W_cycle, err_F_cycle;
        double rel_err_V, rel_err_W, rel_err_F;
        for (int N : N_list)
        {
            std::cout << "\n=== Multigrid Solution for N = " << N << " ===\n";
            double *final_error;
            rel_err_V = run_cycle("V_cycle", N, &MultigridSolver::v_cycle, false, false);
            rel_err_F = run_cycle("F-cycle", N, &MultigridSolver::f_cycle, false, false);
            rel_err_W = run_cycle("W-cycle", N, &MultigridSolver::w_cycle, false, false);
            err_V_cycle.push_back({N, rel_err_V});
            err_W_cycle.push_back({N, rel_err_W});
            err_F_cycle.push_back({N, rel_err_F});
        }

        save_error_h_to_file(iteration_performed, err_V_cycle, err_W_cycle, err_F_cycle);
    }

    void run_all_cycles_time_h(const std::vector<int> &N_list)
    {
        std::vector<std::pair<int, double>> time_V_cycle, time_W_cycle, time_F_cycle;
        for (int N : N_list)
        {
            if (N <= 2049)
            {
                std::cout << "\n=== CPU Multigrid Solution for N = " << N << " ===\n";
                double *final_error;
                auto start_v = std::chrono::high_resolution_clock::now();
                run_cycle("V_cycle", N, &MultigridSolver::v_cycle, false, false);
                auto end_v = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_v = end_v - start_v;
                time_V_cycle.push_back({N, elapsed_v.count()});

                auto start_w = std::chrono::high_resolution_clock::now();
                run_cycle("W-cycle", N, &MultigridSolver::w_cycle, false, false);
                auto end_w = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_w = end_w - start_w;
                time_W_cycle.push_back({N, elapsed_w.count()});
                /*
                auto start_f = std::chrono::high_resolution_clock::now();
                run_cycle("F-cycle", N, &MultigridSolver::f_cycle, false, false);
                auto end_f = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_f = end_f - start_f;
                time_F_cycle.push_back({N, elapsed_f.count()});
                */
            }
        }

        save_timings_to_file(time_V_cycle, time_W_cycle, time_F_cycle);
    }

    void run_v_cycles_err_vector_iteration(const std::vector<int> &N_list)
    {
        for (int N : N_list)
        {
            std::cout << "\n=== Multigrid Solution for N = " << N << " ===\n";
            run_cycle("V-cycle", N, &MultigridSolver::v_cycle, true, false);
        }
    }

    void run_w_cycles_err_vector_iteration(const std::vector<int> &N_list)
    {
        for (int N : N_list)
        {
            std::cout << "\n=== Multigrid Solution for N = " << N << " ===\n";
            run_cycle("W-cycle", N, &MultigridSolver::w_cycle, true, false);
        }
    }

    void run_all_cycles_err_norm_iteration(int iter, const std::vector<int> &N_list)
    {

        for (int N : N_list)
        {
            std::cout << "\n=== Multigrid Solution for N = " << N << " ===\n";
            run_cycle("V-cycle", N, &MultigridSolver::v_cycle, false, true);
            run_cycle("W-cycle", N, &MultigridSolver::w_cycle, false, true);
            run_cycle("F-cycle", N, &MultigridSolver::f_cycle, false, true); // Wrapper handles alpha and n_final
        }

        mg_save_error_to_file(iter, &v_cycle_err_norm_iteration, &w_cycle_err_norm_iteration, &f_cycle_err_norm_iteration);
    }

private:
    using CycleFunction = void (MultigridSolver::*)(double *, const double *, int, double);

    double run_cycle(const std::string &cycle_name, int N, CycleFunction cycle_func, bool flag_err_vector_iteration, bool flag_err_norm_iteration)
    {

        int L = N * N;
        double h = a / (N - 1);
        std::vector<std::vector<double>> err_vect_iteration;

        double *phi = new double[L];
        double *phi_tmp = new double[L];
        double *f = new double[L];
        double *phi_exact = new double[L];
        double *residual = new double[L];

        DynamicGridUtils::initialize_zeros(phi, L);
        DynamicGridUtils::compute_rhs(f, N, N, h);
        DynamicGridUtils::compute_exact_solution(phi_exact, h, N, N);

        JacobiSmoother smoother(epsilon);
        MultigridSolver mg(&smoother, alpha, N);

        int n_coarse = mg.N_coarse;
        int l_coarse = n_coarse * n_coarse;
        double h_coarse = 1.0 / (n_coarse - 1);
        double *x_coarse = new double[l_coarse];
        double *f_coarse = new double[l_coarse];

        DynamicGridUtils::compute_rhs(f_coarse, n_coarse, n_coarse, h_coarse);
        DynamicGridUtils::initialize_zeros(x_coarse, l_coarse);
        std::cout << "Running " << cycle_name << "...\n";

        if (flag_err_vector_iteration)
        {

            double res_norm = DynamicGridUtils::norm(residual, L);

            double *err_iteration = new double[L];
            DynamicGridUtils::compute_error(err_iteration, phi, phi_exact, L);
            std::vector<double> err_vector(err_iteration, err_iteration + L);
            err_vect_iteration.push_back(err_vector);
            delete[] err_iteration;
        }

        if (flag_err_norm_iteration)
        {
            double *err_iteration = new double[L];
            DynamicGridUtils::compute_error(err_iteration, phi, phi_exact, L);

            if (cycle_name == "V-cycle")
            {
                v_cycle_err_norm_iteration.push_back(DynamicGridUtils::norm(err_iteration, L) / DynamicGridUtils::norm(phi_exact, L));
            }
            else if (cycle_name == "W-cycle")
            {
                w_cycle_err_norm_iteration.push_back(DynamicGridUtils::norm(err_iteration, L) / DynamicGridUtils::norm(phi_exact, L));
            }
            else if (cycle_name == "F-cycle")
            {
                f_cycle_err_norm_iteration.push_back(DynamicGridUtils::norm(err_iteration, L) / DynamicGridUtils::norm(phi_exact, L));
            }
            delete[] err_iteration;
        }
        auto start_v = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < max_outer_iterations; ++iter)
        {
            if (cycle_name == "F-cycle")
            {

                double *phi_coarse = new double[l_coarse];
                DynamicGridUtils::initialize_zeros(phi_coarse, l_coarse);

                DynamicGridUtils::copy_vector(phi, phi_tmp, L);
                mg.compute_coarsest_grid(phi_tmp, phi_coarse, N, mg.N_coarse);

                (mg.*cycle_func)(phi_coarse, f_coarse, n_coarse, h_coarse); // Call F-cycle from coarse grid

                DynamicGridUtils::copy_vector(mg.final_solution, phi, L);
                delete[] phi_coarse;
            }
            else
            {
                (mg.*cycle_func)(phi, f, N, h);
            }
            DynamicGridUtils::compute_residual(residual, phi, f, N, N, h);
            double res_norm = DynamicGridUtils::norm(residual, L);
            // std::cout << "  Iter " << std::setw(3) << iter << " | Residual Norm: " << res_norm << "\n";

            if (flag_err_vector_iteration)
            {

                double *err_iteration = new double[L];
                DynamicGridUtils::compute_error(err_iteration, phi, phi_exact, L);
                std::vector<double> err_vector(err_iteration, err_iteration + L);
                err_vect_iteration.push_back(err_vector);
                delete[] err_iteration;
            }

            if (flag_err_norm_iteration)
            {
                double *err_iteration = new double[L];
                DynamicGridUtils::compute_error(err_iteration, phi, phi_exact, L);
                double rel_error = DynamicGridUtils::norm(err_iteration, L) / DynamicGridUtils::norm(phi_exact, L);

                if (cycle_name == "V-cycle")
                {
                    v_cycle_err_norm_iteration.push_back(rel_error);
                }
                else if (cycle_name == "W-cycle")
                {
                    w_cycle_err_norm_iteration.push_back(rel_error);
                }
                else if (cycle_name == "F-cycle")
                {
                    f_cycle_err_norm_iteration.push_back(rel_error);
                }
                delete[] err_iteration;
            }
        }
        auto end_v = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_w = end_v - start_v;

        if (flag_err_vector_iteration)
            save_errors_vector_to_file_last_iteration_cpu(err_vect_iteration);

        // Final error
        double *error = new double[L];
        DynamicGridUtils::compute_error(error, phi, phi_exact, L);
        double rel_error = DynamicGridUtils::norm(error, L) / DynamicGridUtils::norm(phi_exact, L);
        std::cout << "  Final Relative L2 Error: " << rel_error << "\n";
        std::cout << "  Elapsed Time: " << elapsed_w.count() << " seconds\n";
        delete[] phi;
        delete[] f;
        delete[] phi_exact;
        delete[] residual;
        delete[] error;

        return rel_error;
    }
};

#endif // MULTIGRID_TEST_RUNNER_HPP
