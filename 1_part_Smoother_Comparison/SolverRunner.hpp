#ifndef SOLVER_RUNNER_HPP
#define SOLVER_RUNNER_HPP

#include "../DynamicGridUtils.hpp"
#include "../Smoother.hpp"
#include "../globals.hpp"
#include "save_to_file.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <map>
#include <filesystem>
#include <fstream>

class SolverRunner
{
private:
    int N, L;
    double h;
    double epsilon;
    int max_iterations;

    double *x_true;
    double *f;

    std::vector<Smoother *> smoothers;
    std::vector<std::string> names;
    std::map<std::string, int> name_to_index;

    std::vector<std::vector<double>> all_residuals;
    std::vector<std::vector<double>> all_errors;
    std::vector<double> execution_times;

public:
    SolverRunner(int N_, double eps, int max_iter)
        : N(N_), epsilon(eps), max_iterations(max_iter), L(N_ * N_), h(a / (N_ - 1)),
          x_true(new double[L]), f(new double[L]),
          // 4 is the number of iterative solver
          all_residuals(4), all_errors(4), execution_times(4, 0.0)
    {
        DynamicGridUtils::compute_rhs(f, N, N, h);
        DynamicGridUtils::compute_exact_solution(x_true, h, N, N);

        std::cout << "RHS norm: " << DynamicGridUtils::norm(f, L) << std::endl;
        std::cout << "Exact solution norm: " << DynamicGridUtils::norm(x_true, L) << std::endl;

        names = {"Jacobi", "Gauss-Seidel", "Conjugate Gradient", "Steepest Descent"};
        for (size_t i = 0; i < names.size(); ++i)
            name_to_index[names[i]] = i;

        smoothers = {
            new JacobiSmoother(epsilon),
            new GaussSeidelSmoother(epsilon),
            new ConjugateGradientSmoother(epsilon),
            new SteepestDescentSmoother(epsilon)};
    }

    void run_all()
    {
        for (size_t i = 0; i < smoothers.size(); ++i)
        {
            std::cout << "\n--- Running " << names[i] << " Smoother ---" << std::endl;
            double *x = new double[L];
            DynamicGridUtils::initialize_zeros(x, L);

            auto start = std::chrono::steady_clock::now();

            smoothers[i]->smooth(x, f, N, N, h, max_iterations,
                                 x_true, &all_residuals[i], &all_errors[i]);

            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            execution_times[i] = elapsed.count();

            std::cout << "Final Residual: "
                      << (all_residuals[i].empty() ? -1.0 : all_residuals[i].back())
                      << " | Final Relative Error: "
                      << (all_errors[i].empty() ? -1.0 : all_errors[i].back())
                      << " | Iterations: " << all_residuals[i].size()
                      << " | Time: " << execution_times[i] << " seconds"
                      << std::endl;

            delete[] x;
        }
    }

    void run_cg()
    {
        // Find index of Jacobi smoother
        int cg_index = -1;
        for (size_t i = 0; i < names.size(); ++i)
        {
            if (names[i] == "Conjugate Gradient")
            {
                cg_index = static_cast<int>(i);
                break;
            }
        }

        if (cg_index == -1)
        {
            std::cerr << "Conjugate Gradient smoother not found!" << std::endl;
            return;
        }

        std::cout << "\n--- Running " << names[cg_index] << " Smoother ---" << std::endl;
        double *x = new double[L];
        DynamicGridUtils::initialize_zeros(x, L);

        auto start = std::chrono::steady_clock::now();
        smoothers[cg_index]->smooth(x, f, N, N, h, max_iterations,
                                    x_true, &all_residuals[cg_index], &all_errors[cg_index]);

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        execution_times[cg_index] = elapsed.count();

        std::cout << "Final Residual: "
                  << (all_residuals[cg_index].empty() ? -1.0 : all_residuals[cg_index].back())
                  << " | Final Relative Error: "
                  << (all_errors[cg_index].empty() ? -1.0 : all_errors[cg_index].back())
                  << " | Iterations: " << all_residuals[cg_index].size()
                  << " | Time: " << execution_times[cg_index] << " seconds"
                  << std::endl;

        delete[] x;
    }

    void run_jacobi_err_vector_iteration()
    {
        // Find index of Jacobi smoother
        int jacobi_index = -1;
        for (size_t i = 0; i < names.size(); ++i)
        {
            if (names[i] == "Jacobi")
            {
                jacobi_index = static_cast<int>(i);
                break;
            }
        }

        if (jacobi_index == -1)
        {
            std::cerr << "Jacobi smoother not found!" << std::endl;
            return;
        }

        std::cout << "\n--- Running Jacobi Smoother ---" << std::endl;

        double *x = new double[L];
        DynamicGridUtils::initialize_zeros(x, L);

        auto start = std::chrono::steady_clock::now();
        smoothers[jacobi_index]->switch_test_mode();
        smoothers[jacobi_index]->smooth(
            x, f, N, N, h, max_iterations,
            x_true, &all_residuals[jacobi_index], &all_errors[jacobi_index]);

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        execution_times[jacobi_index] = elapsed.count();

        std::cout << "Final Residual: "
                  << (all_residuals[jacobi_index].empty() ? -1.0 : all_residuals[jacobi_index].back())
                  << " | Final Relative Error: "
                  << (all_errors[jacobi_index].empty() ? -1.0 : all_errors[jacobi_index].back())
                  << " | Iterations: " << all_residuals[jacobi_index].size()
                  << " | Time: " << execution_times[jacobi_index] << " seconds"
                  << std::endl;

        delete[] x;
    }

    void save_results()
    {
        std::filesystem::create_directories("1_part/OUTPUT_RESULT");

        save_residuals_to_file(&all_residuals[0], &all_residuals[3],
                               &all_residuals[1], &all_residuals[2]);
        save_error_to_file(&all_errors[0], &all_errors[3],
                           &all_errors[1], &all_errors[2]);

        std::ofstream time_file("1_part/OUTPUT_RESULT/execution_times.txt");
        if (time_file.is_open())
        {
            for (size_t i = 0; i < execution_times.size(); ++i)
                time_file << names[i] << ": " << execution_times[i] << " s\n";
            time_file.close();
        }
        else
        {
            std::cerr << "Failed to save execution times.\n";
        }
    }

    double get_final_time(const std::string &name) const
    {
        auto it = name_to_index.find(name);
        if (it != name_to_index.end())
            return execution_times[it->second];
        return -1.0;
    }

    double get_final_error(const std::string &name) const
    {
        auto it = name_to_index.find(name);
        if (it != name_to_index.end())
        {
            const auto &err_vec = all_errors[it->second];
            if (!err_vec.empty())
                return err_vec.back();
        }
        return -1.0;
    }

    ~SolverRunner()
    {
        for (auto s : smoothers)
            delete s;
        delete[] x_true;
        delete[] f;
    }
};

#endif // SOLVER_RUNNER_HPP
