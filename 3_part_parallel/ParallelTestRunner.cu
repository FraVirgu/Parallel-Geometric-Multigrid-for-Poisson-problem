#include "Parallel_Mg.cu"
#include "../Smoother.hpp"
#include "../DynamicGridUtils.hpp"
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include "save_to_file.hpp"
// Restriction using full-weighting
void restrict_full_weighting_2(const double *fine, double *coarse, int Nf, int Nc)
{
    // Set Dirichlet boundaries on coarse grid (assume zero)
    for (int i = 0; i < Nc; ++i)
    {
        coarse[i] = 0.0;                 // Top row
        coarse[(Nc - 1) * Nc + i] = 0.0; // Bottom row
        coarse[i * Nc] = 0.0;            // Left column
        coarse[i * Nc + (Nc - 1)] = 0.0; // Right column
    }

    // Apply full-weighting to interior coarse grid points
    for (int jc = 1; jc < Nc - 1; ++jc)
    {
        for (int ic = 1; ic < Nc - 1; ++ic)
        {
            int i_f = 2 * ic;
            int j_f = 2 * jc;

            int idx_c = jc * Nc + ic;
            int idx_f = j_f * Nf + i_f;

            coarse[idx_c] = 0.25 * fine[idx_f] +
                            0.125 * (fine[idx_f + 1] + fine[idx_f - 1] +
                                     fine[idx_f + Nf] + fine[idx_f - Nf]) +
                            0.0625 * (fine[idx_f - Nf - 1] + fine[idx_f - Nf + 1] +
                                      fine[idx_f + Nf - 1] + fine[idx_f + Nf + 1]);
        }
    }
}

// Prolongation using bilinear interpolation
void prolongation_2(double *fine, const double *coarse, int Nf, int Nc)
{

    // Bilinear interpolation on interior points
    for (int jc = 1; jc < Nc - 1; ++jc)
    {
        for (int ic = 1; ic < Nc - 1; ++ic)
        {
            int i_f = 2 * ic;
            int j_f = 2 * jc;

            int idx_c = jc * Nc + ic;

            fine[j_f * Nf + i_f] += coarse[idx_c];
            fine[(j_f + 1) * Nf + i_f] += 0.5 * (coarse[idx_c] + coarse[idx_c + Nc]);
            fine[j_f * Nf + i_f + 1] += 0.5 * (coarse[idx_c] + coarse[idx_c + 1]);
            fine[(j_f + 1) * Nf + i_f + 1] += 0.25 * (coarse[idx_c] + coarse[idx_c + 1] +
                                                      coarse[idx_c + Nc] + coarse[idx_c + Nc + 1]);
        }
    }

    // Set Dirichlet boundary (zero) on the fine grid
    for (int j = 0; j < Nf; ++j)
    {
        for (int i = 0; i < Nf; ++i)
        {
            if (i == 0 || i == Nf - 1 || j == 0 || j == Nf - 1)
                fine[j * Nf + i] = 0.0;
        }
    }
}

using namespace std;
class ParallelTestRunner
{
public:
    int N;
    double epsilon = 1e-6;
    int alpha;
    int mg_max_iterations;
    std::vector<double> err_vec;

    std::vector<std::tuple<int, int, double>> time_residual_cpu;
    std::vector<std::tuple<int, int, double>> time_residual_gpu;

    std::vector<std::tuple<int, int, double>> time_jacobi_cpu;
    std::vector<std::tuple<int, int, double>> time_jacobi_gpu;

    std::vector<std::tuple<int, int, double>> time_restriction_cpu;
    std::vector<std::tuple<int, int, double>> time_restriction_gpu;

    std::vector<std::tuple<int, int, double>> time_prolungator_cpu;
    std::vector<std::tuple<int, int, double>> time_prolungator_gpu;

    ParallelTestRunner(int n, int mg_iterations, int alp) : N(n), alpha(alp), mg_max_iterations(mg_iterations) {};

    void plotTimeSequentialVsParallel(std::vector<int> N_list, std::vector<int> N_thread_list)
    {
        for (int n_thread : N_thread_list)
        {
            num_thread = n_thread;
            for (int n : N_list)
            {
                cout << "\t\tN: " << n << endl;
                this->N = n;
                this->run_all_methods();
            }
        }

        save_timings_to_file_all_methods(
            this->time_residual_cpu,
            this->time_residual_gpu,

            this->time_jacobi_cpu,
            this->time_jacobi_gpu,

            this->time_restriction_cpu,
            this->time_restriction_gpu,

            this->time_prolungator_cpu,
            this->time_prolungator_gpu

        );
    }

    void run_all_cycles(const std::vector<int> &N_list)
    {
        std::vector<std::pair<int, double>> time_parallel_V_cycle, time_parallel_W_cycle;

        for (int n : N_list)
        {
            this->N = n;
            std::cout << "\n=== GPU Multigrid Solution for N = " << N << " ===\n";
            auto time_v_cycle = run_v_cycle();
            auto time_w_cycle = run_w_cycle(false);
            time_parallel_V_cycle.push_back({N, time_v_cycle});
            time_parallel_W_cycle.push_back({N, time_w_cycle});
        }
        save_timings_to_file(time_parallel_V_cycle, time_parallel_W_cycle);
    }

    void run_w_cycles_err_vector_iteration(const std::vector<int> &N_list)
    {
        for (int n : N_list)
        {
            this->N = n;
            auto time_w_cycle = run_w_cycle(true);
        }
    }

    double run_v_cycle()
    {

        int L = N * N;
        double h = 1.0 / (N - 1);
        // Unified (device-accessible) memory
        double *phi;
        double *f;
        double *x_true = new double[L];
        size_t bytes = L * sizeof(double);
        cudaMallocManaged(&phi, bytes);
        cudaMallocManaged(&f, bytes);
        // cudaDeviceSynchronize(); // Ensure memory is ready

        DynamicGridUtils::initialize_zeros(phi, L);
        DynamicGridUtils::compute_rhs(f, N, N, h); // Host version
        DynamicGridUtils::compute_exact_solution(x_true, h, N, N);
        ParallelMultiGridSolver parallel_mg_solver(alpha);

        auto start_v = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < mg_max_iterations; iter++)
            parallel_mg_solver.v_cycle(phi, f, N, h);
        auto end_v = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_w = end_v - start_v;

        double *err = new double[L];

        DynamicGridUtils::compute_error(err, phi, x_true, L);
        double err_norm = DynamicGridUtils::norm(err, L);
        double x_true_norm = DynamicGridUtils::norm(x_true, L);
        double rel_err = err_norm / x_true_norm;
        cout << "  Final Relative L2 Error: " << rel_err << endl;
        std::cout << "  Elapsed Time: " << elapsed_w.count() << " seconds\n";
        return elapsed_w.count();
    }

    double run_w_cycle(bool err_vector)
    {
        int L = N * N;
        double h = 1.0 / (N - 1);
        // Unified (device-accessible) memory
        double *phi;
        double *f;
        double *x_true = new double[L];
        size_t bytes = L * sizeof(double);
        cudaMallocManaged(&phi, bytes);
        cudaMallocManaged(&f, bytes);
        // cudaDeviceSynchronize(); // Ensure memory is ready

        DynamicGridUtils::initialize_zeros(phi, L);
        DynamicGridUtils::compute_rhs(f, N, N, h); // Host version
        DynamicGridUtils::compute_exact_solution(x_true, h, N, N);
        ParallelMultiGridSolver parallel_mg_solver(alpha);

        auto start_v = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < mg_max_iterations; iter++)
            parallel_mg_solver.w_cycle(phi, f, N, h);
        auto end_v = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_w = end_v - start_v;

        double *err = new double[L];

        DynamicGridUtils::compute_error(err, phi, x_true, L);
        if (err_vector)
        {
            this->err_vec.assign(err, err + L);
            std::vector<std::vector<double>> err_vect_iteration;
            err_vect_iteration.push_back(this->err_vec);
            save_errors_vector_to_file_last_iteration_gpu(err_vect_iteration);
        }
        double err_norm = DynamicGridUtils::norm(err, L);
        double x_true_norm = DynamicGridUtils::norm(x_true, L);
        double rel_err = err_norm / x_true_norm;
        cout << "  Final Relative L2 Error: " << rel_err << endl;
        std::cout << "  Elapsed Time: " << elapsed_w.count() << " seconds\n";
        return elapsed_w.count();
    }

private:
    void run_jacobi()
    {
        int L = N * N;
        double h = 1.0 / (N - 1);
        auto smoother = new JacobiSmoother(epsilon);
        // Host memory
        double *x_initial = new double[L];
        double *f = new double[L];
        double *res_host = new double[L];
        double *x_out = new double[L];

        // Unified (device-accessible) memory
        double *x_dev_initial;
        double *f_dev;
        double *res_dev;
        double *x_out_dev;

        size_t bytes = L * sizeof(double);
        cudaMallocManaged(&x_dev_initial, bytes);
        cudaMallocManaged(&f_dev, bytes);
        cudaMallocManaged(&res_dev, bytes);
        cudaMallocManaged(&x_out_dev, bytes);

        cudaDeviceSynchronize(); // Ensure memory is ready

        // Init host and device arrays
        DynamicGridUtils::initialize_zeros(x_initial, L);
        cudaDeviceSynchronize();                                        // Ensure previous device ops are done
        cudaMemPrefetchAsync(x_dev_initial, bytes, cudaCpuDeviceId, 0); // Prefetch to CPU
        cudaDeviceSynchronize();                                        // Wait for prefetch to complete
        DynamicGridUtils::initialize_zeros(x_dev_initial, L);
        DynamicGridUtils::initialize_zeros(res_host, L);
        DynamicGridUtils::initialize_zeros(res_dev, L);
        DynamicGridUtils::initialize_zeros(x_out, L);
        DynamicGridUtils::initialize_zeros(x_out_dev, L);

        // Compute RHS on both host and device memory (they are the same in content)
        DynamicGridUtils::compute_rhs(f, N, N, h);     // Host version
        DynamicGridUtils::compute_rhs(f_dev, N, N, h); // Unified memory for device use

        if (N < 4096)
        {
            // Measure host-side Jacobi time
            auto start_cpu = std::chrono::high_resolution_clock::now();
            smoother->smooth(x_out, f, N, N, h, 100);
            auto end_cpu = std::chrono::high_resolution_clock::now();
            double time_cpu = std::chrono::duration<double>(end_cpu - start_cpu).count();
            time_jacobi_cpu.push_back({num_thread, N, time_cpu});
        }

        // Measure device-side Jacobi time
        Parallel device;
        auto start_gpu = std::chrono::high_resolution_clock::now();
        device.ComputeJacobi(x_dev_initial, f_dev, N, N, h, 100);
        cudaDeviceSynchronize();
        auto end_gpu = std::chrono::high_resolution_clock::now();
        double time_gpu = std::chrono::duration<double>(end_gpu - start_gpu).count();
        time_jacobi_gpu.push_back({num_thread, N, time_gpu});

        // cout << DynamicGridUtils::compare_vector(x_dev_initial, x_out, L) << endl;

        // Clean up
        delete[] x_initial;
        delete[] f;
        delete[] res_host;
        cudaFree(x_dev_initial);
        cudaFree(f_dev);
        cudaFree(res_dev);
    }
    void run_residual()
    {
        int L = N * N;
        double h = 1.0 / (N - 1);

        // Host memory
        double *x_initial = new double[L];
        double *f = new double[L];
        double *res_host = new double[L];

        // Unified (device-accessible) memory
        double *x_dev_initial;
        double *f_dev;
        double *res_dev;

        size_t bytes = L * sizeof(double);
        cudaMallocManaged(&x_dev_initial, bytes);
        cudaMallocManaged(&f_dev, bytes);
        cudaMallocManaged(&res_dev, bytes);
        cudaDeviceSynchronize(); // Ensure memory is ready

        // Init host and device arrays
        DynamicGridUtils::initialize_zeros(x_initial, L);
        cudaDeviceSynchronize();                                        // Ensure previous device ops are done
        cudaMemPrefetchAsync(x_dev_initial, bytes, cudaCpuDeviceId, 0); // Prefetch to CPU
        cudaDeviceSynchronize();                                        // Wait for prefetch to complete
        DynamicGridUtils::initialize_zeros(x_dev_initial, L);
        DynamicGridUtils::initialize_zeros(res_host, L);
        DynamicGridUtils::initialize_zeros(res_dev, L);

        // Compute RHS on both host and device memory (they are the same in content)
        DynamicGridUtils::compute_rhs(f, N, N, h);     // Host version
        DynamicGridUtils::compute_rhs(f_dev, N, N, h); // Unified memory for device use

        if (N < 4096)
        {
            // Measure host-side residual time
            auto start_cpu = std::chrono::high_resolution_clock::now();
            DynamicGridUtils::compute_residual(res_host, x_initial, f, N, N, h);
            auto end_cpu = std::chrono::high_resolution_clock::now();
            double time_cpu = std::chrono::duration<double>(end_cpu - start_cpu).count();
            time_residual_cpu.push_back({num_thread, N, time_cpu});
        }

        // Measure device-side residual time
        Parallel device;
        auto start_gpu = std::chrono::high_resolution_clock::now();
        device.ComputeResidual(res_dev, x_dev_initial, f_dev, N, N, h);
        cudaDeviceSynchronize();
        auto end_gpu = std::chrono::high_resolution_clock::now();
        double time_gpu = std::chrono::duration<double>(end_gpu - start_gpu).count();
        time_residual_gpu.push_back({num_thread, N, time_gpu});

        // cout << DynamicGridUtils::compare_vector(res_dev, res_host, L) << endl;

        // Clean up
        delete[] x_initial;
        delete[] f;
        delete[] res_host;
        cudaFree(x_dev_initial);
        cudaFree(f_dev);
        cudaFree(res_dev);
    }

    void run_restriction()
    {
        int L = N * N;
        int n_rest = N / 2;
        double h = 1.0 / (N - 1);
        double *f = new double[L];
        double *f_dev;

        int l_rest = n_rest * n_rest;
        double *output = new double[l_rest];
        double *output_dev;

        size_t bytes = L * sizeof(double);
        size_t bytes_restr = l_rest * sizeof(double);

        cudaMallocManaged(&f_dev, bytes);
        cudaMallocManaged(&output_dev, bytes_restr);

        DynamicGridUtils::initialize_zeros(f, L);
        DynamicGridUtils::initialize_zeros(f_dev, L);

        // Compute RHS on both host and device memory (they are the same in content)
        DynamicGridUtils::compute_rhs(f, N, N, h);     // Host version
        DynamicGridUtils::compute_rhs(f_dev, N, N, h); // Unified memory for device use

        if (N < 4096)
        {
            // Measure host-side restriction time
            auto start_cpu = std::chrono::high_resolution_clock::now();
            restrict_full_weighting_2(f, output, N, n_rest);
            auto end_cpu = std::chrono::high_resolution_clock::now();
            double time_cpu = std::chrono::duration<double>(end_cpu - start_cpu).count();
            time_restriction_cpu.push_back({num_thread, N, time_cpu});
        }

        // Measure device-side restriction time
        Parallel device;
        auto start_gpu = std::chrono::high_resolution_clock::now();
        device.ComputeRestriction(f_dev, output_dev, N, n_rest);
        cudaDeviceSynchronize();
        auto end_gpu = std::chrono::high_resolution_clock::now();
        double time_gpu = std::chrono::duration<double>(end_gpu - start_gpu).count();
        time_restriction_gpu.push_back({num_thread, N, time_gpu});

        // cout << DynamicGridUtils::compare_vector(output_dev, output, l_rest) << endl;

        // Clean up
        delete[] f;
        delete[] output;
        cudaFree(f_dev);
        cudaFree(output_dev);
    }

    void run_prolungator()
    {
        int L = N * N;
        int n_prolungator = N * 2;
        double h = 1.0 / (N - 1);
        double *f = new double[L];
        double *f_dev;

        int l_prol = n_prolungator * n_prolungator;
        double *output = new double[l_prol];
        double *output_dev;

        size_t bytes = L * sizeof(double);
        size_t bytes_restr = l_prol * sizeof(double);

        DynamicGridUtils::initialize_zeros(f, L);
        DynamicGridUtils::initialize_zeros(output, l_prol);

        cudaMallocManaged(&f_dev, bytes);
        cudaMallocManaged(&output_dev, bytes_restr);
        DynamicGridUtils::initialize_zeros(output_dev, l_prol);
        DynamicGridUtils::initialize_zeros(f_dev, L);
        // Compute RHS on both host and device memory (they are the same in content)
        DynamicGridUtils::compute_rhs(f, N, N, h);     // Host version
        DynamicGridUtils::compute_rhs(f_dev, N, N, h); // Unified memory for device use

        if (N < 4096)
        {
            // Measure host-side prolongation time
            auto start_cpu = std::chrono::high_resolution_clock::now();
            prolongation_2(output, f, n_prolungator, N);
            auto end_cpu = std::chrono::high_resolution_clock::now();
            double time_cpu = std::chrono::duration<double>(end_cpu - start_cpu).count();
            time_prolungator_cpu.push_back({num_thread, N, time_cpu});
        }

        // Measure device-side prolongation time
        Parallel device;
        auto start_gpu = std::chrono::high_resolution_clock::now();
        device.ComputeProlungator(f_dev, output_dev, N, n_prolungator);
        cudaDeviceSynchronize();
        auto end_gpu = std::chrono::high_resolution_clock::now();
        double time_gpu = std::chrono::duration<double>(end_gpu - start_gpu).count();
        time_prolungator_gpu.push_back({num_thread, N, time_gpu});
        // cout << DynamicGridUtils::compare_vector(output_dev, output, l_prol) << endl;

        // Clean up
        delete[] f;
        delete[] output;
        cudaFree(f_dev);
        cudaFree(output_dev);
    }

    void run_all_methods()
    {
        run_residual();
        run_jacobi();
        run_restriction();
        run_prolungator();
    }
};
