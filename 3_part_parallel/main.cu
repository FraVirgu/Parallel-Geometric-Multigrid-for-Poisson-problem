#include "ParallelTestRunner.cu"
#include "../2_part_MG/MultiGridTestRunner.hpp"
std::vector<int> N_list = {33, 65, 129, 257, 513, 1025, 2049};
std::vector<int> N_thread_list = {16, 32};
int main()
{
    // Allocate managed memory for temporary data
    double *tmp;
    cudaMallocManaged(&tmp, sizeof(double) * 129);
    // Ensure memory is accessible on the GPU
    cudaDeviceSynchronize();
    // Free the allocated memory after use
    cudaFree(tmp);

    const int alpha = 3;
    const double epsilon = 1e-7;
    const int mg_max_iterations = 3;

    /*
    ParallelTestRunner parallel_runner(0, mg_max_iterations, alpha);
    // PLOT THE TIME DIFFERENCES BETWEEN ALL METHOD USED IN MG PARALLEL vs SEQUENTIAL
    parallel_runner.plotTimeSequentialVsParallel(N_list, N_thread_list);
    */

    /*
    // TIMINGS RESULTS

     // CPU
     MultigridTestRunner runner(epsilon, mg_max_iterations, alpha);
     runner.run_all_cycles_time_h(N_list);
   */
    // GPU
    ParallelTestRunner parallel_runner(0, mg_max_iterations, alpha);
    parallel_runner.run_all_cycles(N_list);

    // SAVE ERROR VECTOR
    /*

  // CPU
    N_list = {2049};
    MultigridTestRunner runner(epsilon, mg_max_iterations, alpha);
    runner.run_w_cycles_err_vector_iteration(N_list);

    // GPU
    N_list = {8193};
    parallel_runner.run_w_cycles_err_vector_iteration(N_list);
    */

    return 0;
}