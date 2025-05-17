#include "Parallel_Method.cu"
#include "../Smoother.hpp"
class ParallelMultiGridSolver
{
private:
    int alpha;
    int N_final;
    int v1 = 1;
    int v2 = 1;

public:
    int N_cpu = 17;
    double *final_solution;
    double epsilon = 1e-7;

    explicit ParallelMultiGridSolver(int alpha_)
        : alpha(alpha_)
    {
    }

    void v_cycle(double *phi, double *f, int N, double h)
    {
        if (N <= N_cpu)
        {
            JacobiSmoother smoother(epsilon);
            MultigridSolver mg(&smoother, alpha, N);
            mg.v_cycle(phi, f, N, h);
            return;
        }
        else
        {
            Parallel device;
            device.ComputeJacobi(phi, f, N, N, h, v1);
            int L = N * N;
            double *res_dev;

            size_t bytes = L * sizeof(double);
            cudaMallocManaged(&res_dev, bytes);
            cudaDeviceSynchronize(); // Ensure memory is ready
            device.ComputeResidual(res_dev, phi, f, N, N, h);

            // Restrict residual to coarser grid
            int Nc = (N - 1) / 2 + 1;
            int Lc = Nc * Nc;

            double *res_coarse;
            size_t bytes_coarse = Lc * sizeof(double);
            cudaMallocManaged(&res_coarse, bytes_coarse);
            cudaDeviceSynchronize(); // Ensure memory is ready
            device.ComputeRestriction(res_dev, res_coarse, N, Nc);

            double *e_coarse;
            cudaMallocManaged(&e_coarse, bytes_coarse);
            cudaDeviceSynchronize(); // Ensure memory is ready
            v_cycle(e_coarse, res_coarse, Nc, 2 * h);

            device.ComputeProlungator(e_coarse, phi, Nc, N);
            device.ComputeJacobi(phi, f, N, N, h, v2);
        }
    }

    void w_cycle(double *phi, double *f, int N, double h)
    {
        if (N <= N_cpu)
        {
            JacobiSmoother smoother(epsilon);
            MultigridSolver mg(&smoother, alpha, N);
            mg.w_cycle(phi, f, N, h);
            return;
        }
        else
        {
            Parallel device;
            device.ComputeJacobi(phi, f, N, N, h, v1);
            int L = N * N;
            double *res_dev;

            size_t bytes = L * sizeof(double);
            cudaMallocManaged(&res_dev, bytes);
            cudaDeviceSynchronize(); // Ensure memory is ready
            device.ComputeResidual(res_dev, phi, f, N, N, h);

            // Restrict residual to coarser grid
            int Nc = (N - 1) / 2 + 1;
            int Lc = Nc * Nc;

            double *res_coarse;
            size_t bytes_coarse = Lc * sizeof(double);
            cudaMallocManaged(&res_coarse, bytes_coarse);
            cudaDeviceSynchronize(); // Ensure memory is ready
            device.ComputeRestriction(res_dev, res_coarse, N, Nc);

            double *e_coarse;
            cudaMallocManaged(&e_coarse, bytes_coarse);
            cudaDeviceSynchronize(); // Ensure memory is ready
            for (int i = 0; i < alpha; i++)
                w_cycle(e_coarse, res_coarse, Nc, 2 * h);

            device.ComputeProlungator(e_coarse, phi, Nc, N);
            device.ComputeJacobi(phi, f, N, N, h, v2);
        }
    }
};