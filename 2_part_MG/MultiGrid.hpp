#ifndef MULTIGRID_SOLVER_HPP
#define MULTIGRID_SOLVER_HPP
#include <iostream>
#include <stdio.h>
#include <vector>
#include "../Smoother.hpp"
#include "../DynamicGridUtils.hpp"
using namespace std;
class MultigridSolver
{
private:
    Smoother *smoother; // Injected smoother strategy
    int alpha;          // W_cycle
    int N_final;        // F_cycle
    int v1 = 1;
    int v2 = 1;

public:
    int N_coarse = 5;
    double *final_solution;

    explicit MultigridSolver(Smoother *smoother_, int alpha_, int N_final_)
        : smoother(smoother_), alpha(alpha_), N_final(N_final_)
    {
        final_solution = new double[N_final * N_final];
    }

    void compute_coarsest_grid(const double *fine_input, double *&coarsest_output, int N_fine, int N_coarsest)
    {
        int N_current = N_fine;
        int L_current = N_current * N_current;

        // Copy initial fine data
        double *current = new double[L_current];
        std::copy(fine_input, fine_input + L_current, current);

        while (N_current > N_coarsest)
        {
            int N_next = (N_current - 1) / 2 + 1;
            int L_next = N_next * N_next;

            double *next = new double[L_next];
            DynamicGridUtils::initialize_zeros(next, L_next);

            // Apply restriction
            restrict_full_weighting(current, next, N_current, N_next);

            delete[] current;
            current = next;
            N_current = N_next;
        }

        // Output pointer to final coarsened result
        coarsest_output = current;
    }

    void v_cycle(double *phi, const double *f, int N, double h)
    {
        if (N <= N_coarse)
        {
            smoother->smooth(phi, const_cast<double *>(f), N, N, h, 10);
            return;
        }

        // Pre-smoothing
        smoother->smooth(phi, const_cast<double *>(f), N, N, h, v1);

        // Compute residual r = f - A * phi
        int L = N * N;
        double *residual = new double[L];
        DynamicGridUtils::compute_residual(residual, phi, f, N, N, h);

        // Restrict residual to coarser grid
        int Nc = (N - 1) / 2 + 1;
        int Lc = Nc * Nc;
        double *res_coarse = new double[Lc];
        DynamicGridUtils::initialize_zeros(res_coarse, Lc);
        restrict_full_weighting(residual, res_coarse, N, Nc);

        // Solve error on coarse grid recursively
        double *e_coarse = new double[Lc];
        DynamicGridUtils::initialize_zeros(e_coarse, Lc);
        v_cycle(e_coarse, res_coarse, Nc, 2 * h);

        // Prolongate and correct
        prolongation(phi, e_coarse, N, Nc);

        // Post-smoothing
        smoother->smooth(phi, const_cast<double *>(f), N, N, h, v2);

        delete[] residual;
        delete[] res_coarse;
        delete[] e_coarse;
    }

    void w_cycle(double *phi, const double *f, int N, double h)
    {
        if (N <= N_coarse)
        {
            smoother->smooth(phi, const_cast<double *>(f), N, N, h, 10);
            return;
        }

        // Pre-smoothing
        smoother->smooth(phi, const_cast<double *>(f), N, N, h, v1);

        // Compute residual
        int L = N * N;
        double *residual = new double[L];
        DynamicGridUtils::compute_residual(residual, phi, f, N, N, h);

        // Restrict to coarse grid
        int Nc = (N - 1) / 2 + 1;
        int Lc = Nc * Nc;
        double *res_coarse = new double[Lc];
        DynamicGridUtils::initialize_zeros(res_coarse, Lc);
        restrict_full_weighting(residual, res_coarse, N, Nc);

        // Allocate error on coarse grid
        double *e_coarse = new double[Lc];
        DynamicGridUtils::initialize_zeros(e_coarse, Lc);

        // Perform alpha recursive solves on the coarse grid
        for (int i = 0; i < alpha; ++i)
            w_cycle(e_coarse, res_coarse, Nc, 2.0 * h); // recursion is W-style

        // Prolongate and correct
        prolongation(phi, e_coarse, N, Nc);

        // Post-smoothing
        smoother->smooth(phi, const_cast<double *>(f), N, N, h, v2);

        delete[] residual;
        delete[] res_coarse;
        delete[] e_coarse;
    }

    void f_cycle(double *phi, const double *f, int N_init, double h_init)
    {
        int N = N_init;
        double h = h_init;

        // Allocate and initialize initial phi and f
        int L = N * N;
        double *phi_current = new double[L];
        double *f_current = new double[L];
        std::copy(phi, phi + L, phi_current);
        std::copy(f, f + L, f_current);

        while (N < N_final)
        {
            // Smooth on current level
            smoother->smooth(phi_current, f_current, N, N, h, 3);

            // Prolongate to finer grid
            int N_fine = 2 * N - 1;
            int L_fine = N_fine * N_fine;

            double *phi_fine = new double[L_fine];
            double *f_fine = new double[L_fine];
            DynamicGridUtils::initialize_zeros(phi_fine, L_fine);
            DynamicGridUtils::compute_rhs(f_fine, N_fine, N_fine, h / 2);

            prolongation(phi_fine, phi_current, N_fine, N);

            // Perform a V-cycle at the finer level
            v_cycle(phi_fine, f_fine, N_fine, h / 2);

            // Clean up coarse level
            delete[] phi_current;
            delete[] f_current;

            // Move to finer level
            phi_current = phi_fine;
            f_current = f_fine;
            N = N_fine;
            h /= 2;
        }
        // Copy final solution back to phi
        std::copy(phi_current, phi_current + (N) * (N), final_solution);
        delete[] phi_current;
        delete[] f_current;
    }

private:
    // Restriction using full-weighting
    void restrict_full_weighting(const double *fine, double *coarse, int Nf, int Nc)
    {
        for (int jc = 1; jc < Nc - 1; ++jc)
        {
            for (int ic = 1; ic < Nc - 1; ++ic)
            {
                int i_f = 2 * ic;
                int j_f = 2 * jc;

                int idx_c = jc * Nc + ic;
                int idx_f = j_f * Nf + i_f;

                coarse[idx_c] = 0.25 * fine[idx_f] +
                                0.125 * (fine[idx_f + 1] + fine[idx_f - 1] + fine[idx_f + Nf] + fine[idx_f - Nf]) +
                                0.0625 * (fine[idx_f - Nf - 1] + fine[idx_f - Nf + 1] +
                                          fine[idx_f + Nf - 1] + fine[idx_f + Nf + 1]);
            }
        }
    }

    // Prolongation using bilinear interpolation
    void prolongation(double *fine, const double *coarse, int Nf, int Nc)
    {
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
    }
};

#endif // MULTIGRID_SOLVER_HPP
