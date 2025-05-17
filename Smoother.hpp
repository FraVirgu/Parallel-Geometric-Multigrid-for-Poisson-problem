#ifndef SMOOTHER_HPP
#define SMOOTHER_HPP
#include <vector>
#include "DynamicGridUtils.hpp"
#include <stdio.h>
#include "save_vector_err_file.hpp"
using namespace std;
class Smoother
{
protected:
    double epsilon; // Convergence tolerance
    int iteration_performed;

public:
    bool test = false;
    explicit Smoother(double eps = 1e-6) : epsilon(eps) {}

    void switch_test_mode()
    {
        test = true;
    }

    virtual void smooth(double *x, double *f,
                        int width, int height,
                        double h, int num_iter,
                        double *x_true = nullptr,
                        std::vector<double> *residuals = nullptr,
                        std::vector<double> *errors = nullptr) = 0;

    virtual ~Smoother() {}
};

class JacobiSmoother : public Smoother
{
public:
    using Smoother::Smoother; // Inherit constructor

    void smooth(double *x, double *f,
                int width, int height,
                double h, int num_iter,
                double *x_true = nullptr,
                std::vector<double> *residuals = nullptr,
                std::vector<double> *errors = nullptr) override
    {
        int L = width * height;
        double *jacobi_output = new double[L];
        std::copy(x, x + L, jacobi_output); // start from same initial condition
        std::vector<std::vector<double>> err_vect_iteration;

        if (test)
        {
            double *err_iteration = new double[L];
            DynamicGridUtils::compute_error(err_iteration, x, x_true, L);
            std::vector<double> err_vector(err_iteration, err_iteration + L);
            err_vect_iteration.push_back(err_vector);
            delete[] err_iteration;
        }

        for (int iter = 0; iter <= num_iter; ++iter)
        {
            for (int y = 1; y < height - 1; ++y)
            {
                for (int x_pos = 1; x_pos < width - 1; ++x_pos)
                {
                    int idx = y * width + x_pos;
                    jacobi_output[idx] = 0.25 * ((h * h * f[idx]) +
                                                 x[idx - 1] + x[idx + 1] +
                                                 x[idx - width] + x[idx + width]);
                }
            }

            std::copy(jacobi_output, jacobi_output + L, x); // copy result to x for next iteration

            // --- Residual computation
            double *r = new double[L];
            DynamicGridUtils::compute_residual(r, x, f, width, height, h);
            double res_norm = DynamicGridUtils::norm(r, L);
            if (residuals)
            {

                residuals->push_back(res_norm);
            }

            if (res_norm < epsilon)
            {
                iteration_performed = iter;
                break;
            }

            // --- Error computation (if exact solution is provided)
            if (errors && x_true)
            {
                double *err = new double[L];
                DynamicGridUtils::compute_error(err, x, x_true, L);
                double norm_x_true = DynamicGridUtils::norm(x_true, L);
                double norm_err = DynamicGridUtils::norm(err, L);
                errors->push_back(norm_err / norm_x_true);
                delete[] err;
            }
            if (test)
            {
                if (iter % 10 == 0)
                {
                    double *err_iteration = new double[L];
                    DynamicGridUtils::compute_error(err_iteration, x, x_true, L);
                    std::vector<double> err_vector(err_iteration, err_iteration + L);
                    err_vect_iteration.push_back(err_vector);
                    delete[] err_iteration;
                }
            }
        }

        delete[] jacobi_output;
        if (test)
            save_errors_vector_to_file(err_vect_iteration);
    }
};

class GaussSeidelSmoother : public Smoother
{
public:
    using Smoother::Smoother;

    void smooth(double *x, double *f,
                int width, int height,
                double h, int num_iter,
                double *x_true = nullptr,
                std::vector<double> *residuals = nullptr,
                std::vector<double> *errors = nullptr) override
    {
        int L = width * height;

        for (int iter = 0; iter < num_iter; ++iter)
        {
            for (int y = 1; y < height - 1; ++y)
            {
                for (int x_pos = 1; x_pos < width - 1; ++x_pos)
                {
                    int idx = y * width + x_pos;
                    x[idx] = 0.25 * (x[idx - 1] + x[idx + 1] +
                                     x[idx - width] + x[idx + width] +
                                     h * h * f[idx]);
                }
            }

            double *r = new double[L];
            DynamicGridUtils::compute_residual(r, x, f, width, height, h);
            double res_norm = DynamicGridUtils::norm(r, L);
            delete[] r;

            if (residuals)
                residuals->push_back(res_norm);
            if (res_norm < epsilon)
            {
                iteration_performed = iter;
                break;
            }

            if (errors && x_true)
            {
                double *err = new double[L];
                DynamicGridUtils::compute_error(err, x, x_true, L);
                errors->push_back(DynamicGridUtils::norm(err, L) / DynamicGridUtils::norm(x_true, L));
                delete[] err;
            }
        }
    }
};

class ConjugateGradientSmoother : public Smoother
{
public:
    using Smoother::Smoother;

    void smooth(double *x, double *f,
                int width, int height,
                double h, int num_iter,
                double *x_true = nullptr,
                std::vector<double> *residuals = nullptr,
                std::vector<double> *errors = nullptr) override
    {

        int L = width * height;
        double *r = new double[L];
        double *p_d = new double[L];
        double *Ap_d = new double[L];
        double *err = new double[L];
        DynamicGridUtils::initialize_zeros(x, L);
        DynamicGridUtils::initialize_zeros(r, L);
        DynamicGridUtils::initialize_zeros(p_d, L);
        DynamicGridUtils::initialize_zeros(Ap_d, L);

        DynamicGridUtils::compute_residual(r, x, f, width, height, h);

        double norm_residual = DynamicGridUtils::norm(r, L);
        DynamicGridUtils::compute_residual(r, x, f, width, height, h);

        if (residuals)
            residuals->push_back(norm_residual);

        if (errors && x_true)
        {
            DynamicGridUtils::compute_error(err, x, x_true, L);
            errors->push_back(DynamicGridUtils::norm(err, L) / DynamicGridUtils::norm(x_true, L));
        }

        std::copy(r, r + L, p_d);

        for (int iter = 0; iter < num_iter; ++iter)
        {
            double rTr = DynamicGridUtils::dot(r, r, L);
            DynamicGridUtils::apply_laplacian(p_d, Ap_d, width, height, h);
            double pAp = DynamicGridUtils::dot(p_d, Ap_d, L);
            double alpha = rTr / pAp;

            for (int i = 0; i < L; ++i)
                x[i] += alpha * p_d[i];
            for (int i = 0; i < L; ++i)
                r[i] -= alpha * Ap_d[i];

            norm_residual = DynamicGridUtils::norm(r, L);
            DynamicGridUtils::compute_residual(r, x, f, width, height, h);
            if (residuals)
                residuals->push_back(norm_residual);

            if (norm_residual < epsilon)
            {
                iteration_performed = iter;
                break;
            }

            if (errors && x_true)
            {
                DynamicGridUtils::compute_error(err, x, x_true, L);
                errors->push_back(DynamicGridUtils::norm(err, L) / DynamicGridUtils::norm(x_true, L));
            }

            double rTr_new = DynamicGridUtils::dot(r, r, L);
            double beta = rTr_new / rTr;
            for (int i = 0; i < L; ++i)
                p_d[i] = r[i] + beta * p_d[i];
        }

        // Final error update
        if (errors && x_true)
        {
            DynamicGridUtils::compute_error(err, x, x_true, L);
            errors->push_back(DynamicGridUtils::norm(err, L) / DynamicGridUtils::norm(x_true, L));
        }

        delete[] r;
        delete[] p_d;
        delete[] Ap_d;
        delete[] err;
    }
};

class SteepestDescentSmoother : public Smoother
{
public:
    using Smoother::Smoother;

    void smooth(double *x, double *f,
                int width, int height,
                double h, int num_iter,
                double *x_true = nullptr,
                std::vector<double> *residuals = nullptr,
                std::vector<double> *errors = nullptr) override
    {
        int L = width * height;
        double *r = new double[L];
        double *err = new double[L];

        DynamicGridUtils::initialize_zeros(x, L);
        DynamicGridUtils::initialize_zeros(r, L);
        for (int iter = 0; iter < num_iter; ++iter)
        {
            DynamicGridUtils::compute_residual(r, x, f, width, height, h);
            double res_norm = DynamicGridUtils::norm(r, L);
            if (residuals)
                residuals->push_back(res_norm);

            if (res_norm < epsilon)
            {
                iteration_performed = iter;
                break;
            }

            if (errors && x_true)
            {
                DynamicGridUtils::compute_error(err, x, x_true, L);
                errors->push_back(DynamicGridUtils::norm(err, L) / DynamicGridUtils::norm(x_true, L));
            }

            double alpha = DynamicGridUtils::compute_alpha_opt(r, width, height, h);
            for (int i = 0; i < L; ++i)
                x[i] += alpha * r[i];
        }

        // Final error update
        if (errors && x_true)
        {
            DynamicGridUtils::compute_error(err, x, x_true, L);
            errors->push_back(DynamicGridUtils::norm(err, L) / DynamicGridUtils::norm(x_true, L));
        }

        delete[] r;
        delete[] err;
    }
};

#endif