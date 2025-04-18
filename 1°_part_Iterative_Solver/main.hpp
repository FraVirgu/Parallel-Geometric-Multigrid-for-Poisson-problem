#include "head.hpp"
#include "gauss_seidel.hpp"
#include "jacobian.hpp"
#include "steepest_descent.hpp"
#include "conjugate_gradient.hpp"
#include <chrono>
#include "save_file.hpp"

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
    cout << "error jacobian: " << error_jacobian->back() << endl;
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
        cout << "error cg: " << error_cg->back() << endl;
    }
    else
    {
        std::cout << "Did not converge within the maximum number of iterations." << std::endl;
        std::cout << "Residual reached: " << *residual_reached << std::endl;
        cout << "error cg: " << error_cg->back() << endl;
    }
}

vector<int> n_initialization()
{
    vector<int> n;
    for (int i = 8; i <= 256; i *= 2)
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
