#ifndef SAVE_TO_FILE_HPP
#define SAVE_TO_FILE_HPP
#include <vector>
#include <fstream>
#include <iostream>

#include <sys/stat.h>
#include <sys/types.h>

void create_directory_if_not_exists(const std::string &path)
{
    struct stat info;
    if (stat(path.c_str(), &info) != 0)
    {
        mkdir(path.c_str(), 0777);
    }
}

void save_residuals_to_file(std::vector<double> *residuals_jacobian, std::vector<double> *residuals_steepest, std::vector<double> *residuals_gs, std::vector<double> *residuals_cg)
{
    create_directory_if_not_exists("OUTPUT_RESULT");

    std::ofstream file_jacobian("OUTPUT_RESULT/residuals_jacobian.txt");
    if (file_jacobian.is_open())
    {
        for (const auto &residual : *residuals_jacobian)
        {
            file_jacobian << residual << "\n";
        }
        file_jacobian.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Jacobian residuals.\n";
    }

    std::ofstream file_gs("OUTPUT_RESULT/residuals_gs.txt");
    if (file_gs.is_open())
    {
        for (const auto &residual : *residuals_gs)
        {
            file_gs << residual << "\n";
        }
        file_gs.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Gauss-Seidel residuals.\n";
    }

    std::ofstream file_cg("OUTPUT_RESULT/residuals_cg.txt");
    if (file_cg.is_open())
    {
        for (const auto &residual : *residuals_cg)
        {
            file_cg << residual << "\n";
        }
        file_cg.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Conjugate Gradient residuals.\n";
    }

    std::ofstream file_steepest("OUTPUT_RESULT/residuals_steepest_descent.txt");
    if (file_steepest.is_open())
    {
        for (const auto &residual : *residuals_steepest)
        {
            file_steepest << residual << "\n";
        }
        file_steepest.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Steepest Descent residuals.\n";
    }
}

void save_error_to_file(std::vector<double> *error_jacobian, std::vector<double> *error_steepest, std::vector<double> *error_gs, std::vector<double> *error_cg)
{
    create_directory_if_not_exists("OUTPUT_RESULT");

    std::ofstream file_jacobian("OUTPUT_RESULT/error_jacobian.txt");
    if (file_jacobian.is_open())
    {
        for (const auto &error : *error_jacobian)
        {
            file_jacobian << error << "\n";
        }
        file_jacobian.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Jacobian errors.\n";
    }

    std::ofstream file_steepest("OUTPUT_RESULT/error_steepest_descent.txt");
    if (file_steepest.is_open())
    {
        for (const auto &error : *error_steepest)
        {
            file_steepest << error << "\n";
        }
        file_steepest.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Steepest Descent errors.\n";
    }

    std::ofstream file_gs("OUTPUT_RESULT/error_gs.txt");
    if (file_gs.is_open())
    {
        for (const auto &error : *error_gs)
        {
            file_gs << error << "\n";
        }
        file_gs.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Gauss-Seidel errors.\n";
    }

    std::ofstream file_cg("OUTPUT_RESULT/error_cg.txt");
    if (file_cg.is_open())
    {
        for (const auto &error : *error_cg)
        {
            file_cg << error << "\n";
        }
        file_cg.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Conjugate Gradient errors.\n";
    }
}

void save_timings_to_file(std::vector<std::pair<int, double>> &timings_jacobi, std::vector<std::pair<int, double>> &timings_gs, std::vector<std::pair<int, double>> &timings_steepest, std::vector<std::pair<int, double>> &timings_cg)
{
    create_directory_if_not_exists("OUTPUT_RESULT");

    std::ofstream file_jacobian("OUTPUT_RESULT/timings_jacobian.txt");
    if (file_jacobian.is_open())
    {
        for (const auto &timing : timings_jacobi)
        {
            file_jacobian << timing.first << " " << timing.second << "\n";
        }
        file_jacobian.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Jacobian timings.\n";
    }

    std::ofstream file_steepest("OUTPUT_RESULT/timings_steepest_descent.txt");
    if (file_steepest.is_open())
    {
        for (const auto &timing : timings_steepest)
        {
            file_steepest << timing.first << " " << timing.second << "\n";
        }
        file_steepest.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Steepest Descent timings.\n";
    }

    std::ofstream file_gs("OUTPUT_RESULT/timings_gs.txt");
    if (file_gs.is_open())
    {
        for (const auto &timing : timings_gs)
        {
            file_gs << timing.first << " " << timing.second << "\n";
        }
        file_gs.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Gauss-Seidel timings.\n";
    }

    std::ofstream file_cg("OUTPUT_RESULT/timings_cg.txt");
    if (file_cg.is_open())
    {
        for (const auto &timing : timings_cg)
        {
            file_cg << timing.first << " " << timing.second << "\n";
        }
        file_cg.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Conjugate Gradient timings.\n";
    }
}

void save_error_h_to_file(std::vector<std::pair<int, double>> &error_j, std::vector<std::pair<int, double>> &error_gs, std::vector<std::pair<int, double>> &error_steepest, std::vector<std::pair<int, double>> &error_cg)
{
    create_directory_if_not_exists("OUTPUT_RESULT");

    std::ofstream file_jacobian("OUTPUT_RESULT/h_errors_jacobi.txt");
    if (file_jacobian.is_open())
    {
        for (const auto &error : error_j)
        {
            file_jacobian << error.first << " " << error.second << "\n";
        }
        file_jacobian.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Jacobian errors.\n";
    }

    std::ofstream file_steepest("OUTPUT_RESULT/h_errors_steepest_descent.txt");
    if (file_steepest.is_open())
    {
        for (const auto &error : error_steepest)
        {
            file_steepest << error.first << " " << error.second << "\n";
        }
        file_steepest.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Steepest Descent errors.\n";
    }

    std::ofstream file_gs("OUTPUT_RESULT/h_errors_gs.txt");
    if (file_gs.is_open())
    {
        for (const auto &error : error_gs)
        {
            file_gs << error.first << " " << error.second << "\n";
        }
        file_gs.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Gauss-Seidel errors.\n";
    }

    std::ofstream file_cg("OUTPUT_RESULT/h_errors_cg.txt");
    if (file_cg.is_open())
    {
        for (const auto &error : error_cg)
        {
            file_cg << error.first << " " << error.second << "\n";
        }
        file_cg.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing Conjugate Gradient errors.\n";
    }
}

#endif