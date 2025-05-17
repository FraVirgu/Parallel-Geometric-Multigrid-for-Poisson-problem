#include <vector>
#include <fstream>
#include <iostream>

#include <filesystem>

void create_directory_if_not_exists_3(const std::string &path)
{
    std::filesystem::create_directories(path);
}

void mg_save_error_to_file(int iter, std::vector<double> *error_v_cycle, std::vector<double> *error_w_cycle, std::vector<double> *error_f_cycle)
{
    create_directory_if_not_exists_3("OUTPUT_RESULT");

    std::ofstream file_v_cycle("OUTPUT_RESULT/error_v_cycle" + std::to_string(iter) + ".txt");
    if (file_v_cycle.is_open())
    {
        for (const auto &error : *error_v_cycle)
        {
            file_v_cycle << error << "\n";
        }
        file_v_cycle.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing V-Cycle errors.\n";
    }

    std::ofstream file_w_cycle("OUTPUT_RESULT/error_w_cycle" + std::to_string(iter) + ".txt");
    if (file_w_cycle.is_open())
    {
        for (const auto &error : *error_w_cycle)
        {
            file_w_cycle << error << "\n";
        }
        file_w_cycle.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing W-Cycle errors.\n";
    }

    std::ofstream file_f_cycle("OUTPUT_RESULT/error_f_cycle" + std::to_string(iter) + ".txt");
    if (file_f_cycle.is_open())
    {
        for (const auto &error : *error_f_cycle)
        {
            file_f_cycle << error << "\n";
        }
        file_f_cycle.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing F-Cycle errors.\n";
    }
}

void save_timings_to_file(std::vector<std::pair<int, double>> &timings_v_cycle, std::vector<std::pair<int, double>> &timings_w_cycle, std::vector<std::pair<int, double>> &timings_f_cycle)
{
    create_directory_if_not_exists_3("OUTPUT_RESULT");

    std::ofstream file_v_cycle("OUTPUT_RESULT/timings_v_cycle.txt");
    if (file_v_cycle.is_open())
    {
        for (const auto &timing : timings_v_cycle)
        {
            file_v_cycle << timing.first << " " << timing.second << "\n";
        }
        file_v_cycle.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing V-Cycle timings.\n";
    }

    std::ofstream file_w_cycle("OUTPUT_RESULT/timings_w_cycle.txt");
    if (file_w_cycle.is_open())
    {
        for (const auto &timing : timings_w_cycle)
        {
            file_w_cycle << timing.first << " " << timing.second << "\n";
        }
        file_w_cycle.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing W-Cycle timings.\n";
    }

    std::ofstream file_f_cycle("OUTPUT_RESULT/timings_f_cycle.txt");
    if (file_f_cycle.is_open())
    {
        for (const auto &timing : timings_f_cycle)
        {
            file_f_cycle << timing.first << " " << timing.second << "\n";
        }
        file_f_cycle.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing F-Cycle timings.\n";
    }
}

void save_error_h_to_file(int iteration_performed, std::vector<std::pair<int, double>> &error_v_cycle, std::vector<std::pair<int, double>> &error_w_cycle, std::vector<std::pair<int, double>> &error_f_cycle)
{
    create_directory_if_not_exists_3("OUTPUT_RESULT");

    std::ofstream file_v_cycle("OUTPUT_RESULT/h_errors_v_cycle" + std::to_string(iteration_performed) + ".txt");
    if (file_v_cycle.is_open())
    {
        for (const auto &error : error_v_cycle)
        {
            file_v_cycle << error.first << " " << error.second << "\n";
        }
        file_v_cycle.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing V-Cycle errors.\n";
    }

    std::ofstream file_w_cycle("OUTPUT_RESULT/h_errors_w_cycle" + std::to_string(iteration_performed) + ".txt");
    if (file_w_cycle.is_open())
    {
        for (const auto &error : error_w_cycle)
        {
            file_w_cycle << error.first << " " << error.second << "\n";
        }
        file_w_cycle.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing W-Cycle errors.\n";
    }

    std::ofstream file_f_cycle("OUTPUT_RESULT/h_errors_f_cycle" + std::to_string(iteration_performed) + ".txt");
    if (file_f_cycle.is_open())
    {
        for (const auto &error : error_f_cycle)
        {
            file_f_cycle << error.first << " " << error.second << "\n";
        }
        file_f_cycle.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing F-Cycle errors.\n";
    }
}
