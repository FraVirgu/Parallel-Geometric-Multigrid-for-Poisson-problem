#ifndef SAVE_ERR_VECTOR_HPP
#define SAVE_ERR_VECTOR_HPP
#include <vector>
#include <fstream>
#include <iostream>

#include <sys/stat.h>
#include <sys/types.h>

void create_directory_if_not_exists_2(const std::string &path)
{
    struct stat info;
    if (stat(path.c_str(), &info) != 0)
    {
        mkdir(path.c_str(), 0777);
    }
}

void save_errors_vector_to_file(std::vector<std::vector<double>> err_vect_iteration)
{
    create_directory_if_not_exists_2("./OUTPUT_RESULT/ERR_VECTOR");
    for (int i = 0; i < err_vect_iteration.size(); i++)
    {
        std::ofstream file("./OUTPUT_RESULT/ERR_VECTOR/iteration_" + std::to_string(i * 10) + ".txt");
        if (file.is_open())
        {
            file << err_vect_iteration[i].size() << "\n"; // Write the length of the vector in the first line
            for (const auto &error : err_vect_iteration[i])
            {
                file << error << "\n"; // Write each component on a new line
            }
            file.close();
        }
        else
        {
            std::cerr << "Unable to open file for writing Jacobian errors.\n";
        }
    }
}

void save_errors_vector_to_file_last_iteration_cpu(std::vector<std::vector<double>> err_vect_iteration)
{
    create_directory_if_not_exists_2("./OUTPUT_RESULT/ERR_VECTOR");
    for (int i = 0; i < err_vect_iteration.size(); i++)
    {
        std::ofstream file("./OUTPUT_RESULT/ERR_VECTOR/iteration_last_cpu.txt");
        if (file.is_open())
        {
            file << err_vect_iteration[i].size() << "\n"; // Write the length of the vector in the first line
            for (const auto &error : err_vect_iteration[i])
            {
                file << error << "\n"; // Write each component on a new line
            }
            file.close();
        }
        else
        {
            std::cerr << "Unable to open file for writing Jacobian errors.\n";
        }
    }
}

void save_errors_vector_to_file_last_iteration_gpu(std::vector<std::vector<double>> err_vect_iteration)
{
    create_directory_if_not_exists_2("./OUTPUT_RESULT/ERR_VECTOR");
    for (int i = 0; i < err_vect_iteration.size(); i++)
    {
        std::ofstream file("./OUTPUT_RESULT/ERR_VECTOR/iteration_last_gpu.txt");
        if (file.is_open())
        {
            file << err_vect_iteration[i].size() << "\n"; // Write the length of the vector in the first line
            for (const auto &error : err_vect_iteration[i])
            {
                file << error << "\n"; // Write each component on a new line
            }
            file.close();
        }
        else
        {
            std::cerr << "Unable to open file for writing Jacobian errors.\n";
        }
    }
}

#endif