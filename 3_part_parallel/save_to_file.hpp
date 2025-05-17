#include <vector>
#include <fstream>
#include <iostream>

#include <filesystem>
void create_directory_if_not_exists_4(const std::string &path)
{
    std::filesystem::create_directories(path);
}
void save_timing_vector_to_file(const std::string &filename, const std::vector<std::tuple<int, int, double>> &timings)
{
    std::ofstream file(filename);
    if (file.is_open())
    {
        for (const auto &timing : timings)
        {
            int num_thread, N;
            double time;
            std::tie(num_thread, N, time) = timing;
            file << num_thread << " " << N << " " << time << "\n";
        }
        file.close();
    }
    else
    {
        std::cerr << "Unable to open file: " << filename << "\n";
    }
}

// save the time for the v-cycle and w-cycle
void save_timings_to_file(std::vector<std::pair<int, double>> &timings_parallel_v_cycle, std::vector<std::pair<int, double>> &timings_parallel_w_cycle)
{
    create_directory_if_not_exists_4("OUTPUT_RESULT");

    std::ofstream file_v_cycle("OUTPUT_RESULT/timings_parallel_v_cycle.txt");
    if (file_v_cycle.is_open())
    {
        for (const auto &timing : timings_parallel_v_cycle)
        {
            file_v_cycle << timing.first << " " << timing.second << "\n";
        }
        file_v_cycle.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing V-Cycle timings.\n";
    }

    std::ofstream file_w_cycle("OUTPUT_RESULT/timings_parallel_w_cycle.txt");
    if (file_w_cycle.is_open())
    {
        for (const auto &timing : timings_parallel_w_cycle)
        {
            file_w_cycle << timing.first << " " << timing.second << "\n";
        }
        file_w_cycle.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing W-Cycle timings.\n";
    }
}

void save_timings_to_file_all_methods(
    const std::vector<std::tuple<int, int, double>> time_residual_cpu,
    const std::vector<std::tuple<int, int, double>> time_residual_gpu,
    const std::vector<std::tuple<int, int, double>> time_jacobi_cpu,
    const std::vector<std::tuple<int, int, double>> time_jacobi_gpu,
    const std::vector<std::tuple<int, int, double>> time_restriction_cpu,
    const std::vector<std::tuple<int, int, double>> time_restriction_gpu,
    const std::vector<std::tuple<int, int, double>> time_prolungator_cpu,
    const std::vector<std::tuple<int, int, double>> time_prolungator_gpu

)
{
    create_directory_if_not_exists_4("OUTPUT_RESULT");

    save_timing_vector_to_file("OUTPUT_RESULT/timings_residual_cpu.txt", time_residual_cpu);
    save_timing_vector_to_file("OUTPUT_RESULT/timings_residual_gpu.txt", time_residual_gpu);

    save_timing_vector_to_file("OUTPUT_RESULT/timings_jacobi_cpu.txt", time_jacobi_cpu);
    save_timing_vector_to_file("OUTPUT_RESULT/timings_jacobi_gpu.txt", time_jacobi_gpu);

    save_timing_vector_to_file("OUTPUT_RESULT/timings_restriction_cpu.txt", time_restriction_cpu);
    save_timing_vector_to_file("OUTPUT_RESULT/timings_restriction_gpu.txt", time_restriction_gpu);

    save_timing_vector_to_file("OUTPUT_RESULT/timings_prolungator_cpu.txt", time_prolungator_cpu);
    save_timing_vector_to_file("OUTPUT_RESULT/timings_prolungator_gpu.txt", time_prolungator_gpu);
}