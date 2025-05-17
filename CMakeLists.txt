cmake_minimum_required(VERSION 3.10)
project(GeometricMultigrid_Project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Global compiler warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/1_part_Smoother_Comparison
    ${PROJECT_SOURCE_DIR}/2_part_MG
    ${PROJECT_SOURCE_DIR}/3_part_parallel
)

# Executable 1: CPU iterative
add_executable(cpu_iterative_exec
    1_part_Smoother_Comparison/main.cpp
    globals.cpp
)

# Executable 2: Multigrid CPU
add_executable(mg_cpu_exec
    2_part_MG/main.cpp
    globals.cpp
)

# Executable 3: GPU parallel
add_executable(gpu_exec
    3_part_parallel/main.cu
   
    globals.cpp
)

# Set CUDA architecture if needed (adjust if your GPU is newer)
set_target_properties(gpu_exec PROPERTIES
    CUDA_ARCHITECTURES "50"
)
