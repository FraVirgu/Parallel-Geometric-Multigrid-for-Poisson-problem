# Parallel Geometric Multigrid for Poisson Problems

This project implements and analyzes multigrid methods for solving the 2D Poisson equation using both sequential and parallelized algorithms (CPU and CUDA GPU). The report includes mathematical foundations, convergence analyses, implementation details, and benchmarking results.

## Project Overview

The solver handles the Poisson problem with homogeneous Dirichlet boundary conditions on a square domain. It includes:

- Classical stationary iterative methods (Jacobi, Gauss-Seidel, etc.)
- Krylov methods (Conjugate Gradient, Steepest Descent)
- Multigrid methods (V-cycle, W-cycle, Full Multigrid)
- CUDA-accelerated parallel versions of:
  - Jacobi smoothing
  - Restriction
  - Prolongation
  - Residual calculation

## Folder Structure

.
├── 1_part_Smoother_Comparison       # Sequential solvers
│   └── main.cpp
├── 2_part_MG                        # Multigrid CPU implementation
│   └── main.cpp
├── 3_part_parallel                  # CUDA (GPU) implementation
│   ├── main.cu
│   ├── Parallel_Method.cu
│   ├── Parallel_Mg.cu
│   └── ParallelTestRunner.cu
├── globals.cpp, globals.hpp        # Shared configuration and utility
├── CMakeLists.txt                  # Build configuration
├── build/                          # Created after compilation
└── README.md

## Build Instructions

This project uses CMake for cross-platform configuration and building.

### Requirements

- A C++17-compatible compiler (e.g., g++)
- CUDA Toolkit (e.g., version 12.0+)
- CMake 3.10 or newer

### Build Process

Open a terminal in the project root directory and run:

    mkdir build
    cd build
    cmake ..
    make

This will compile three executables:

- cpu_iterative_exec  — classic iterative solvers (Part 1)
- mg_cpu_exec         — multigrid method (Part 2)
- gpu_exec            — parallel multigrid with CUDA (Part 3)

## Running the Executables

After building, from the `build/` directory, run:

    ./cpu_iterative_exec
    ./mg_cpu_exec
    ./gpu_exec

Each executable may take parameters like grid size or number of iterations, depending on your implementation.

## Report

The LaTeX project report includes:

- Mathematical derivation
- Convergence and error analysis
- Details on Multigrid cycles (V/W/F)
- CUDA parallel strategies
- Full performance and timing comparisons

## System Configuration (used for testing)

- CPU: Intel Core i9-14900HX, 24 cores
- GPU: NVIDIA RTX 4070 Laptop GPU (8 GB VRAM)
- CUDA Version: 12.9
- C++ Standard: C++17
- OS: Linux
- Build System: CMake 3.10+

## Authors

- Francesco Virgulti  
- Laura Gioanna Paxton  
- Supervisor: Emile Parolin

## License

This project is for academic and educational use.
