import os
import matplotlib.pyplot as plt
import numpy as np

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Correct relative path from script to OUTPUT_RESULT
input_dir = os.path.join(script_dir, "..", "OUTPUT_RESULT")


# Define the directory where the plot will be saved
plot_dir = os.path.join(script_dir, "..", "PLOT")
os.makedirs(plot_dir, exist_ok=True)

# Read (N, time) pairs from a timing file
def read_timings(file_path):
    timings = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                problem_size = int(parts[0])
                time = float(parts[1])
                timings.append((problem_size, time))
    return timings

# File paths
timings_jacobi_file = os.path.join(input_dir, "timings_jacobian.txt")
timings_steepest_file = os.path.join(input_dir, "timings_steepest_descent.txt")
timings_gs_file = os.path.join(input_dir, "timings_gs.txt")
timings_cg_file = os.path.join(input_dir, "timings_cg.txt")

# Read data
timings_jacobi = read_timings(timings_jacobi_file)
timings_steepest = read_timings(timings_steepest_file)
timings_gs = read_timings(timings_gs_file)
timings_cg = read_timings(timings_cg_file)

# Unpack into size/time arrays
sizes_jacobi, times_jacobi = zip(*timings_jacobi)
sizes_steepest, times_steepest = zip(*timings_steepest)
sizes_gs, times_gs = zip(*timings_gs)
sizes_cg, times_cg = zip(*timings_cg)

sizes = np.array(sizes_cg)
square = sizes ** 2 / 100000
cubic = sizes ** 3 / 1000000

# Print summary info
def print_summary(method, sizes, times):
    print(f"{method} Method:")
    print(f"  Largest Problem Size: {sizes[-1]}")
    print(f"  Total Time: {times[-1]:.6f} seconds")
    print(f"  Average Time per Problem Size: {sum(times) / len(times):.6f} seconds\n")

print_summary("Jacobi", sizes_jacobi, times_jacobi)
print_summary("Steepest Descent", sizes_steepest, times_steepest)
print_summary("Gauss-Seidel", sizes_gs, times_gs)
print_summary("Conjugate Gradient", sizes_cg, times_cg)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(sizes_jacobi, times_jacobi, label="Jacobi", marker='o')
plt.plot(sizes_steepest, times_steepest, label="Steepest Descent", marker='s')
plt.plot(sizes_gs, times_gs, label="Gauss-Seidel", marker='^')
plt.plot(sizes_cg, times_cg, label="Conjugate Gradient", marker='d', linestyle='--')
plt.plot(sizes, square, label="O(N²)", linestyle="--", color="black")
plt.plot(sizes, cubic, label="O(N³)", linestyle="--", color="gray")

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Problem Size (N)")
plt.ylabel("Time (seconds)")
plt.title("Computation Time vs Problem Size for Iterative Methods (Log-Log Scale)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Save plot
pdf_path = os.path.join(plot_dir, "timings_convergence_plot_log.pdf")
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
plt.show()

print(f"Log-log plot saved successfully in {pdf_path}")
