import os
import matplotlib.pyplot as plt
import numpy as np

# Define the directory where the plot will be saved
plot_dir = "PLOT"
os.makedirs(plot_dir, exist_ok=True)


# Function to read timings from a file
def read_timings(file_path):
    timings = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                problem_size = int(parts[0])  # Assuming the first column is N*N
                time = float(parts[1])
                timings.append((problem_size, time))
    return timings

# File paths
output_dir = "OUTPUT_RESULT"
timings_jacobi_file = os.path.join(output_dir, "timings_jacobian.txt")
timings_steepest_file = os.path.join(output_dir, "timings_steepest_descent.txt")
timings_gs_file = os.path.join(output_dir, "timings_gs.txt")
timings_cg_file = os.path.join(output_dir, "timings_cg.txt")

# Read timings
timings_jacobi = read_timings(timings_jacobi_file)
timings_steepest = read_timings(timings_steepest_file)
timings_gs = read_timings(timings_gs_file)
timings_cg = read_timings(timings_cg_file)

# Extract problem size and time values
sizes_jacobi, times_jacobi = zip(*timings_jacobi)
sizes_steepest, times_steepest = zip(*timings_steepest)
sizes_gs, times_gs = zip(*timings_gs)
sizes_cg, times_cg = zip(*timings_cg)

sizes_cg = np.array(sizes_cg)
square = sizes_cg ** 2 / 100000
cubic = sizes_cg ** 3 / 1000000

# Print results summary
def print_summary(method, sizes, times):
    print(f"{method} Method:")
    print(f"  Largest Problem Size: {sizes[-1]}")
    print(f"  Total Time: {times[-1]:.6f} seconds")
    print(f"  Average Time per Problem Size: {sum(times) / len(times):.6f} seconds\n")

print_summary("Jacobi", sizes_jacobi, times_jacobi)
print_summary("Steepest Descent", sizes_steepest, times_steepest)
print_summary("Gauss-Seidel", sizes_gs, times_gs)
print_summary("Conjugate Gradient", sizes_cg, times_cg)


# Plot the timings with log scale
plt.figure(figsize=(10, 6))
plt.plot(sizes_jacobi, times_jacobi, label="Jacobi", marker='o')
plt.plot(sizes_steepest, times_steepest, label="Steepest Descent", marker='s')
plt.plot(sizes_gs, times_gs, label="Gauss-Seidel", marker='^')
plt.plot(sizes_cg, times_cg, label="Conjugate Gradient", linestyle="--", marker='d')
plt.plot(sizes_cg, square, label="O(N^2)", linestyle="--", color="black")
plt.plot(sizes_cg, cubic, label="O(N^3)", linestyle="--", color="gray")

plt.xscale('log')  # Log scale for problem size
plt.yscale('log')  # Log scale for time

plt.xlabel("Problem Size (N)")
plt.ylabel("Time (seconds)")
plt.title("Computation Time vs Problem Size for Iterative Methods (Log-Log Scale)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Save the plot as a PDF in the "PLOT" folder
pdf_path = os.path.join(plot_dir, "timings_convergence_plot_log.pdf")
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")

# Show the plot
plt.show()

print(f"Log-log plot saved successfully in {pdf_path}")
print(f"Plot saved successfully in {pdf_path}")
