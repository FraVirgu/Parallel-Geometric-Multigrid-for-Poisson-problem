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
                time_ms = float(parts[1])  # Time in milliseconds
                time_s = time_ms / 1000.0  # Convert to seconds
                timings.append((problem_size, time_s))
    return timings

# File paths
output_dir = "OUTPUT_RESULT"
timings_cg_file = os.path.join(output_dir, "timings_cg.txt")
timings_mg_file = os.path.join(output_dir, "timings_MG.txt")
timings_fmg_file = os.path.join(output_dir, "timings_FMG.txt")

# Read timings
timings_cg = read_timings(timings_cg_file)
timings_mg = read_timings(timings_mg_file)
timings_fmg = read_timings(timings_fmg_file)

# Extract problem size and time values
sizes_cg, times_cg = zip(*timings_cg)
sizes_mg, times_mg = zip(*timings_mg)
sizes_fmg, times_fmg = zip(*timings_fmg)

sizes_cg = np.array(sizes_cg)
square = sizes_cg ** 2 / 100000
cubic = sizes_cg ** 3 / 1000000

# Print results summary
def print_summary(method, sizes, times):
    print(f"{method} Method:")
    print(f"  Largest Problem Size: {sizes[-1]}")
    print(f"  Total Time: {times[-1]:.6f} seconds")
    print(f"  Average Time per Problem Size: {sum(times) / len(times):.6f} seconds\n")

print_summary("Conjugate Gradient", sizes_cg, times_cg)
print_summary("Multigrid", sizes_mg, times_mg)
print_summary("Full Multigrid", sizes_fmg, times_fmg)

# Plot the timings with log scale
plt.figure(figsize=(10, 6))
plt.plot(sizes_cg, times_cg, label="Conjugate Gradient", linestyle="--", marker='d')
plt.plot(sizes_mg, times_mg, label="Multigrid (v1=v2=5)", marker='o')
plt.plot(sizes_fmg, times_fmg, label="Full Multigrid (v1=v2=5)", marker='s')

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
