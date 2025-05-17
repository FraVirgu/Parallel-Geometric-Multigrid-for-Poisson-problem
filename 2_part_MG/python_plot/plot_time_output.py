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
timings_v_file = os.path.join(input_dir, "timings_v_cycle.txt")
timings_w_file = os.path.join(input_dir, "timings_w_cycle.txt")
timings_f_file = os.path.join(input_dir, "timings_f_cycle.txt")

# Read data
timings_v = read_timings(timings_v_file)
timings_w = read_timings(timings_w_file)
timings_f = read_timings(timings_f_file)


# Unpack into size/time arrays
sizes_v, times_v = zip(*timings_v)
sizes_w, times_w = zip(*timings_w)
sizes_f, times_f = zip(*timings_f)

sizes = np.array(sizes_f)
square = sizes ** 2 / 100000
cubic = sizes ** 3 / 1000000
# Print summary info
def print_summary(method, sizes, times):
    print(f"{method} Method:")
    print(f"  Largest Problem Size: {sizes[-1]}")
    print(f"  Total Time: {times[-1]:.6f} seconds")
    print(f"  Average Time per Problem Size: {sum(times) / len(times):.6f} seconds\n")

print_summary("V-Cycle", sizes_v, times_v)
print_summary("W-Cycle", sizes_w, times_w)
print_summary("F-Cycle", sizes_f, times_f)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(sizes_v, times_v, label="V-Cycle", marker='o')
plt.plot(sizes_w, times_w, label="W-Cycle", marker='s')
plt.plot(sizes_f, times_f, label="F-Cycle", marker='^')
plt.plot(sizes, sizes, label="O(N)", linestyle="--", color="black")
plt.plot(sizes, square, label="O(N²)", linestyle="--", color="green")

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Problem Size (N)")
plt.ylabel("Time (seconds)")
plt.title("Computation Time vs Problem Size for Multigrid Cycles (Log-Log Scale)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Save plot
pdf_path = os.path.join(plot_dir, "timings_multigrid_cycles_log.pdf")
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
plt.show()

print(f"Log-log plot saved successfully in {pdf_path}")
