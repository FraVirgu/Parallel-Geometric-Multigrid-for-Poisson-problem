import os
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Setup Paths
# ---------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "..", "OUTPUT_RESULT")
plot_dir = os.path.join(script_dir, "..", "PLOT")
os.makedirs(plot_dir, exist_ok=True)

# ---------------------------
# Data Reading Function
# ---------------------------

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

def compute_speedup(serial_data, parallel_data):
    serial_dict = dict(serial_data)
    parallel_dict = dict(parallel_data)
    common_sizes = sorted(set(serial_dict.keys()) & set(parallel_dict.keys()))
    speedups = [(N, serial_dict[N] / parallel_dict[N]) for N in common_sizes]
    return zip(*speedups)

# ---------------------------
# Read Files
# ---------------------------

timings_v = read_timings(os.path.join(input_dir, "timings_v_cycle.txt"))
timings_w = read_timings(os.path.join(input_dir, "timings_w_cycle.txt"))
timings_parallel_v = read_timings(os.path.join(input_dir, "timings_parallel_v_cycle.txt"))
timings_parallel_w = read_timings(os.path.join(input_dir, "timings_parallel_w_cycle.txt"))

# ---------------------------
# Unpack and Convert to Arrays
# ---------------------------

sizes_v, times_v = zip(*timings_v)
sizes_w, times_w = zip(*timings_w)
sizes_parallel_v, times_parallel_v = zip(*timings_parallel_v)
sizes_parallel_w, times_parallel_w = zip(*timings_parallel_w)

sizes_v = np.array(sizes_v)
times_v = np.array(times_v)
times_w = np.array(times_w)
times_parallel_v = np.array(times_parallel_v)
times_parallel_w = np.array(times_parallel_w)

# ---------------------------
# Compute Speedups
# ---------------------------

sizes_v_speedup, speedups_v = compute_speedup(timings_v, timings_parallel_v)
sizes_w_speedup, speedups_w = compute_speedup(timings_w, timings_parallel_w)
sizes_speedup = np.array(sizes_v_speedup)
speedups_v = np.array(speedups_v)
speedups_w = np.array(speedups_w)

# ---------------------------
# Plot Combined: Timing + Speedup
# ---------------------------

fig, ax1 = plt.subplots(figsize=(11, 6))

# Primary Y-axis: Execution Time
ax1.set_xlabel("Problem Size (N)")
ax1.set_ylabel("Time (seconds)", color='black')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.plot(sizes_v, times_v, label="V-Cycle", marker='o')
ax1.plot(sizes_v, times_w, label="W-Cycle", marker='s')
ax1.plot(sizes_parallel_v, times_parallel_v, label="Parallel V-Cycle", marker='x')
ax1.plot(sizes_parallel_w, times_parallel_w, label="Parallel W-Cycle", marker='d')
ax1.plot(sizes_v, sizes_v, label="O(N)", linestyle="--", color="black")
ax1.plot(sizes_v, sizes_v**2 / 1e5, label="O(N²)", linestyle="--", color="green")
ax1.tick_params(axis='y', labelcolor='black')

ax1.text(
    0.98, 0.02,
    "mg_iteration = 3\nalpha = 1\nsmoother_iteration = 1",
    transform=ax1.transAxes,
    fontsize=10,
    verticalalignment='bottom',
    horizontalalignment='right',
    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3')
)



# Secondary Y-axis: Speedup
ax2 = ax1.twinx()
ax2.set_ylabel("Speedup (Serial / Parallel)", color='blue')
ax2.plot(sizes_speedup, speedups_v, label="V-Cycle Speedup", marker='^', color='blue', linestyle='--')
ax2.plot(sizes_speedup, speedups_w, label="W-Cycle Speedup", marker='v', color='navy', linestyle='--')
ax2.tick_params(axis='y', labelcolor='blue')

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

# Title and grid
plt.title("Multigrid Timing and Parallel Speedup vs Problem Size (Log-Log Time)")
ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

# Save and show
combined_path = os.path.join(plot_dir, "timings_and_speedup_multigrid.pdf")
plt.savefig(combined_path, format="pdf", bbox_inches="tight")
plt.show()

print(f"Combined timing and speedup plot saved in {combined_path}")
