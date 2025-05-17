import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Setup directories
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "..", "OUTPUT_RESULT")
plot_dir = os.path.join(script_dir, "..", "PLOT")
os.makedirs(plot_dir, exist_ok=True)

# Read (thread_count, N, time)
def read_timings(file_path):
    timings = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                thread_count = int(parts[0])
                problem_size = int(parts[1])
                time = float(parts[2])
                timings.append((thread_count, problem_size, time))
    return timings

# Group timings by thread count
def group_timings_by_thread(timings):
    grouped = defaultdict(list)
    for thread, N, t in timings:
        grouped[thread].append((N, t))
    return grouped

# Timing files and labels
timing_files = [
    ("timings_residual_cpu.txt", "Residual - CPU"),
    ("timings_residual_gpu.txt", "Residual - GPU"),
    ("timings_jacobi_cpu.txt", "Jacobi - CPU"),
    ("timings_jacobi_gpu.txt", "Jacobi - GPU"),
    ("timings_restriction_cpu.txt", "Restriction - CPU"),
    ("timings_restriction_gpu.txt", "Restriction - GPU"),
    ("timings_prolungator_cpu.txt", "Prolongator - CPU"),
    ("timings_prolungator_gpu.txt", "Prolongator - GPU")
]

# Define visual styles
colors = ['blue', 'blue', 'orange', 'orange', 'green', 'green', 'red', 'red']
markers = ['o', 'x', 'o', 'x', 'o', 'x', 'o', 'x']

# Initialize figure
plt.figure(figsize=(12, 8))

for i, (filename, base_label) in enumerate(timing_files):
    file_path = os.path.join(input_dir, filename)
    timings = read_timings(file_path)
    grouped = group_timings_by_thread(timings)

    # Add iteration count to Jacobi labels
    if "Jacobi" in base_label:
        base_label += " (100 iters)"

    cpu_plotted = False

    for thread_count, values in sorted(grouped.items()):
        values.sort()
        Ns, Ts = zip(*values)

        if "CPU" in base_label:
            if cpu_plotted:
                continue
            label = base_label
            cpu_plotted = True
        else:
            label = f"{base_label} ({thread_count}^2 threads)"

        plt.plot(Ns, Ts, label=label, marker=markers[i % len(markers)], color=colors[i % len(colors)])

# Reference complexity lines
all_Ns = sorted({N for fname, _ in timing_files for _, N, _ in read_timings(os.path.join(input_dir, fname))})
Ns = np.array(all_Ns)
plt.plot(Ns, Ns, linestyle="--", color="black", label="O(N)")
plt.plot(Ns, (Ns ** 2) / 1e5, linestyle="--", color="gray", label="O(N²)")

# Axis and formatting
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Problem Size (N)")
plt.ylabel("Time (seconds)")
plt.title("Timing Comparison: All Multigrid Operations (Log-Log)")
plt.legend(fontsize=8)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Save and show
output_path = os.path.join(plot_dir, "all_timings_combined_loglog.pdf")
plt.savefig(output_path, format="pdf", bbox_inches="tight")
plt.show()
print(f"Combined log-log plot saved to: {output_path}")
