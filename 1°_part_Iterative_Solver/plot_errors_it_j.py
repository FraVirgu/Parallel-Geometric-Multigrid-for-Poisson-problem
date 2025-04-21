import os
import matplotlib.pyplot as plt
from collections import defaultdict

# Directory and file setup
plot_dir = "PLOT"
os.makedirs(plot_dir, exist_ok=True)

# File containing errors with format: N  iteration  error
error_file = os.path.join("OUTPUT_RESULT", "errors_it_j.txt")

def read_error_curves(file_path):
    curves = defaultdict(list)  # Dictionary mapping N -> list of (iteration, error)
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                N = int(parts[0])
                iteration = int(parts[1])
                error = float(parts[2])
                curves[N].append((iteration, error))
    return curves

def plot_linear(curves, output_path):
    plt.figure(figsize=(12, 6))
    for N in sorted(curves.keys()):
        iterations, errors = zip(*curves[N])
        plt.plot(iterations, errors, label=f"N={N}")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.title("Error Convergence (Linear Scale) per Iteration for Jacobi Method")
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.show()
    print(f"Linear plot saved to {output_path}")

def plot_loglog(curves, output_path):
    plt.figure(figsize=(12, 6))
    for N in sorted(curves.keys()):
        iterations, errors = zip(*curves[N])
        # Avoid zero values since log(0) is undefined
        iterations_log = [i if i > 0 else 1 for i in iterations]
        errors_log = [e if e > 0 else 1e-16 for e in errors]
        plt.plot(iterations_log, errors_log, label=f"N={N}")
    plt.xlabel("Iteration (log scale)")
    plt.ylabel("Error (log scale)")
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Error Convergence (Log-Log Scale) per Iteration for Jacobi Method")
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.show()
    print(f"Log-log plot saved to {output_path}")

# Load and plot
error_curves = read_error_curves(error_file)

# Linear plot
linear_path = os.path.join(plot_dir, "jacobi_error_curves_linear.pdf")
plot_linear(error_curves, linear_path)

# Log-log plot
loglog_path = os.path.join(plot_dir, "jacobi_error_curves_loglog.pdf")
plot_loglog(error_curves, loglog_path)
