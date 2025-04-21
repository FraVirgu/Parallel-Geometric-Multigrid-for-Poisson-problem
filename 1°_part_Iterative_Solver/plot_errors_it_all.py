import os
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_error_comparison_per_N():
    # Setup
    plot_dir = "PLOT"
    data_dir = "OUTPUT_RESULT"
    os.makedirs(plot_dir, exist_ok=True)

    # Mapping method names to filenames
    method_files = {
        "Jacobi": "errors_it_j.txt",
        "Gauss-Seidel": "errors_it_gs.txt",
        "Steepest Descent": "errors_it_steepest.txt",
        "Conjugate Gradient": "errors_it_cg.txt"
    }

    # Read error curves from a file
    def read_error_curves(file_path):
        curves = defaultdict(list)
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    N = int(parts[0])
                    iteration = int(parts[1])
                    error = float(parts[2])
                    curves[N].append((iteration, error))
        return curves

    # Read all methods into a dict of dicts: method -> N -> (iteration, error)
    all_curves = {}
    Ns_union = set()
    for method, filename in method_files.items():
        path = os.path.join(data_dir, filename)
        if os.path.isfile(path):
            curves = read_error_curves(path)
            all_curves[method] = curves
            Ns_union.update(curves.keys())
        else:
            print(f"Warning: {filename} not found in {data_dir}")

    # Plot per N
    for N in sorted(Ns_union):
        # --- Linear plot ---
        plt.figure(figsize=(10, 5))
        for method, curves in all_curves.items():
            if N in curves:
                iterations, errors = zip(*curves[N])
                plt.plot(iterations, errors, label=method)
        # Add a reference curve, skipping i=0 to avoid division by zero
        ref_iterations = [i for i in iterations if i != 0]
        ref_values = [1 / (i * i) for i in ref_iterations]
        plt.plot(ref_iterations, ref_values, label="1/i² (reference)", linestyle='--', color='gray')

        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.title(f"Error vs Iteration (N={N})")
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"comparison_N{N}_linear.pdf"))
        plt.close()

        # --- Log-log plot ---
        plt.figure(figsize=(10, 5))
        for method, curves in all_curves.items():
            if N in curves:
                iterations, errors = zip(*curves[N])
                iterations_log = [i if i > 0 else 1 for i in iterations]
                errors_log = [e if e > 0 else 1e-16 for e in errors]
                plt.plot(iterations_log, errors_log, label=method)
        plt.plot(iterations_log, [1/((i*i)+1) for i in iterations_log], label="1/N^2", linestyle='--', color='gray')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Iteration (log)")
        plt.ylabel("Error (log)")
        plt.title(f"Log-Log Error vs Iteration (N={N})")
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"comparison_N{N}_loglog.pdf"))
        plt.close()


plot_error_comparison_per_N()
