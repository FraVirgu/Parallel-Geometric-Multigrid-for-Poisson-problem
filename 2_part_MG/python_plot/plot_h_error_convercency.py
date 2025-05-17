import os
import matplotlib.pyplot as plt
import numpy as np

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Correct relative path from script to OUTPUT_RESULT
inpute_dir = os.path.join(script_dir, "..", "OUTPUT_RESULT")


# Define the directory where the plot will be saved
plot_dir = os.path.join(script_dir, "..", "PLOT")
os.makedirs(plot_dir, exist_ok=True)

# Function to read errors from a file
def read_errors(file_path):
    errors = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                problem_size = int(parts[0])  # Assuming the first column is N*N
                error = float(parts[1])
                errors.append((problem_size, error))
    return errors

# Define the methods and their corresponding files
methods_iter = {
        "Jacobi": "h_errors_jacobi.txt",
    "Gauss-Seidel": "h_errors_gs.txt",
    "Conjugate Gradient": "h_errors_cg.txt",
    "Steepest Descent": "h_errors_steepest_descent.txt"

}

methods_cycle = {
    #"f_cycle_1_cycle": "h_errors_f_cycle1",
    #"f_cycle_2_cycle": "h_errors_f_cycle2",
    #"f_cycle_3_cycle": "h_errors_f_cycle3",
    "w_cycle_1_cycle": "h_errors_w_cycle1",
    "w_cycle_2_cycle": "h_errors_w_cycle2",
    "w_cycle_3_cycle": "h_errors_w_cycle3",
}

# Dictionary to store results
results = {}

# Read errors for each method
for method, filename in methods_iter.items():
    file_path = os.path.join(inpute_dir, filename)
    if os.path.exists(file_path):
        try:
            errors = read_errors(file_path)
            if errors:
                sizes, error_values = zip(*errors)
                results[method] = (sizes, error_values)
            else:
                print(f"Warning: {method} file is empty or malformed.")
        except Exception as e:
            print(f"Error reading {method} file: {e}")
    else:
        print(f"Warning: File not found for {method}: {file_path}")

iter_cycle = {1,2,3}
for method, filename in methods_cycle.items():
   
    file_path = os.path.join(inpute_dir, f"{filename}.txt")
    if os.path.exists(file_path):
        try:
            errors = read_errors(file_path)
            if errors:
                sizes, error_values = zip(*errors)
                results[method] = (sizes, error_values)
            else:
                print(f"Warning: {method} file is empty or malformed.")
        except Exception as e:
            print(f"Error reading {method} file: {e}")
    else:
        print(f"Warning: File not found for {method}: {file_path}")

# Print summary for each method
def print_summary(method, sizes, errors):
    print(f"{method} Method:")
    print(f"  Problem Sizes: {list(sizes)}")
    print(f"  Final Error: {errors[-1]:.6e}")
    print(f"  Average Error: {np.mean(errors):.6e}")
    print("-" * 40)

print("\n=== Error Summary ===\n")
for method, (sizes, errors) in results.items():
    print_summary(method, sizes, errors)

# Plot the errors for all methods and all iterations
plt.figure(figsize=(12, 7))

for method, (sizes, errors) in results.items():
    # Detect if this is a cycle method with an iteration label (e.g., "w_cycle (k=2)")
    if "cycle" in method:
        label = method.replace("_", " ").title()
    else:
        label = method
    plt.plot(sizes, errors, marker='o', linestyle='-', label=label)

# Reference convergence lines
all_sizes = sorted(set().union(*(set(sizes) for sizes, _ in results.values())))
Ns_ref = np.array(all_sizes)
plt.plot(Ns_ref, 1 / Ns_ref**2, 'k--', label="O(N⁻²)")
plt.plot(Ns_ref, 1 / Ns_ref, 'k:', label="O(N⁻¹)")

# Styling
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Problem Size N (log scale)")
plt.ylabel("Relative L2 Error (log scale)")
plt.title("Error Convergence vs Problem Size (All Methods and Cycles)")
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, which='both', linestyle="--", linewidth=0.5)

# Save
pdf_path = os.path.join(plot_dir, "h_errors_convergence_all_methods_and_cycles.pdf")
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
plt.show()

print(f"\n✅ Combined plot saved in: {pdf_path}")
