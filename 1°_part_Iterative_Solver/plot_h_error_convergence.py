import os
import matplotlib.pyplot as plt

# Define the directory where the plot will be saved
plot_dir = "PLOT"
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
methods = {
    "Jacobi": "h_errors_jacobi.txt",
    "Gauss-Seidel": "h_errors_gs.txt",
    "Conjugate Gradient": "h_errors_cg.txt",
    "Steepest Descent": "h_errors_steepest_descent.txt"
}

# Dictionary to store results
results = {}

# Read errors for each method
for method, filename in methods.items():
    file_path = os.path.join("OUTPUT_RESULT", filename)
    if os.path.exists(file_path):
        errors = read_errors(file_path)
        if errors:
            sizes, error_values = zip(*errors)
            results[method] = (sizes, error_values)
        else:
            print(f"Warning: {method} file is empty or malformed.")
    else:
        print(f"Warning: File not found for {method}: {file_path}")

# Print summary for each method
def print_summary(method, sizes, errors):
    print(f"{method} Method:")
    print(f"  Largest Problem Size: {sizes[-1]}")
    print(f"  Final Error: {errors[-1]:.6f}")
    print(f"  Average Error per Problem Size: {sum(errors) / len(errors):.6f}\n")

for method, (sizes, errors) in results.items():
    print_summary(method, sizes, errors)

# Plot the errors for all methods
plt.figure(figsize=(10, 6))
for method, (sizes, errors) in results.items():
    plt.plot(sizes, errors, label=method)

plt.yscale("log")
plt.xlabel("Problem Size (N)")
plt.ylabel("Error Norm (log scale)")
plt.title("Error Convergence vs Problem Size for All Methods")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)

# Save the plot as a PDF
pdf_path = os.path.join(plot_dir, "h_errors_convergence_all_methods.pdf")
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")

# Show the plot
plt.show()

print(f"Plot saved successfully in {pdf_path}")
