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

# File path
errors_jacobi_file = os.path.join("OUTPUT_RESULT", "h_errors_jacobian.txt")

# Read errors
errors_jacobi = read_errors(errors_jacobi_file)

# Extract problem size and error values
sizes_jacobi, errors_jacobi_values = zip(*errors_jacobi)

# Print results summary
def print_summary(method, sizes, errors):
    print(f"{method} Method:")
    print(f"  Largest Problem Size: {sizes[-1]}")
    print(f"  Final Error: {errors[-1]:.6f}")
    print(f"  Average Error per Problem Size: {sum(errors) / len(errors):.6f}\n")

print_summary("Jacobi", sizes_jacobi, errors_jacobi_values)

# Plot the errors
plt.figure(figsize=(10, 6))
plt.plot(sizes_jacobi, errors_jacobi_values, label="Jacobi")
plt.yscale("log")
plt.xlabel("Problem Size (N)")
plt.ylabel("Error Norm (log scale)")
plt.title("Error Convergence vs Problem Size for Jacobi Method")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)

# Save the plot as a PDF in the "PLOT" folder
pdf_path = os.path.join(plot_dir, "h_errors_convergence_plot_jacobi.pdf")
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")

# Show the plot
plt.show()

print(f"Plot saved successfully in {pdf_path}")
