import os
import matplotlib.pyplot as plt

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the directory where the plot will be saved
plot_dir = os.path.join(script_dir, "..", "PLOT")
os.makedirs(plot_dir, exist_ok=True)

# Define the input directory containing error files
output_dir = os.path.join(script_dir, "..", "OUTPUT_RESULT")

# Read error data from files
def read_errors(file_path):
    with open(file_path, 'r') as file:
        return [float(line.strip()) for line in file]

# File paths for error data
jacobi_file = os.path.join(output_dir, "error_jacobian.txt")
steepest_file = os.path.join(output_dir, "error_steepest_descent.txt")
gs_file = os.path.join(output_dir, "error_gs.txt")
cg_file = os.path.join(output_dir, "error_cg.txt")

# Load data
errors_jacobi = read_errors(jacobi_file)
errors_steepest = read_errors(steepest_file)
errors_gs = read_errors(gs_file)
errors_cg = read_errors(cg_file)

# Plot errors
plt.figure(figsize=(10, 6))
plt.plot(errors_jacobi, label="Jacobi")
plt.plot(errors_steepest, label="Steepest Descent")
plt.plot(errors_gs, label="Gauss-Seidel")
plt.plot(errors_cg, label="Conjugate Gradient", linestyle="--")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Relative Error (log scale)")
plt.title("Error Convergence of Iterative Methods")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Save the plot
pdf_path = os.path.join(plot_dir, "ERR_convergence_plot.pdf")
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
plt.show()

print(f"Plot saved successfully in {pdf_path}")
