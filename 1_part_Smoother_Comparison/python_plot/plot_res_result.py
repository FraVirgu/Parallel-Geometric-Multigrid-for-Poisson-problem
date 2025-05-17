import os
import matplotlib.pyplot as plt

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Correct relative path from script to OUTPUT_RESULT
output_dir = os.path.join(script_dir, "..", "OUTPUT_RESULT")

# Define plot output directory
plot_dir = os.path.join(script_dir, "..", "PLOT")
os.makedirs(plot_dir, exist_ok=True)

# Helper to read residuals
def read_residuals(file_path):
    with open(file_path, 'r') as file:
        return [float(line.strip()) for line in file]

# File paths
jacobi_file = os.path.join(output_dir, "residuals_jacobian.txt")
steepest_file = os.path.join(output_dir, "residuals_steepest_descent.txt")
gs_file = os.path.join(output_dir, "residuals_gs.txt")
cg_file = os.path.join(output_dir, "residuals_cg.txt")

# Load data
residuals_jacobi = read_residuals(jacobi_file)
residuals_steepest = read_residuals(steepest_file)
residuals_gs = read_residuals(gs_file)
residuals_cg = read_residuals(cg_file)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(residuals_jacobi, label="Jacobi")
plt.plot(residuals_steepest, label="Steepest Descent")
plt.plot(residuals_gs, label="Gauss-Seidel")
plt.plot(residuals_cg, label="Conjugate Gradient")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Residual Norm (log scale)")
plt.title("Convergence Comparison of Solvers")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Save the plot
pdf_path = os.path.join(plot_dir, "RES_convergence_plot.pdf")
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
plt.show()

print(f"Plot saved to: {pdf_path}")
