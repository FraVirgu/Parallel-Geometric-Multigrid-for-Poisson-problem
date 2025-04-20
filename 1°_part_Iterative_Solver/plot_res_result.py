import os
import matplotlib.pyplot as plt

n_1 = 64
n_2 = n_1 * 2
# Define the directory where the plot will be saved
plot_dir = "PLOT"
os.makedirs(plot_dir, exist_ok=True)

# Read residuals from files
def read_residuals(file_path):
    with open(file_path, 'r') as file:
        residuals = [float(line.strip()) for line in file]
    return residuals

# File paths
output_dir = "OUTPUT_RESULT"
jacobi_file_n1 = os.path.join(output_dir, f"residuals_jacobian_{n_1}.txt")
steepest_file_n1 = os.path.join(output_dir, f"residuals_steepest_descent_{n_1}.txt")
gs_file_n1 = os.path.join(output_dir, f"residuals_gs_{n_1}.txt")
cg_file_n1 = os.path.join(output_dir, f"residuals_cg_{n_1}.txt")

jacobi_file_n2 = os.path.join(output_dir, f"residuals_jacobian_{n_2}.txt")
steepest_file_n2 = os.path.join(output_dir, f"residuals_steepest_descent_{n_2}.txt")
gs_file_n2 = os.path.join(output_dir, f"residuals_gs_{n_2}.txt")
cg_file_n2 = os.path.join(output_dir, f"residuals_cg_{n_2}.txt")
# Read residuals
residuals_jacobi_n1 = read_residuals(jacobi_file_n1)
residuals_steepest_n1 = read_residuals(steepest_file_n1)
residuals_gs_n1 = read_residuals(gs_file_n1)
residuals_cg_n1 = read_residuals(cg_file_n1)

residuals_jacobi_n2 = read_residuals(jacobi_file_n2)
residuals_steepest_n2 = read_residuals(steepest_file_n2)
residuals_gs_n2 = read_residuals(gs_file_n2)
residuals_cg_n2 = read_residuals(cg_file_n2)

# Plot residuals
plt.figure(figsize=(10, 6))
plt.plot(residuals_jacobi_n1, label=f"Jacobi (n1 = {n_1})", color="blue")
plt.plot(residuals_jacobi_n2, label=f"Jacobi (n2 = {n_2})", color="dodgerblue")

plt.plot(residuals_gs_n1, label=f"Gauss-Seidel (n1 = {n_1})", color="green")
plt.plot(residuals_gs_n2, label=f"Gauss-Seidel (n2 = {n_2})", color="lime")

#plt.plot(residuals_steepest_n1, label=f"Steepest Descent (n1 = {n_1})", color="orange")
#plt.plot(residuals_steepest_n2, label=f"Steepest Descent (n2 = {n_2})", color="gold")

#plt.plot(residuals_cg_n1, label=f"Conjugate Gradient (n1 = {n_1})", color="red")
#plt.plot(residuals_cg_n2, label=f"Conjugate Gradient (n2 = {n_2})", color="darkred")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Residual Norm (log scale)")
plt.title("Convergence of Jacobi vs Steepest Descent vs Gauss-Seidel vs Conjugate Gradient")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Save the plot as a PDF in the "PLOT" folder
pdf_path = os.path.join(plot_dir, "RES_convergence_plot.pdf")
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")

# Show the plot
plt.show()

print(f"Plot saved successfully in {pdf_path}")