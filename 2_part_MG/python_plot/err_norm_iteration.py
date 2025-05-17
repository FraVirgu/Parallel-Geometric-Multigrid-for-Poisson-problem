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


v_file_1_cycle = os.path.join(output_dir, "error_v_cycle1.txt")
#w_file_2_cycle = os.path.join(output_dir, "error_v_cycle2.txt")
#v_file_3_cycle = os.path.join(output_dir, "error_v_cycle3.txt")
w_file_1_cycle = os.path.join(output_dir, "error_w_cycle1.txt")
#w_file_2_cycle = os.path.join(output_dir, "error_w_cycle2.txt")
#w_file_3_cycle = os.path.join(output_dir, "error_w_cycle3.txt")

f_file_1_cycle = os.path.join(output_dir, "error_f_cycle1.txt")
#f_file_2_cycle = os.path.join(output_dir, "error_f_cycle2.txt")
#f_file_3_cycle = os.path.join(output_dir, "error_f_cycle3.txt")

# Load data
errors_v_1_cycle = read_errors(v_file_1_cycle)
#errors_v_2_cycle = read_errors(v_file_2_cycle)
#errors_v_3_cycle = read_errors(v_file_3_cycle)

errors_w_1_cycle = read_errors(w_file_1_cycle)
#errors_w_2_cycle = read_errors(w_file_2_cycle)
#errors_w_3_cycle = read_errors(w_file_3_cycle)

errors_f_1_cycle = read_errors(f_file_1_cycle)
#errors_f_2_cycle = read_errors(f_file_2_cycle)
#errors_f_3_cycle = read_errors(f_file_3_cycle)

# Plot errors
plt.figure(figsize=(10, 6))

plt.plot(errors_v_1_cycle, label="V-Cycle 1", linestyle="-.")
#plt.plot(errors_v_2_cycle, label="V-Cycle 2", linestyle="-.")
#plt.plot(errors_v_3_cycle, label="V-Cycle 3", linestyle="-.")
plt.plot(errors_w_1_cycle, label="W-Cycle 1", linestyle=":")
#plt.plot(errors_w_2_cycle, label="W-Cycle 2", linestyle=":")
#plt.plot(errors_w_3_cycle, label="W-Cycle 3", linestyle=":")
plt.plot(errors_f_1_cycle, label="F-Cycle 1", linestyle="-")
#plt.plot(errors_f_2_cycle, label="F-Cycle 2", linestyle="-")
#plt.plot(errors_f_3_cycle, label="F-Cycle 3", linestyle="-")
plt.yscale("log")
plt.xlabel("Cycle")
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.ylabel("Relative Error (log scale)")
plt.title("Error Convergence of Iterative Methods")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Save the plot
pdf_path = os.path.join(plot_dir, "ERR_convergence_plot.pdf")
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
plt.show()

print(f"Plot saved successfully in {pdf_path}")
