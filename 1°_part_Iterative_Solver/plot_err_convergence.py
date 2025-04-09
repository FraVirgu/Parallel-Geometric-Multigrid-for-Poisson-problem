import os
import numpy as np
import matplotlib.pyplot as plt

# Define the directory and filename
output_dir = "OUTPUT_RESULT"
filename = "jacobi_convergence.txt"

# Full path to the file
filepath = os.path.join(output_dir, filename)

# Load the data
data = np.loadtxt(filepath, delimiter=",", skiprows=1)
h = data[:, 0]
error_norm = data[:, 1]

# Create the figure
plt.figure(figsize=(8,6))

# Plot the Jacobi error
plt.loglog(h, error_norm, 'o-', label='Jacobi Method Error')

# Plot a reference h^2 line (scaled to match the errors roughly)
ref_line = (error_norm[0] / h[0]**2) * h**2
plt.loglog(h, ref_line, '--', label=r'Reference $\mathcal{O}(h^2)$')

# Labels and title
plt.xlabel('Grid spacing $h$')
plt.ylabel('Error norm $\|u_h - u\|$')
plt.title('Jacobi Method Convergence')
plt.legend()
plt.grid(True, which="both", ls="--")

# Show the plot
plt.show()
