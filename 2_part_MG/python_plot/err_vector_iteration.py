import os
import matplotlib.pyplot as plt
import numpy as np

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "..", "OUTPUT_RESULT", "ERR_VECTOR")
plot_dir = os.path.join(script_dir, "..", "PLOT")
os.makedirs(plot_dir, exist_ok=True)

iteration_performed = 5 + 1  # Adjust based on actual number of iterations + initial_err

def read_errors(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        L = int(lines[0].strip())
        values = [float(line.strip()) for line in lines[1:]]
    return np.array(values), L

# Prepare the plot layout
cols = iteration_performed
rows = 1
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4), constrained_layout=True)

if cols == 1:
    axes = [axes]  # Ensure axes is iterable

for i in range(iteration_performed):
    file_path = os.path.join(input_dir, f"iteration_{i}.txt")
    err_vector, L = read_errors(file_path)

    N = int(np.sqrt(L))
    if N * N != L:
        raise ValueError(f"Error vector length {L} is not a perfect square. Cannot reshape to 2D.")

    err_2d = err_vector.reshape((N, N))

    fft_2d = np.fft.fft2(err_2d)
    fft_shifted = np.fft.fftshift(np.abs(fft_2d))
    log_magnitude = np.log1p(fft_shifted)

    ax = axes[i]
    im = ax.imshow(log_magnitude, cmap='inferno', origin='lower')
    ax.set_title(f"Iter {i}")
    ax.axis('off')

# Add shared colorbar
cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label("log(1 + Magnitude)")

# Save and show
combined_path = os.path.join(plot_dir, "all_error_2d_fft.pdf")
plt.savefig(combined_path, format='pdf', bbox_inches='tight')
plt.show()

print(f"Combined FFT subplot saved to: {combined_path}")
