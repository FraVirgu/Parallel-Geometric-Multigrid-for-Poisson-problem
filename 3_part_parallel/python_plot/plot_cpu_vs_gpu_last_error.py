import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# ---------------------------
# Paths
# ---------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "..", "OUTPUT_RESULT", "ERR_VECTOR")
plot_dir = os.path.join(script_dir, "..", "PLOT")
os.makedirs(plot_dir, exist_ok=True)

# ---------------------------
# Load and reshape error file
# ---------------------------

def read_error_2d(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        L = int(lines[0].strip())
        err_vector = np.array([float(line.strip()) for line in lines[1:]])

    N = int(np.sqrt(L))
    if N * N != L:
        raise ValueError(f"Error vector length {L} is not a perfect square.")
    
    return err_vector.reshape((N, N)), N

# ---------------------------
# Read both error fields
# ---------------------------

cpu_error, N_cpu = read_error_2d(os.path.join(input_dir, "iteration_last_cpu.txt"))
gpu_error, N_gpu = read_error_2d(os.path.join(input_dir, "iteration_last_gpu.txt"))

# ---------------------------
# Match sizes via interpolation
# ---------------------------

if N_cpu != N_gpu:
    if N_cpu > N_gpu:
        # Upscale GPU error to match CPU size
        scale = N_cpu / N_gpu
        gpu_error_resized = zoom(gpu_error, zoom=scale, order=1)
        cpu_error_resized = cpu_error
    else:
        # Upscale CPU error to match GPU size
        scale = N_gpu / N_cpu
        cpu_error_resized = zoom(cpu_error, zoom=scale, order=1)
        gpu_error_resized = gpu_error
else:
    cpu_error_resized = cpu_error
    gpu_error_resized = gpu_error

# ---------------------------
# Compute and plot difference
# ---------------------------

difference = gpu_error_resized - cpu_error_resized

plt.figure(figsize=(6, 5))
im = plt.imshow(difference, cmap='bwr', origin='lower')
plt.colorbar(im, label='GPU Error - CPU Error')
plt.title(f"Difference Map (rescaled to N = {difference.shape[0]})")
plt.axis('off')

# Save
diff_path = os.path.join(plot_dir, "error_difference_gpu_minus_cpu.pdf")
plt.savefig(diff_path, format='pdf', bbox_inches='tight')
plt.show()

print(f"Difference map saved to: {diff_path}")
