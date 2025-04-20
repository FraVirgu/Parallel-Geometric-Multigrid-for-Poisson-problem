import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

def txt_to_array(file_path):
    """
    Converts a .txt file with the first line as N and subsequent lines as values into a NumPy array.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            N = int(lines[0].strip())
            array = np.loadtxt(lines[1:])
        return N, array
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None, None

def compute_fft_log_magnitude(array_2d):
    fft_2d = np.fft.fft2(array_2d)
    fft_shifted = np.fft.fftshift(fft_2d)
    magnitude_log = np.log(np.abs(fft_shifted) + 1e-12)
    return magnitude_log

if __name__ == "__main__":
    error_dir = "OUTPUT_RESULT/ERROR_VECTOR"
    plot_dir = "PLOT/FREQ_2D_3D"
    os.makedirs(plot_dir, exist_ok=True)

    # File paths for iteration 0 and 50
    file_0 = os.path.join(error_dir, "h_errors_jacobi_0.txt")
    file_5 = os.path.join(error_dir, "h_errors_jacobi_50.txt")

    N0, array0 = txt_to_array(file_0)
    N5, array5 = txt_to_array(file_5)

    if array0 is not None and array5 is not None and N0 == N5:
        array0_2d = np.reshape(array0, (N0, N0))
        array5_2d = np.reshape(array5, (N5, N5))

        # Compute log-magnitude of FFTs
        mag0 = compute_fft_log_magnitude(array0_2d)
        mag5 = compute_fft_log_magnitude(array5_2d)

        # Compute difference
        delta_mag = mag0 - mag5  # Change in log magnitude

        # Create frequency grid
        freq_x = np.fft.fftshift(np.fft.fftfreq(N0))
        freq_y = np.fft.fftshift(np.fft.fftfreq(N0))
        X, Y = np.meshgrid(freq_x, freq_y)

        # Plot the difference in 3D
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, delta_mag, cmap='coolwarm', edgecolor='none')

        ax.set_title("Frequency Spectrum Change: Iter 0 - Iter 50")
        ax.set_xlabel("Frequency X")
        ax.set_ylabel("Frequency Y")
        ax.set_zlabel("Δ log(|FFT|)")
        ax.view_init(elev=45, azim=135)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

        # Save the plot
        save_path = os.path.join(plot_dir, "jacobi_fft_difference_0_50.png")
        plt.savefig(save_path)
        plt.show()

        print(f"Saved 3D frequency difference plot to: {save_path}")
    else:
        print("Error loading data or mismatched dimensions.")
