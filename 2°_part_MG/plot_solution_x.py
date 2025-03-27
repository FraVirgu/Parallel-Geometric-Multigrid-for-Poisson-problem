import numpy as np
import matplotlib.pyplot as plot
def txt_to_array(file_path):
    """
    Converts a .txt file full of values into a NumPy array.
    
    Parameters:
        file_path (str): Path to the .txt file.
    
    Returns:
        np.ndarray: Array of values from the file.
    """
    try:
        array = np.loadtxt(file_path)
        return array
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None

# Example usage
if __name__ == "__main__":
    file_path = "solution.txt"  # Replace with your .txt file path
    array = txt_to_array(file_path)
    N = 128
    array = np.reshape(array, (N, N))
    if array is not None:
        print("Array loaded successfully:")
        print(np.shape(array))
    
    x = np.arange(0, N + 1)
    y = np.arange(0, N + 1)
    z = array

    fig, ax = plot.subplots()
    ax.pcolormesh(x, y, z, cmap='viridis')
    plot.savefig("solution.png")
    
    
  



