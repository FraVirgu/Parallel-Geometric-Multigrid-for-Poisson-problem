import numpy as np
import matplotlib.pyplot as plot
N = 0

def txt_to_array(file_path):
    """
    Converts a .txt file with the first line as N and subsequent lines as values into a NumPy array.
    
    Parameters:
        file_path (str): Path to the .txt file.
    
    Returns:
        np.ndarray: Array of values from the file.
    """
    global N
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            N = int(lines[0].strip())  # Read the first line as N
            array = np.loadtxt(lines[1:])  # Load the remaining lines as the array
        return array
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None

# Example usage
if __name__ == "__main__":
    file_path = "solution.txt"  # Replace with your .txt file path
    array = txt_to_array(file_path)
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
    
    
  



