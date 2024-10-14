import pandas as pd
import scipy.io as sio
import numpy as np

def normalize_data(data, new_min, new_max):
    # Get the minimum and maximum of the original data
    min_val = np.min(data)
    max_val = np.max(data)
    
    # Apply min-max normalization formula
    normalized_data = (data - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
    
    return normalized_data

def csv_to_mat(csv_file_path, mat_file_path):
    # Step 1: Read the CSV file into a pandas DataFrame, assuming no header
    df = pd.read_csv(csv_file_path, header=None)
    
    # Step 2: Convert the DataFrame to a NumPy array
    data_array = df.to_numpy()
    
    # Step 3: Normalize the data with the desired min and max values
    normalized_array = normalize_data(data_array, new_min=-18.2524, new_max=18.2524)
    
    # Step 4: Store the normalized data in a dictionary under the key 'VORTALL'
    data_dict = {'VORTALL': normalized_array, 'nx': 2**7, 'ny': 2**8}
    
    # Step 5: Save the dictionary to a .mat file
    sio.savemat(mat_file_path, data_dict)
    
    print(f"Successfully converted {csv_file_path} to {mat_file_path} with normalized data stored under 'VORTALL'")

# Example usage:
csv_file = 'E:/benchmark_SINDy/FullVorticityMapRe100_h60.csv'
mat_file = 'FullVorticityMapRe100_h60.mat'
csv_to_mat(csv_file, mat_file)
