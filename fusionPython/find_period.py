import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to compute the autocorrelation
def autocorrelation_with_reference(data):
    # Take the last column as the reference state
    reference_state = data[:, -1]
    
    # Compute the autocorrelation for each column with the reference column
    autocorrelations = []
    for i in range(data.shape[1]):
        # Normalize the data
        col = data[:, i]
        col = (col - np.mean(col)) / (np.std(col) + 1e-10)
        ref = (reference_state - np.mean(reference_state)) / (np.std(reference_state) + 1e-10)
        
        # Compute the autocorrelation between the current column and the reference state
        correlation = np.correlate(col, ref, mode='valid')[0]
        autocorrelations.append(correlation)

    return autocorrelations

# Function to plot the autocorrelation results
def plot_autocorrelation(autocorrelations):
    plt.figure(figsize=(10, 6))
    plt.plot(autocorrelations, marker='o', linestyle='-', color='b')
    plt.title('Autocorrelation of System States with Reference State (Last Column) for the worst case scenario (heighest speed and smallest distance)')
    plt.xlabel('Time Steps')
    plt.ylabel('Autocorrelation')
    plt.grid(True)
    plt.show()

def load_data(path):
    """Load data from a CSV file"""
    data = pd.read_csv(path, header=None)
    snapshots = data.values  # Convert dataframe into numpy array
    return snapshots

# Example usage:
# Assume your dataset is a 2D NumPy array where each column is the state at a time step
# data = np.array([[...], [...], ...])  # Replace with your actual dataset
full_data_path = 'E:/benchmark_SINDy/FullPressureMapRe10000_h6_extended.csv'
data = load_data(full_data_path)

# Calculate autocorrelation
autocorr_results = autocorrelation_with_reference(data)

# Plot the autocorrelation results
plot_autocorrelation(autocorr_results)