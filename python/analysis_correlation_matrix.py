from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Set global font size
plt.rcParams.update({'font.size': 14})

auto_cross_correlation_result = np.load("E:/crop_nadia/auto_cross-correlation.npy")
cross_correlation_result = np.load("E:/crop_nadia/cross-correlation.npy")
cyl_correlation_results = np.load("E:/crop_nadia/auto_cross-correlation_cylinder.npy")


def analyze_mean_modes(matrix):
    # Calculate the means of each row
    means = np.mean(matrix, axis=1)
    plt.figure()
    path = 'E:/simuChina/cartesianBDD_FullPressureMap'
    file = "FullMap.csv"
    folder = sorted(os.listdir(path), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf'))

    plt.scatter(folder,means)
    print(np.linspace(1,5,10))
    
    # Prepare the histogram and KDE
    plt.figure(figsize=(10, 6))
    plt.hist(means, bins=50, alpha=0.75, density=True, color='lightblue', label='Histogram of Means')
    
    # Apply KDE to the means
    kde = gaussian_kde(means)
    x_grid = np.linspace(means.min() - 1, means.max() + 1, 1000)
    kde_estimates = kde.evaluate(x_grid)
    
    # Plotting the KDE
    plt.plot(x_grid, kde_estimates, label='KDE of Means', color='red')
    
    # Find peaks in the KDE to determine modes
    peaks, _ = find_peaks(kde_estimates)
    plt.scatter(x_grid[peaks], kde_estimates[peaks], color='darkred', zorder=10, s=100, label='Peaks')
    
    # Adding plot details
    plt.title('Histogram and KDE for Mean Distribution of Cross-correlations')
    plt.xlabel('Mean of Maximum Cross-correlation Value')
    plt.ylabel('Density')
    plt.xlim([0,1])
    plt.legend()
    plt.show()
    
    # Outputting the number of modes
    print(f"The distribution of mean values has {len(peaks)} modes.")
    print(f"The peaks are situated at {x_grid[peaks]}")

# analyze_mean_modes(auto_cross_correlation_result)
# analyze_mean_modes(cross_correlation_result)
analyze_mean_modes(cyl_correlation_results)

print("Mean of the difference:", np.mean(auto_cross_correlation_result)-np.mean(cross_correlation_result))