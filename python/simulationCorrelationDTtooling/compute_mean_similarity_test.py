# We will be using different metrics to measure the similarity between the real pressure and the
# simulated pressure:
# - The Frechet distance correspond to the maximum difference between the two curves
# - The DTW (Dynamic Time Wraping) to compute the optimal match between two time series. 
#       The algorithm calculates the minimum cumulative distance needed to align one sequence with another, 
#       taking into account possible stretching and compressing of the time axis.

from dtw import *
import numpy as np
import matplotlib.pyplot as plt
import similaritymeasures

# Generate data points
x = np.linspace(-3*np.pi, 3*np.pi, 100)
x2 = np.linspace(-3*np.pi, 3*np.pi, 100)
sine = np.sin(x)
sine_10 = np.sin(1.1*x)
cosine = np.cos(x)
sinx = np.sin(x2)/x2

# Calculate DTW
dtw_sine_cosine = dtw(sine, cosine, keep_internals=True)
dtw_sine_sine = dtw(cosine, sine_10, keep_internals=True)
dtw_sine_sinx = dtw(cosine, sinx, keep_internals=True)

# Prepare the data for similaritymeasures
# Frechet distance requires input in the form of 2D arrays
sine_curve = np.column_stack((x, sine))
sine_10_curve = np.column_stack((x, sine_10))
cosine_curve = np.column_stack((x, cosine))
sinx_curve = np.column_stack((x2, sinx))

# Calculate the Frechet distances
frechet_dist_sine_cosine = similaritymeasures.frechet_dist(sine_curve, cosine_curve)
frechet_dist_sine_sine = similaritymeasures.frechet_dist(cosine_curve, sine_10_curve)
frechet_dist_sine_sinx = similaritymeasures.frechet_dist(sine_curve, sinx_curve)

# Print DTW distances
print("DTW Distance (Sine vs. Cosine):", dtw_sine_cosine.distance)
print("DTW Distance (Sine vs. Sine):", dtw_sine_sine.distance)
print("DTW Distance (Cosine vs. Sine / x):", dtw_sine_sinx.distance)

# Print Frechet distances
print("Frechet Distance (Sine vs. Cosine):", frechet_dist_sine_cosine)
print("Frechet Distance (Sine vs. Sine):", frechet_dist_sine_sine)
print("Frechet Distance (Cosine vs. Sinx):", frechet_dist_sine_sinx)

# Plot the alignments and the cost matrices
fig, axs = plt.subplots(3, 2, figsize=(12, 10))

# Plotting Sine vs. Cosine
axs[0, 0].plot(x, sine, label='Sine')
axs[0, 0].plot(x, cosine, label='Cosine', linestyle='--')
axs[0, 0].set_title("Sine vs. Cosine")
axs[0, 0].legend()

axs[0, 1].imshow(dtw_sine_cosine.costMatrix.T, origin='lower', aspect='auto', cmap='viridis')
axs[0, 1].plot(dtw_sine_cosine.index1, dtw_sine_cosine.index2, 'w')
axs[0, 1].set_title("DTW Cost Matrix (Sine vs. Cosine)")
axs[0, 1].set_xlabel('Sine Index')
axs[0, 1].set_ylabel('Cosine Index')

# Plotting Sine vs. Hyperbolic Sine
axs[1, 0].plot(x, cosine, label='Cosine')
axs[1, 0].plot(x2, sinx, label='Sine / x', linestyle='--')
axs[1, 0].set_title("Cosine vs. Sine / x")
axs[1, 0].legend()

axs[1, 1].imshow(dtw_sine_sinx.costMatrix.T, origin='lower', aspect='auto', cmap='viridis')
axs[1, 1].plot(dtw_sine_sinx.index1, dtw_sine_sinx.index2, 'w')
axs[1, 1].set_title("DTW Cost Matrix (Cosine vs. Sine / x)")
axs[1, 1].set_xlabel('Cosine Index')
axs[1, 1].set_ylabel('Sine / x Index')

# Plotting Sine vs. Sine
axs[2, 0].plot(x, sine, label='Sine')
axs[2, 0].plot(x, sine_10, label='Sine', linestyle='--')
axs[2, 0].set_title("Sine vs. Sine")
axs[2, 0].legend()

axs[2, 1].imshow(dtw_sine_sine.costMatrix.T, origin='lower', aspect='auto', cmap='viridis')
axs[2, 1].plot(dtw_sine_sine.index1, dtw_sine_sine.index2, 'w')
axs[2, 1].set_title("DTW Cost Matrix (Sine vs. Sine)")
axs[2, 1].set_xlabel('Sine Index')
axs[2, 1].set_ylabel('Sine Index')

plt.tight_layout()
plt.show()

## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
dtw(cosine, sinx, keep_internals=True,
    step_pattern=rabinerJuangStepPattern(6, "c"))\
    .plot(type="twoway",offset=-2)

## See the recursion relation, as formula and diagram
# print(rabinerJuangStepPattern(6,"c"))
# rabinerJuangStepPattern(6,"c").plot()
plt.show()