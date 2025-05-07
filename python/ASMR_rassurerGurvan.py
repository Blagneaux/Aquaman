import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Seed for reproducibility
np.random.seed(42)

# Function definitions
def complex_function(X):
    return np.sin(X) * np.cos(X * 0.5) + np.cos(X * 1.5)

def sec_complex_function(X):
    return np.exp(1 / (X + 1)) * np.sin(X) * X

index = []

# Initial data
X_train = np.array([0, 10])[:, np.newaxis]  # extremities
X_train = np.vstack([X_train, np.random.uniform(0, 10, 2)[:, np.newaxis]])  # adding more points
y_train = complex_function(X_train).ravel()

X2_train = np.array([0, 10])[:, np.newaxis]  # extremities
X2_train = np.vstack([X2_train, np.random.uniform(0, 10, 2)[:, np.newaxis]])  # adding more points
y2_train = sec_complex_function(X2_train).ravel()

# Gaussian Process setup
kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gpr2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Number of iterations and plotting frequency
num_iterations = 30
plot_frequency = 10  # plot every iteration

# Determine number of plots and set up the figure
num_plots = num_iterations // plot_frequency + (num_iterations % plot_frequency > 0)
plt.figure(figsize=(16, 8 * num_iterations))  # adjust subplot size based on number of iterations

for i in range(num_iterations):
    # Fit GPRs
    gpr.fit(X_train, y_train)
    gpr2.fit(X2_train, y2_train)

    # Predictions
    X_plot = np.linspace(0, 10, 1000)[:, np.newaxis]
    y_pred, sigma = gpr.predict(X_plot, return_std=True)
    y_pred2, sigma2 = gpr2.predict(X_plot, return_std=True)

    # Point of maximum disagreement (based on standard deviation this time)
    disagreement = np.abs(sigma - sigma2)
    new_index = np.argmax(disagreement)
    index.append(new_index)

    # Check if the point has already been measured
    while X_plot[new_index] in X_train:
        available_indices = [idx for idx in range(len(X_plot)) if X_plot[idx] not in X_train]
        if not available_indices:
            break
        new_index = np.random.choice(available_indices)

    new_X = X_plot[new_index].reshape(-1, 1)
    new_y = complex_function(new_X).ravel()
    new_y2 = sec_complex_function(new_X).ravel()
    X_train = np.vstack([X_train, new_X])
    y_train = np.concatenate([y_train, new_y])
    X2_train = np.vstack([X2_train, new_X])
    y2_train = np.concatenate([y2_train, new_y2])

    # Plotting only on specific iterations
    if i % plot_frequency == 0:
        # First function plot
        plt.subplot(num_iterations, 2, 2 * i + 1)
        plt.plot(X_plot, y_pred, 'b-', label='Prediction of f1')
        plt.fill_between(X_plot[:, 0], y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, color='blue')
        plt.plot(X_plot, disagreement, 'k--', label='Disagreement')
        plt.scatter(X_train, y_train, c='r', s=50, zorder=10, edgecolors=(0, 0, 0), label='Data Points of f1')
        plt.scatter(new_X, new_y, c='g', s=100, label='New Point from Max Disagreement', edgecolors=(0, 0, 0))
        plt.title(f'Iteration {i+1} for Function 1')
        plt.legend()

        # Second function plot
        plt.subplot(num_iterations, 2, 2 * i + 2)
        plt.plot(X_plot, y_pred2, 'b-', label='Prediction of f2')
        plt.fill_between(X_plot[:, 0], y_pred2 - 1.96 * sigma2, y_pred2 + 1.96 * sigma2, alpha=0.2, color='blue')
        plt.plot(X_plot, disagreement, 'k--', label='Disagreement')
        plt.scatter(X2_train, y2_train, c='r', s=50, zorder=10, edgecolors=(0, 0, 0), label='Data Points of f2')
        plt.scatter(new_X, new_y2, c='g', s=100, label='New Point from Max Disagreement', edgecolors=(0, 0, 0))
        plt.title(f'Iteration {i+1} for Function 2')
        plt.legend()

plt.tight_layout()
plt.show()

print(index)
