import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pysindyOld as ps 
from sklearn.decomposition import TruncatedSVD

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-15
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-10

# Initialize quadratic SINDy library, with custom ordering 
# to be consistent with the constraint
library_functions = [lambda x:x, lambda x, y:x * y, lambda x:x ** 2]
library_function_names = [lambda x:x, lambda x, y:x + y, lambda x:x + x]
sindy_library = ps.CustomLibrary(library_functions=library_functions, 
                                 function_names=library_function_names)

# Step 1: Load data from CSV
def load_data(file_path):
    """Load data from a CSV file."""
    data = pd.read_csv(file_path, header=None)
    flow_snapshots = data.values  # Convert dataframe to numpy array
    return flow_snapshots#[883:901, :]

# Step 2: Compute the mean flow
def compute_mean_flow(flow_snapshots):
    """Compute the mean flow by averaging the flow snapshots over time."""
    mean_flow = np.mean(flow_snapshots, axis=1)  # Mean over time (axis=1)
    return mean_flow

# Step 3: Apply POD using TruncatedSVD
def apply_pod(flow_snapshots, n_modes):
    """Perform mean-substracted POD using TruncatedSVD and return the POD modes and coefficients."""

    # Compute mean flow
    mean_flow = compute_mean_flow(flow_snapshots)

    # Compute the mean flow deformation (difference between snapshots and mean flow)
    mean_deformation = flow_snapshots - mean_flow[:, np.newaxis]

    # Transpose the matrix for correct application of SVD
    svd = TruncatedSVD(n_components=n_modes)
    pod_modes = svd.fit(mean_deformation).components_  # POD modes (spatial basis functions)
    pod_coefficients = svd.transform(mean_deformation)  # Time-dependent coefficients
    return pod_modes.T, pod_coefficients

# Step 4: Add mean flow deformation mode
def add_mean_flow_deformation_mode(file_path, pod_modes):
    """Add the mean flow deformation as the last mode."""
    # Load shift mode
    print(file_path[:-4]+"_baseFlow.csv")
    shift_flow = load_data(file_path[:-4]+"_baseFlow.csv")
    shift_mode = np.zeros(len(shift_flow))
    for i in range(len(shift_flow)):
        shift_mode[i] = shift_flow[i][249]
    
    # Normalize if norm is non-zero
    norm_deformation = np.linalg.norm(shift_mode)
    if norm_deformation != 0:
        shift_mode /= norm_deformation  # Normalize the deformation mode
    
    # Reshape the mean_flow_deformation_mode to (1, n_spatial_points) for stacking
    shift_mode = shift_mode.reshape(-1, 1)

    # Append the mean flow deformation mode to the existing POD modes
    pod_modes_with_deformation = np.hstack([pod_modes, shift_mode])
    
    return pod_modes_with_deformation

# Main routine to load the data and apply the methods
def main(file_path, n_modes):
    # Load data
    flow_snapshots = load_data(file_path)
    flow_snapshots = normalize_data(flow_snapshots[:, 1000:], new_min=-18.2524, new_max=18.2524)

    # Apply POD
    pod_modes, pod_coefficients = apply_pod(flow_snapshots, n_modes)

    # Add mean flow deformation mode
    # pod_modes_with_deformation = add_mean_flow_deformation_mode(file_path, pod_modes)

    return pod_modes, pod_coefficients

# Build the energy-preserving quadratic nonlinearity constraints
def make_constraints(r):
    q = 0
    N = int((r ** 2 + 3 * r) / 2.0)
    p = r + r * (r - 1) + int(r * (r - 1) * (r - 2) / 6.0)
    constraint_zeros = np.zeros(p)
    constraint_matrix = np.zeros((p, r * N))    
    
    # Set coefficients adorning terms like a_i^3 to zero
    for i in range(r):
        constraint_matrix[q, r * (N - r) + i * (r + 1)] = 1.0
        q = q + 1

    # Set coefficients adorning terms like a_ia_j^2 to be antisymmetric
    for i in range(r):
        for j in range(i + 1, r):
            constraint_matrix[q, r * (N - r + j) + i] = 1.0
            constraint_matrix[q, r * (r + j - 1) + j + r * int(i * (2 * r - i - 3) / 2.0)] = 1.0
            q = q + 1
    for i in range(r):
         for j in range(0, i):
            constraint_matrix[q, r * (N - r + j) + i] = 1.0
            constraint_matrix[q, r * (r + i - 1) + j + r * int(j * (2 * r - j - 3) / 2.0)] = 1.0
            q = q + 1

    # Set coefficients adorning terms like a_ia_ja_k to be antisymmetric
    for i in range(r):
        for j in range(i + 1, r):
            for k in range(j + 1, r):
                constraint_matrix[q, r * (r + k - 1) + i + r * int(j * (2 * r - j - 3) / 2.0)] = 1.0
                constraint_matrix[q, r * (r + k - 1) + j + r * int(i * (2 * r - i - 3) / 2.0)] = 1.0
                constraint_matrix[q, r * (r + j - 1) + k + r * int(i * (2 * r - i - 3) / 2.0)] = 1.0
                q = q + 1
                
    return constraint_zeros, constraint_matrix

def normalize_data(data, new_min, new_max):
    # Get the minimum and maximum of the original data
    min_val = np.min(data)
    max_val = np.max(data)
    
    # Apply min-max normalization formula
    normalized_data = (data - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
    
    return normalized_data


# -------------------- Benchmark with the POD result from the tutorial ---------
# define parameters and load in POD modes obtained from DNS
a = np.loadtxt('fusionPython/data/vonKarman_pod/vonKarman_a.dat')
# for i in range(1,10):
#     plt.plot(a[1400:,i])
# plt.show()

t = a[:, 0]
r = 9
if r < 9:
    a_temp = a[:, 1:r]
    a_temp = np.hstack((a_temp, a[:, -1].reshape(3000, 1)))
    a = a_temp
else:
    a = a[:, 1:r + 1]
tbegin = 0
tend = 3000
skip = 1
t = t[tbegin:tend:skip]
a = a[tbegin:tend:skip, :]
x_test9 = a
x_train9 = a
dt = t[1] - t[0]
a0 = np.zeros(9)
a0[0] = 1e-3

# initial guesses to speed up convergence
A_guess = np.asarray([[-1.00000000e+00, -6.81701156e-18,  1.64265568e-17,
         1.68310896e-17, -1.46247179e-17, -1.45278659e-17,
        -4.41460518e-17, -9.75820242e-18,  1.06635090e-18],
       [ 2.11962730e-17, -1.00000000e+00,  5.91916098e-17,
         1.02939874e-16,  1.07502638e-17,  1.16153337e-17,
         8.84561388e-17, -2.38407466e-17,  2.25533310e-17],
       [ 7.70290614e-18, -2.37493643e-17, -1.00000000e+00,
         5.74638700e-17, -2.17428595e-18,  2.41426251e-17,
         6.89329350e-18, -3.65633938e-18,  7.79059612e-17],
       [-2.44656476e-17, -6.72113002e-18,  4.74779569e-17,
        -1.00000000e+00, -8.10241368e-18,  6.23271318e-18,
         1.12682516e-18,  3.01538601e-17,  2.94041671e-16],
       [-1.29102703e-16,  1.28776608e-17,  4.48182893e-17,
        -7.15006179e-19, -1.00000000e+00,  4.11221110e-18,
        -3.33172879e-16, -4.22913612e-17, -5.88841351e-17],
       [ 7.10726824e-18,  9.55210532e-18, -5.30624590e-17,
        -1.99630356e-17, -4.88598954e-18, -1.00000000e+00,
        -1.72238787e-17,  1.45840342e-17, -1.29732583e-17],
       [-1.01481442e-17,  4.78393464e-17, -2.53411865e-17,
         1.31394318e-17, -5.96906289e-18,  1.68124806e-18,
        -1.00000000e+00, -1.51574587e-17, -2.15989255e-18],
       [-9.48456158e-18, -5.41527305e-18, -3.05384371e-19,
        -1.99553156e-18,  8.37718659e-17,  6.05188865e-17,
         3.94017953e-17, -1.00000000e+00, -1.69209548e-18],
       [ 2.85170680e-18,  2.40387704e-17,  8.14566833e-17,
         2.74548940e-17, -4.62236236e-18, -7.34952555e-18,
         4.64207566e-18,  1.69214151e-18, -1.00000000e+00]])
m_guess = np.asarray([-1.03179775e+00, -3.31216061e-01, -6.71172384e-01, 
                      -6.75994493e-01, -8.50522236e-03, -2.12185379e-02, 
                      -9.16401064e-04,  1.03271372e-03, 4.70226212e-01])

# define hyperparameters
max_iter = 10
eta = 1.0
threshold = 1
alpha_m = 5e-1 * eta

constraint_zeros, constraint_matrix = make_constraints(r)

# run trapping SINDy
sindy_opt = ps.TrappingSR3(threshold=threshold, eta=eta, alpha_m=alpha_m,
                           m0=m_guess, A0=A_guess, max_iter=max_iter, 
                           constraint_lhs=constraint_matrix,
                           constraint_rhs=constraint_zeros,
                           constraint_order="feature",
                           )
model = ps.SINDy(
    optimizer=sindy_opt,
    feature_library=sindy_library,
    differentiation_method=ps.FiniteDifference(drop_endpoints=True),
)
# model.fit(x_train9, t=t)
# Xi = model.coefficients().T
# xdot_test9 = model.differentiate(x_test9, t=t)
# xdot_test_pred9 = model.predict(x_test9)
# PL_tensor = sindy_opt.PL_
# PQ_tensor = sindy_opt.PQ_
# L = np.tensordot(PL_tensor, Xi, axes=([3, 2], [0, 1]))
# Q = np.tensordot(PQ_tensor, Xi, axes=([4, 3], [0, 1]))
# x_train_pred9 = model.simulate(x_train9[0, :], t)
# x_test_pred9 = model.simulate(a0, t)

# ----------------------- Custom DMD modes -------------------------
from pydmd import DMD
from pydmd.plotter import plot_summary

flow_snapshots = load_data('E:/benchmark_SINDy/FullVorticityMapRe100_h60.csv')
normalized_array = normalize_data(flow_snapshots, new_min=-18.2524, new_max=18.2524)
dmd = DMD(svd_rank=12)
dmd.fit(normalized_array)

# plot_summary(
#     dmd,
#     figsize=(12, 7),
#     index_modes=(0, 2, 4),
#     snapshots_shape=(2**7, 2**8),
#     order="F",
#     mode_cmap="seismic",
#     dynamics_color="k",
#     flip_continuous_axes=True,
#     max_sval_plot=30
# )

# ----------------------- Custom POD modes -------------------------
file_path = 'E:/benchmark_SINDy/FullVorticityMapRe100_h60.csv'
n_modes = 9  # Number of POD modes to extract
# pod_modes_with_deformation, pod_coefficients = main(file_path, n_modes)
pod_modes_with_deformation, pod_coefficients = np.linalg.norm(dmd.dynamics), 0
t_custom = np.arange(0, 0.6194 * pod_modes_with_deformation.size, 0.6194)
r = 9
if r < 9:
    a_temp = pod_modes_with_deformation
    a_temp = np.hstack((a_temp, pod_modes_with_deformation[:, -1].reshape(1676, 1)))
    a_custom = a_temp
else:
    a_custom = pod_modes_with_deformation
tbegin = 0
tend = 1676
skip = 1
t_custom = t_custom[tbegin:tend:skip]
a_custom = a_custom[tbegin:tend:skip, :]
x_test9_custom = a_custom
x_train9_custom = a_custom
dt_custom = t[1] - t[0]
a0_custom = np.zeros(9)
a0_custom[0] = 1e-3

# initial guesses to speed up convergence
A_guess = np.asarray([[-1.00000000e+00, -6.81701156e-18,  1.64265568e-17,
         1.68310896e-17, -1.46247179e-17, -1.45278659e-17,
        -4.41460518e-17, -9.75820242e-18,  1.06635090e-18],
       [ 2.11962730e-17, -1.00000000e+00,  5.91916098e-17,
         1.02939874e-16,  1.07502638e-17,  1.16153337e-17,
         8.84561388e-17, -2.38407466e-17,  2.25533310e-17],
       [ 7.70290614e-18, -2.37493643e-17, -1.00000000e+00,
         5.74638700e-17, -2.17428595e-18,  2.41426251e-17,
         6.89329350e-18, -3.65633938e-18,  7.79059612e-17],
       [-2.44656476e-17, -6.72113002e-18,  4.74779569e-17,
        -1.00000000e+00, -8.10241368e-18,  6.23271318e-18,
         1.12682516e-18,  3.01538601e-17,  2.94041671e-16],
       [-1.29102703e-16,  1.28776608e-17,  4.48182893e-17,
        -7.15006179e-19, -1.00000000e+00,  4.11221110e-18,
        -3.33172879e-16, -4.22913612e-17, -5.88841351e-17],
       [ 7.10726824e-18,  9.55210532e-18, -5.30624590e-17,
        -1.99630356e-17, -4.88598954e-18, -1.00000000e+00,
        -1.72238787e-17,  1.45840342e-17, -1.29732583e-17],
       [-1.01481442e-17,  4.78393464e-17, -2.53411865e-17,
         1.31394318e-17, -5.96906289e-18,  1.68124806e-18,
        -1.00000000e+00, -1.51574587e-17, -2.15989255e-18],
       [-9.48456158e-18, -5.41527305e-18, -3.05384371e-19,
        -1.99553156e-18,  8.37718659e-17,  6.05188865e-17,
         3.94017953e-17, -1.00000000e+00, -1.69209548e-18],
       [ 2.85170680e-18,  2.40387704e-17,  8.14566833e-17,
         2.74548940e-17, -4.62236236e-18, -7.34952555e-18,
         4.64207566e-18,  1.69214151e-18, -1.00000000e+00]])
m_guess = np.asarray([-1.03179775e+00, -3.31216061e-01, -6.71172384e-01, 
                      -6.75994493e-01, -8.50522236e-03, -2.12185379e-02, 
                      -9.16401064e-04,  1.03271372e-03, 4.70226212e-01])

# define hyperparameters
max_iter = 1
eta = 1.0
threshold = 1
alpha_m = 5e-1 * eta

constraint_zeros, constraint_matrix = make_constraints(r)

# run trapping SINDy
sindy_opt = ps.TrappingSR3(threshold=threshold, eta=eta, alpha_m=alpha_m,
                           m0=m_guess, A0=A_guess, max_iter=max_iter, 
                           constraint_lhs=constraint_matrix,
                           constraint_rhs=constraint_zeros,
                           constraint_order="feature",
                           )
model = ps.SINDy(
    optimizer=sindy_opt,
    feature_library=sindy_library,
    differentiation_method=ps.FiniteDifference(drop_endpoints=True),
)
print(x_test9_custom.shape[-2], len(t_custom))
model.fit(x_train9_custom, t=t_custom)
Xi_custom = model.coefficients().T
xdot_test9_custom = model.differentiate(x_test9_custom, t=t_custom)
xdot_test_pred9_custom = model.predict(x_test9_custom)
PL_tensor_custom = sindy_opt.PL_
PQ_tensor_custom = sindy_opt.PQ_
L_custom = np.tensordot(PL_tensor_custom, Xi_custom, axes=([3, 2], [0, 1]))
Q_custom = np.tensordot(PQ_tensor_custom, Xi_custom, axes=([4, 3], [0, 1]))
x_train_pred9_custom = model.simulate(x_train9_custom[0, :], t_custom)
x_test_pred9_custom = model.simulate(a0_custom, t_custom)

# define energies of the DNS, and the trapping SINDy models
E = np.sum(a ** 2, axis=1)
# E_sindy9 = np.sum(x_test_pred9 ** 2, axis=1)
E_custom = np.sum(a_custom ** 2, axis=1)
E_sindy9_custom = np.sum(x_test_pred9_custom ** 2, axis=1)

# plot the energies
plt.figure(figsize=(16, 4))
# plt.plot(t, E, 'r', label='DNS')
# plt.plot(t, E_sindy9, 'c', label=r'SINDy-9')
plt.plot(t_custom, E_custom, 'b', label='DNS')
plt.plot(t_custom, E_sindy9_custom, 'o', label=r'SINDy-9')

# do some formatting and save
plt.legend(fontsize=22, loc=2)
plt.grid()
plt.xlim([0, 300])
ax = plt.gca()
# ax.set_yticks([0, 10, 20])
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.ylabel('Total energy', fontsize=20)
plt.xlabel('t', fontsize=20)
plt.show()