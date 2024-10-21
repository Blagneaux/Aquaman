import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pysindy as ps 
import matplotlib.animation as animation

def load_data(path):
    """Load data from a CSV file"""
    data = pd.read_csv(path, header=None)
    snapshots = data.values  # Convert dataframe into numpy array
    return snapshots

def compute_mean_flow(snapshots, strat_index=0):
    """Compute the mean flow by averaging the flow snapshots over time"""
    # period_to_add = (3000 - len(snapshots[0]))//(3398-2400)
    # for i in range(period_to_add):
    #     snapshots = np.hstack([snapshots, snapshots[:, 2400:]])
    mean_flow = np.mean(snapshots[:, strat_index:], axis=1)
    return mean_flow

def find_matching_column(arr):
    """Compute the index of the column of the full data to apply the POD to a single period"""
    # Get the last column of the array
    last_column = arr[:, -1]
    
    # Iterate over columns from the second last to the first
    for i in range(arr.shape[1] - 2, -1, -1):
        if np.array_equal(arr[:, i], last_column):
            return i
            
    # If no match is found, return None
    return None

def find_min_difference_column(arr):
    """Compute the index of the column that minimizes the difference to the last column of the array."""
    # Get the last column of the array
    last_column = arr[:, -1]

    # Initialize a variable to store the index of the column with the minimal difference
    min_index = None
    # Initialize a variable to store the minimal sum of squared differences
    min_diff = np.inf  # Set to infinity initially

    # Iterate over all columns, excluding the last one
    for i in range(arr.shape[1] - 1):  # Exclude the last column in the comparison
        # Compute the sum of squared differences with the last column
        diff = np.sum((arr[:, i] - last_column) ** 2)

        # Check if the current column has a smaller difference than what we have seen so far
        if diff < min_diff:
            min_diff = diff
            min_index = i

    return min_index

def min_max_normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = 2*((arr - min_val) / (max_val - min_val)) - 1
    return normalized_arr

def plot_frames_and_signals(frames, signals):
    # Number of frames/signals
    num_frames = frames.shape[1]
    
    # Check if dimensions match
    if signals.shape[0] != num_frames:
        print("Error: The number of columns in the frames array must match the number of rows in the signals array.")
        return
    
    # Iterate through each frame
    for i in range(num_frames):
        # Reshape the flattened frame to 256x128
        frame = frames[:, i].reshape(256, 128)

         # Rotate the frame by 90 degrees
        frame_rotated = np.rot90(frame)
        
        # Corresponding signal
        signal = signals[i, :]
        
        # Create a new figure for each frame and signal
        plt.figure(figsize=(16, 6))
        
        # Plotting the frame
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
        plt.imshow(frame_rotated, cmap='seismic', aspect='auto')
        plt.title(f'Frame {i+1}')
        plt.colorbar(label='Pixel Intensity')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        
        # Plotting the signals
        plt.subplot(1, 2, 2)
        max_length = max(len(signal), len(a[:, i+1]))
        x_axis_signal = np.linspace(0, max_length-1, len(signal))
        x_axis_a = np.linspace(0, max_length-1, len(a[:, i+1]))
        
        plt.plot(x_axis_signal, signal, label='Signal')
        plt.plot(x_axis_a, a[:, i+1], label='Real Signal')
        plt.title(f'Signal {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Signal Amplitude')
        plt.xlim([0, max_length-1])  # Set the same x limits for both plots

        plt.legend()
        plt.tight_layout()

    plt.figure(figsize=(16, 6))
        
    # Plotting the frame
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.imshow(np.rot90(compute_mean_flow(load_data(full_data_path), 2400).reshape(256, 128)), cmap='seismic', aspect='auto')
    plt.title(f'Frame {i+1}')
    plt.colorbar(label='Pixel Intensity')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    plt.show()


def flow_reconstruction(frames, signals):
    steps = signals.shape[1]
    fig = plt.figure()  # Initialize the figure outside the loop

    real_frames = load_data(full_data_path)

    img = []
    mean_flow = compute_mean_flow(load_data(full_data_path), 2400)

    for i in range(steps):
        frame = frames @ signals[:, i]
        frame = np.vstack([frame + mean_flow, real_frames[:, i]])
        frame = frame.reshape(256*2, 128)
        frame_rotated = np.rot90(frame)

        img.append([plt.imshow(frame_rotated, cmap="seismic", aspect='auto')])

    ani = animation.ArtistAnimation(fig, img, interval=50, blit=True, repeat_delay=100)
    plt.show()

def POD_with_shift_mode(flow_path, base_path, n_modes, period_start=None):
    """Load the full data and the base flow, apply a POD to the full data, create a new 
    orthogonal base with the Gram-Schmidt process from the POD modes and the shift mode,
    and compute the dynamics of the new mode."""

    # Load the full data 
    full_data = load_data(flow_path)

    # Apply the POD to the mean-substracted flow on a single period of vortex shedding
    if period_start == None:
        period_cutoff = find_matching_column(full_data)
        if period_cutoff == None:
            return ("Error: ", "Flow has no periodic vortex shedding")
    else:
        print("Period start:", find_min_difference_column(full_data))
        period_cutoff = find_min_difference_column(full_data)

    # Compute the mean flow
    mean_flow = compute_mean_flow(full_data, 2400)

    # Select a vortex shedding period
    cut_data = full_data[:, period_cutoff:] - mean_flow[:, np.newaxis]

    # Compute the covariance matrix
    C = np.dot(cut_data.T, cut_data) / (cut_data.shape[-1] - 1)
    U, S, VT = np.linalg.svd(C, False)

    # Compute the modes of the POD
    Ur = np.dot(cut_data + mean_flow[:, np.newaxis], U)

    # Select the first n_modes - 1
    Ur, Sr, VTr = Ur[:, :n_modes-1], S[:n_modes-1], VT[:n_modes-1, :]

    # Compute the mean-field correction u⁰ - u_s, with u_s the solution to steady Navier-Stokes 
    # print("Uncomment following line if you have access to a real base flow measurement")
    steady_flow = load_data(base_path)
    uΔa = min_max_normalize(compute_mean_flow(full_data, 2400) - steady_flow[:,0])

    # Apply Gram-Schmidt to the POD modes + shift mode to build a new orhtonormal base
    Ur = np.column_stack((Ur, uΔa))  # Augmented spatial modes
    Ur = np.linalg.qr(Ur)[0]

    # Compute the dynamics from the projected full data and the modes
    A = np.dot(Ur.T, full_data - mean_flow[:, np.newaxis])

    # Return the modes from the POD and the shift mode, and the dynamics of these modes on the full data
    return (Ur, A)

full_data_path = 'E:/benchmark_SINDy/testRe100_h60.csv'
base_flow_path = 'E:/benchmark_SINDy/FullVorticityMapRe100_h60_baseFlow.csv'
n_modes = 9  # Number of POD modes to extract + shift mode
period_start = 2400 

# plt.figure()
# plt.imshow(np.rot90(compute_mean_flow(load_data(full_data_path), 2400).reshape(256, 128)), cmap='seismic', aspect='auto')
# plt.show()

U, A = POD_with_shift_mode(full_data_path, base_flow_path, n_modes, period_start)
U_test = U[:,:-1]
A_test = A[:-1, :]
a = np.loadtxt('fusionPython/data/vonKarman_pod/vonKarman_a.dat')

if (isinstance(U, str)):
    print(A)
else:
    plot_frames_and_signals(U, A)
    # flow_reconstruction(U, A)
    # flow_reconstruction(U_test, A_test)

# define energies of the DNS
E = np.sum(A.T ** 2, axis=1)
E_test = np.sum(A_test.T ** 2, axis=1)

# plot the energies
plt.figure(figsize=(16, 4))
plt.plot(E, 'r', label='POD')
plt.plot(E_test, 'b', label='POD without shift mode')

# do some formatting
plt.legend(fontsize=22, loc=2)
plt.grid()
ax = plt.gca()
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.ylabel('Total energy', fontsize=20)
plt.xlabel('t', fontsize=20)
plt.show()

# # Import libraries
# import warnings


# # ignore user warnings
# warnings.filterwarnings("ignore")

# rng = np.random.default_rng(1)

# from trapping_utils import (
#     integrator_keywords,
#     sindy_library_no_bias,
#     make_fits,
#     make_lissajou,
#     check_local_stability,
#     make_progress_plots,
# )

# # define parameters and load in POD modes obtained from DNS
# A1 = A[:, :3000]
# a[:, 1:] = A1.T
# t = a[:, 0]
# r = 5
# a_temp = a[:, 1:r]
# a_temp = np.hstack((a_temp, a[:, -1].reshape(3000, 1)))
# a = a_temp
# tbegin = 0
# tend = 3000
# skip = 1
# t = t[tbegin:tend:skip]
# a = a[tbegin:tend:skip, :]
# dt = t[1] - t[0]
# x_train = a
# x_test = a

# a0 = np.zeros(r)
# a0[0] = 1e-3

# # define hyperparameters
# max_iter = 1000000
# eta = 1.0e-2

# # don't need a reg_weight_lam if eta is sufficiently small
# # which is good news because CVXPY is much slower
# reg_weight_lam = 0
# alpha_m = 1e-1 * eta

# # run trapping SINDy
# sindy_opt = ps.TrappingSR3(
#     _n_tgts=5,
#     _include_bias=False,
#     reg_weight_lam=reg_weight_lam,
#     eta=eta,
#     alpha_m=alpha_m,
#     max_iter=max_iter,
#     verbose=True,
# )
# model = ps.SINDy(
#     optimizer=sindy_opt,
#     feature_library=sindy_library_no_bias,
#     differentiation_method=ps.FiniteDifference(drop_endpoints=True),
# )
# model.fit(x_train, t=t)
# Xi = model.coefficients().T
# xdot_test = model.differentiate(x_test, t=t)
# xdot_test_pred = model.predict(x_test)
# PL_tensor = sindy_opt.PL_
# PQ_tensor = sindy_opt.PQ_
# L = np.tensordot(PL_tensor, Xi, axes=([3, 2], [0, 1]))
# Q = np.tensordot(PQ_tensor, Xi, axes=([4, 3], [0, 1]))
# Q_sum = np.max(np.abs((Q + np.transpose(Q, [1, 2, 0]) + np.transpose(Q, [2, 0, 1]))))
# print("Max deviation from the constraints = ", Q_sum)
# if check_local_stability(Xi, sindy_opt, 1):
#     x_train_pred = model.simulate(x_train[0, :], t, integrator_kws=integrator_keywords)
#     x_test_pred = model.simulate(a0, t, integrator_kws=integrator_keywords)
#     make_progress_plots(r, sindy_opt)

#     # plotting and analysis
#     make_fits(r, t, xdot_test, xdot_test_pred, x_test, x_test_pred, "vonKarman")
#     make_lissajou(r, x_train, x_test, x_train_pred, x_test_pred, "VonKarman")
#     mean_val = np.mean(x_test_pred, axis=0)
#     mean_val = np.sqrt(np.sum(mean_val**2))
#     check_local_stability(Xi, sindy_opt, mean_val)
#     A_guess = sindy_opt.A_history_[-1]
#     m_guess = sindy_opt.m_history_[-1]
#     E_pred = np.linalg.norm(x_test - x_test_pred) / np.linalg.norm(x_test)
#     print("Frobenius Error = ", E_pred)
# make_progress_plots(r, sindy_opt)

# # Compute time-averaged dX/dt error
# deriv_error = np.zeros(xdot_test.shape[0])
# for i in range(xdot_test.shape[0]):
#     deriv_error[i] = np.dot(
#         xdot_test[i, :] - xdot_test_pred[i, :], xdot_test[i, :] - xdot_test_pred[i, :]
#     ) / np.dot(xdot_test[i, :], xdot_test[i, :])
# print("Time-averaged derivative error = ", np.nanmean(deriv_error))

# # define energies of the DNS, and both the 5D and 9D models
# # for POD-Galerkin and the trapping SINDy models
# E = np.sum(a**2, axis=1)
# E_sindy5 = np.sum(x_test_pred**2, axis=1)

# # plot the energies
# plt.figure(figsize=(16, 4))
# plt.plot(t, E, "r", label="DNS")
# plt.plot(t, E_sindy5, "k", label=r"SINDy-5")

# # do some formatting and save
# plt.legend(fontsize=22, loc=2)
# plt.grid()
# plt.xlim([0, 300])
# ax = plt.gca()
# ax.set_yticks([0, 10, 20])
# ax.tick_params(axis="x", labelsize=20)
# ax.tick_params(axis="y", labelsize=20)
# plt.ylabel("Total energy", fontsize=20)
# plt.xlabel("t", fontsize=20)
# plt.show()