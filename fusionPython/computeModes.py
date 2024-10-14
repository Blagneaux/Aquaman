import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pysindyOld as ps 

def load_data(path):
    """Load data from a CSV file"""
    data = pd.read_csv(path, header=None)
    snapshots = data.values  # Convert dataframe into numpy array
    return snapshots

def compute_mean_flow(snapshots):
    """Compute the mean flow by averaging the flow snapshots over time"""
    mean_flow = np.mean(snapshots, axis=1)
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
        plt.show()

import matplotlib.animation as animation

def flow_reconstruction(frames, signals):
    steps = signals.shape[1]
    fig = plt.figure()  # Initialize the figure outside the loop

    real_frames = load_data(full_data_path)

    img = []

    for i in range(steps):
        frame = frames @ signals[:, i]
        frame = np.vstack([frame, real_frames[:, i]])
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

    # Compute the mean flow
    mean_flow = compute_mean_flow(full_data)

    # Estimate base flow from full flow by manually selecting the frame before the vortex shedding
    steady_flow = full_data[:, 249]

    # Apply the POD to the mean-substracted flow on a single period of vortex shedding
    if period_start == None:
        period_cutoff = find_matching_column(full_data)
        if period_cutoff == None:
            return ("Error: ", "Flow has no periodic vortex shedding")
    else:
        period_cutoff = period_start
    cut_data = full_data[:, period_cutoff:] - mean_flow[:, np.newaxis]
    U, S, VT = np.linalg.svd(cut_data, False)

    # Select the first n_modes - 1
    Ur, Sr, VTr = U[:, :n_modes-1], S[:n_modes-1], VT[:n_modes-1, :]
    Sr = np.diag(Sr)
    Ar = Sr @ VTr  # Flow is approximated to mean_flow + Ur @ Ar, ie u(x,t) ≈ uⁿ = u⁰(x) + Σ[u_i(x).a_i(t)]

    # Compute the mean-field correction u⁰ - u_s, with u_s the solution to steady Navier-Stokes 
    print("Uncomment following line if you have access to a real base flow measurement")
    # steady_flow = load_data(base_path)
    uΔa = mean_flow - steady_flow

    # Apply Gram-Schmidt to the POD modes + shift mode to build a new orhtonormal base
    inner_products = [np.inner(uΔa,Ur[:,i]) for i in range(n_modes-1)]
    if not(isinstance(inner_products[0], float)) and not(isinstance(inner_products[0], int)):
        print("Is float: ", isinstance(inner_products[0], float))
        print("Is int: ", isinstance(inner_products[0], int))
        return ("Error: ", "Problem with the computation of the inner product in Gram-Schmidt")
    Ur_temp = Ur
    for i, elm in enumerate(inner_products):
        Ur_temp[:,i] = elm*(Ur_temp[:, i])
    uΔb = uΔa - np.sum(Ur_temp, axis=1)
    uΔ = uΔb / np.sqrt(np.inner(uΔb, uΔb))  # u(x,t) ≈ uⁿ⁺¹ = Σ[u_i(x).a_i(t)] with uΔ the n+1th mode
    Ur = np.column_stack((Ur, uΔ))  # Augmented spatial modes

    # Normalize the POD modes
    for i in range(len(Ur[0])):
        Ur[:, i] = Ur[:, i] / np.sqrt(np.inner(Ur[:, i], Ur[:, i]))

    # Compute the dynamic of the new mode uΔ: a_i(t) = (u-u⁰, uΔ) [wont be useful but still interesting]
    aΔ = np.array([np.inner(cut_data[:, i], uΔ) for i in range(len(cut_data[0]))])
    Ar = np.vstack([Ar, aΔ])  # Augmented temporal modes

    # Project the full data on the new orthonormal base 
    projU = Ur @ np.linalg.inv(Ur.T @ Ur) @ Ur.T
    projected = projU @ min_max_normalize(full_data)

    # Compute the dynamics from the projected full data and the modes
    A = np.linalg.pinv(Ur) @ projected

    # Return the modes from the POD and the shift mode, and the dynamics of these modes on the full data
    return (Ur, A)

full_data_path = 'E:/benchmark_SINDy/FullVorticityMapRe100_h60.csv'
base_flow_path = 'E:/benchmark_SINDy/FullVorticityMapRe100_h60_baseFlow.csv'
n_modes = 9  # Number of POD modes to extract + shift mode
period_start = 1400

U, A = POD_with_shift_mode(full_data_path, base_flow_path, n_modes, period_start)
# U_test, A_test = POD_with_shift_mode(full_data_path, base_flow_path, n_modes+1, period_start)
# U_test = U[:,:8]
# A_test = A[:8, :]
a = np.loadtxt('fusionPython/data/vonKarman_pod/vonKarman_a.dat')

if (isinstance(U, str)):
    print(A)
else:
    print(len(U), len(U[0]), len(A), len(A[0]))
    # plot_frames_and_signals(U, A)
    flow_reconstruction(U, A)

# # --------------------------------------------------- Test with SINDy -----------------
# # Build the energy-preserving quadratic nonlinearity constraints
# def make_constraints(r):
#     q = 0
#     N = int((r ** 2 + 3 * r) / 2.0)
#     p = r + r * (r - 1) + int(r * (r - 1) * (r - 2) / 6.0)
#     constraint_zeros = np.zeros(p)
#     constraint_matrix = np.zeros((p, r * N))    
    
#     # Set coefficients adorning terms like a_i^3 to zero
#     for i in range(r):
#         constraint_matrix[q, r * (N - r) + i * (r + 1)] = 1.0
#         q = q + 1

#     # Set coefficients adorning terms like a_ia_j^2 to be antisymmetric
#     for i in range(r):
#         for j in range(i + 1, r):
#             constraint_matrix[q, r * (N - r + j) + i] = 1.0
#             constraint_matrix[q, r * (r + j - 1) + j + r * int(i * (2 * r - i - 3) / 2.0)] = 1.0
#             q = q + 1
#     for i in range(r):
#          for j in range(0, i):
#             constraint_matrix[q, r * (N - r + j) + i] = 1.0
#             constraint_matrix[q, r * (r + i - 1) + j + r * int(j * (2 * r - j - 3) / 2.0)] = 1.0
#             q = q + 1

#     # Set coefficients adorning terms like a_ia_ja_k to be antisymmetric
#     for i in range(r):
#         for j in range(i + 1, r):
#             for k in range(j + 1, r):
#                 constraint_matrix[q, r * (r + k - 1) + i + r * int(j * (2 * r - j - 3) / 2.0)] = 1.0
#                 constraint_matrix[q, r * (r + k - 1) + j + r * int(i * (2 * r - i - 3) / 2.0)] = 1.0
#                 constraint_matrix[q, r * (r + j - 1) + k + r * int(i * (2 * r - i - 3) / 2.0)] = 1.0
#                 q = q + 1
                
#     return constraint_zeros, constraint_matrix

# # Initialize quadratic SINDy library, with custom ordering 
# # to be consistent with the constraint
# library_functions = [lambda x:x, lambda x, y:x * y, lambda x:x ** 2]
# library_function_names = [lambda x:x, lambda x, y:x + y, lambda x:x + x]
# sindy_library = ps.CustomLibrary(library_functions=library_functions, 
#                                  function_names=library_function_names)

# a = A.T
# pod_modes_with_deformation = a
# t_custom = np.arange(0, 0.6194 * pod_modes_with_deformation.size, 0.6194)
# r = 9
# if r < 9:
#     a_temp = pod_modes_with_deformation
#     a_temp = np.hstack((a_temp, pod_modes_with_deformation[:, -1].reshape(1676, 1)))
#     a_custom = a_temp
# else:
#     a_custom = pod_modes_with_deformation
# tbegin = 0
# tend = 1676
# skip = 1
# t_custom = t_custom[tbegin:tend:skip]
# a_custom = a_custom[tbegin:tend:skip, :]
# x_test9_custom = a_custom
# x_train9_custom = a_custom
# dt_custom = t_custom[1] - t_custom[0]
# a0_custom = np.zeros(9)
# a0_custom[0] = 1e-3

# # initial guesses to speed up convergence
# A_guess = np.asarray([[-1.00000000e+00, -6.81701156e-18,  1.64265568e-17,
#          1.68310896e-17, -1.46247179e-17, -1.45278659e-17,
#         -4.41460518e-17, -9.75820242e-18,  1.06635090e-18],
#        [ 2.11962730e-17, -1.00000000e+00,  5.91916098e-17,
#          1.02939874e-16,  1.07502638e-17,  1.16153337e-17,
#          8.84561388e-17, -2.38407466e-17,  2.25533310e-17],
#        [ 7.70290614e-18, -2.37493643e-17, -1.00000000e+00,
#          5.74638700e-17, -2.17428595e-18,  2.41426251e-17,
#          6.89329350e-18, -3.65633938e-18,  7.79059612e-17],
#        [-2.44656476e-17, -6.72113002e-18,  4.74779569e-17,
#         -1.00000000e+00, -8.10241368e-18,  6.23271318e-18,
#          1.12682516e-18,  3.01538601e-17,  2.94041671e-16],
#        [-1.29102703e-16,  1.28776608e-17,  4.48182893e-17,
#         -7.15006179e-19, -1.00000000e+00,  4.11221110e-18,
#         -3.33172879e-16, -4.22913612e-17, -5.88841351e-17],
#        [ 7.10726824e-18,  9.55210532e-18, -5.30624590e-17,
#         -1.99630356e-17, -4.88598954e-18, -1.00000000e+00,
#         -1.72238787e-17,  1.45840342e-17, -1.29732583e-17],
#        [-1.01481442e-17,  4.78393464e-17, -2.53411865e-17,
#          1.31394318e-17, -5.96906289e-18,  1.68124806e-18,
#         -1.00000000e+00, -1.51574587e-17, -2.15989255e-18],
#        [-9.48456158e-18, -5.41527305e-18, -3.05384371e-19,
#         -1.99553156e-18,  8.37718659e-17,  6.05188865e-17,
#          3.94017953e-17, -1.00000000e+00, -1.69209548e-18],
#        [ 2.85170680e-18,  2.40387704e-17,  8.14566833e-17,
#          2.74548940e-17, -4.62236236e-18, -7.34952555e-18,
#          4.64207566e-18,  1.69214151e-18, -1.00000000e+00]])
# m_guess = np.asarray([-1.03179775e+00, -3.31216061e-01, -6.71172384e-01, 
#                       -6.75994493e-01, -8.50522236e-03, -2.12185379e-02, 
#                       -9.16401064e-04,  1.03271372e-03, 4.70226212e-01])

# # define hyperparameters
# max_iter = 1
# eta = 1.0
# threshold = 1
# alpha_m = 5e-1 * eta

# constraint_zeros, constraint_matrix = make_constraints(r)

# # run trapping SINDy
# sindy_opt = ps.TrappingSR3(threshold=threshold, eta=eta, alpha_m=alpha_m,
#                            m0=m_guess, A0=A_guess, max_iter=max_iter, 
#                            constraint_lhs=constraint_matrix,
#                            constraint_rhs=constraint_zeros,
#                            constraint_order="feature",
#                            )
# model = ps.SINDy(
#     optimizer=sindy_opt,
#     feature_library=sindy_library,
#     differentiation_method=ps.FiniteDifference(drop_endpoints=True),
# )
# print(x_test9_custom.shape[-2], len(t_custom))
# model.fit(x_train9_custom, t=t_custom)
# Xi_custom = model.coefficients().T
# xdot_test9_custom = model.differentiate(x_test9_custom, t=t_custom)
# xdot_test_pred9_custom = model.predict(x_test9_custom)
# PL_tensor_custom = sindy_opt.PL_
# PQ_tensor_custom = sindy_opt.PQ_
# L_custom = np.tensordot(PL_tensor_custom, Xi_custom, axes=([3, 2], [0, 1]))
# Q_custom = np.tensordot(PQ_tensor_custom, Xi_custom, axes=([4, 3], [0, 1]))
# x_train_pred9_custom = model.simulate(x_train9_custom[0, :], t_custom)
# x_test_pred9_custom = model.simulate(a0_custom, t_custom)

# define energies of the DNS
E = np.sum(A.T ** 2, axis=1)
# E_test = np.sum(A_test.T ** 2, axis=1)
# E_sindy9_custom = np.sum(x_test_pred9_custom ** 2, axis=1)

# plot the energies
plt.figure(figsize=(16, 4))
plt.plot(E, 'r', label='DNS')
# plt.plot(E_test, 'bo', label='DNS without shift mode')
# plt.plot(t_custom, E_sindy9_custom, 'o', label=r'SINDy-9')

# do some formatting
plt.legend(fontsize=22, loc=2)
plt.grid()
# plt.xlim([0, 300])
ax = plt.gca()
# ax.set_yticks([0, 10, 20])
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.ylabel('Total energy', fontsize=20)
plt.xlabel('t', fontsize=20)
plt.show()