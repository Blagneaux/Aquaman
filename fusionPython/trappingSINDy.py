# Import libraries
import warnings

import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

import pysindy as ps

# ignore user warnings
warnings.filterwarnings("ignore")

rng = np.random.default_rng(1)

from trapping_utils import (
    integrator_keywords,
    sindy_library_no_bias,
    make_fits,
    make_lissajou,
    check_local_stability,
    make_progress_plots,
    galerkin_model,
)

# define parameters and load in POD modes obtained from DNS
a = np.loadtxt("fusionPython/data/vonKarman_pod/vonKarman_a.dat")
t = a[:, 0]
r = 5
a_temp = a[:, 1:r]
a_temp = np.hstack((a_temp, a[:, -1].reshape(3000, 1)))
a = a_temp
tbegin = 0
tend = 3000
skip = 1
t = t[tbegin:tend:skip]
a = a[tbegin:tend:skip, :]
dt = t[1] - t[0]

# define the POD-Galerkin models from Noack (2003)
galerkin9 = sio.loadmat("fusionPython/data/vonKarman_pod/galerkin9.mat")

# make the Galerkin model nonlinearity exactly energy-preserving
# rather than just approximately energy-preserving
gQ = 0.5 * (galerkin9["Q"] + np.transpose(galerkin9["Q"], [0, 2, 1]))
galerkin9["Q"] = (
    gQ
    - (
        gQ
        + np.transpose(gQ, [1, 0, 2])
        + np.transpose(gQ, [2, 1, 0])
        + np.transpose(gQ, [0, 2, 1])
        + np.transpose(gQ, [2, 0, 1])
        + np.transpose(gQ, [1, 2, 0])
    )
    / 6.0
)

# time base for simulating Galerkin models
t_sim = np.arange(0, 500, dt)

# Generate initial condition from unstable eigenvectors
lamb, Phi = np.linalg.eig(galerkin9["L"])
idx = np.argsort(-np.real(lamb))
lamb, Phi = lamb[idx], Phi[:, idx]
a0 = np.zeros(9)
a0[0] = 1e-3
# np.real( 1e-3 * Phi[:, :2] @ rng.random((2)) )

# get the 5D POD-Galerkin coefficients
inds5 = np.ix_([0, 1, 2, 3, -1], [0, 1, 2, 3, -1])
galerkin5 = {}
galerkin5["L"] = galerkin9["L"][inds5]
inds5 = np.ix_([0, 1, 2, 3, -1], [0, 1, 2, 3, -1], [0, 1, 2, 3, -1])
galerkin5["Q"] = galerkin9["Q"][inds5]
model5 = lambda t, a: galerkin_model(a, galerkin5["L"], galerkin5["Q"])  # noqa: E731

# make the 3D, 5D, and 9D POD-Galerkin trajectories
t_span = (t[0], t[-1])
a_galerkin5 = solve_ivp(model5, t_span, a0[:5], t_eval=t, **integrator_keywords).y.T
adot_galerkin5 = np.gradient(a_galerkin5, axis=0) / (t[1] - t[0])

# plot the first 4 POD modes + the shift mode
mode_numbers = [0, 1, 2, 3, -1]
plt.figure(figsize=(12, 8))
for i in range(r):
    plt.subplot(r, 1, i + 1)
    if i == 0:
        plt.title(
            "DNS and POD-Galerkin models on first 4 POD modes + shift mode", fontsize=16
        )
    plt.plot(t, a[:, mode_numbers[i]], "r", label="POD from DNS")
    plt.plot(t, a_galerkin5[:, mode_numbers[i]], "b", label="POD-5 model")
    ax = plt.gca()
    plt.ylabel(r"$a_{" + str(mode_numbers[i]) + "}$", fontsize=20)
    plt.grid(True)
    if i == r - 1:
        plt.xlabel("t", fontsize=18)
        plt.legend(loc="upper left", fontsize=16)
    else:
        ax.set_xticklabels([])
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
a0 = np.zeros(r)
a0[0] = 1e-3

# same test and train trajectory for simplicity here
a = np.loadtxt("fusionPython/data/vonKarman_pod/vonKarman_a.dat")
t = a[:, 0]
r = 5
a_temp = a[:, 1:r]
a_temp = np.hstack((a_temp, a[:, -1].reshape(3000, 1)))
a = a_temp
tbegin = 0
tend = 3000
skip = 1
t = t[tbegin:tend:skip]
a = a[tbegin:tend:skip, :]
dt = t[1] - t[0]
x_train = a
x_test = a

# define hyperparameters
max_iter = 10000
eta = 1.0

# don't need a reg_weight_lam if eta is sufficiently small
# which is good news because CVXPY is much slower
reg_weight_lam = 0
alpha_m = 1e-1 * eta

# run trapping SINDy
sindy_opt = ps.TrappingSR3(
    _n_tgts=5,
    _include_bias=False,
    reg_weight_lam=reg_weight_lam,
    eta=eta,
    alpha_m=alpha_m,
    max_iter=max_iter,
    verbose=True,
)
model = ps.SINDy(
    optimizer=sindy_opt,
    feature_library=sindy_library_no_bias,
    differentiation_method=ps.FiniteDifference(drop_endpoints=True),
)
model.fit(x_train, t=t)
Xi = model.coefficients().T
xdot_test = model.differentiate(x_test, t=t)
xdot_test_pred = model.predict(x_test)
PL_tensor = sindy_opt.PL_
PQ_tensor = sindy_opt.PQ_
L = np.tensordot(PL_tensor, Xi, axes=([3, 2], [0, 1]))
Q = np.tensordot(PQ_tensor, Xi, axes=([4, 3], [0, 1]))
Q_sum = np.max(np.abs((Q + np.transpose(Q, [1, 2, 0]) + np.transpose(Q, [2, 0, 1]))))
print("Max deviation from the constraints = ", Q_sum)
if check_local_stability(Xi, sindy_opt, 1):
    x_train_pred = model.simulate(x_train[0, :], t, integrator_kws=integrator_keywords)
    x_test_pred = model.simulate(a0, t, integrator_kws=integrator_keywords)
    make_progress_plots(r, sindy_opt)

    # plotting and analysis
    make_fits(r, t, xdot_test, xdot_test_pred, x_test, x_test_pred, "vonKarman")
    make_lissajou(r, x_train, x_test, x_train_pred, x_test_pred, "VonKarman")
    mean_val = np.mean(x_test_pred, axis=0)
    mean_val = np.sqrt(np.sum(mean_val**2))
    check_local_stability(Xi, sindy_opt, mean_val)
    A_guess = sindy_opt.A_history_[-1]
    m_guess = sindy_opt.m_history_[-1]
    E_pred = np.linalg.norm(x_test - x_test_pred) / np.linalg.norm(x_test)
    print("Frobenius Error = ", E_pred)
make_progress_plots(r, sindy_opt)

# Compute time-averaged dX/dt error
deriv_error = np.zeros(xdot_test.shape[0])
for i in range(xdot_test.shape[0]):
    deriv_error[i] = np.dot(
        xdot_test[i, :] - xdot_test_pred[i, :], xdot_test[i, :] - xdot_test_pred[i, :]
    ) / np.dot(xdot_test[i, :], xdot_test[i, :])
print("Time-averaged derivative error = ", np.nanmean(deriv_error))

# define energies of the DNS, and both the 5D and 9D models
# for POD-Galerkin and the trapping SINDy models
E = np.sum(a**2, axis=1)
E_galerkin5 = np.sum(a_galerkin5**2, axis=1)
E_sindy5 = np.sum(x_test_pred**2, axis=1)

# plot the energies
plt.figure(figsize=(16, 4))
plt.plot(t, E, "r", label="DNS")
plt.plot(t, E_galerkin5, "m", label="POD-5")
plt.plot(t, E_sindy5, "k", label=r"SINDy-5")

# do some formatting and save
plt.legend(fontsize=22, loc=2)
plt.grid()
plt.xlim([0, 300])
ax = plt.gca()
ax.set_yticks([0, 10, 20])
ax.tick_params(axis="x", labelsize=20)
ax.tick_params(axis="y", labelsize=20)
plt.ylabel("Total energy", fontsize=20)
plt.xlabel("t", fontsize=20)
plt.show()