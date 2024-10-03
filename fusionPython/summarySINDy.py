# From https://github.com/dynamicslab/pysindy/blob/master/examples/15_pysindy_lectures.ipynb
# Sum up the youtube playlist tutorial on how to use pySINDy

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_error

from pysindy.utils import lorenz, lorenz_control
import pysindy as ps

# bad code but allows us to ignore warnings
import warnings
from scipy.integrate import ODEintWarning
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=ODEintWarning)

# Seed the random number generators for reproducibility
np.random.seed(100)

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12


# Make coefficient plot for threshold scan
def plot_pareto(coefs, opt, model, threshold_scan, x_test, t_test):
    dt = t_test[1] - t_test[0]
    mse = np.zeros(len(threshold_scan))
    mse_sim = np.zeros(len(threshold_scan))
    for i in range(len(threshold_scan)):
        opt.coef_ = coefs[i]
        mse[i] = model.score(x_test, t=dt, metric=mean_squared_error)
        x_test_sim = model.simulate(x_test[0, :], t_test, integrator="odeint")
        if np.any(x_test_sim > 1e4):
            x_test_sim = 1e4
        mse_sim[i] = np.sum((x_test - x_test_sim) ** 2)
    plt.figure()
    plt.semilogy(threshold_scan, mse, "bo")
    plt.semilogy(threshold_scan, mse, "b")
    plt.ylabel(r"$\dot{X}$ RMSE", fontsize=20)
    plt.xlabel(r"$\lambda$", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.figure()
    plt.semilogy(threshold_scan, mse_sim, "bo")
    plt.semilogy(threshold_scan, mse_sim, "b")
    plt.ylabel(r"$\dot{X}$ RMSE", fontsize=20)
    plt.xlabel(r"$\lambda$", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)


# Make plots of the data and its time derivative
def plot_data_and_derivative(x, dt, deriv):
    feature_name = ["x", "y", "z"]
    plt.figure(figsize=(20, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(x[:, i], label=feature_name[i])
        plt.grid(True)
        plt.xlabel("t", fontsize=24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=24)
    x_dot = deriv(x, t=dt)
    plt.figure(figsize=(20, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(x_dot[:, i], label=r"$\dot{" + feature_name[i] + "}$")
        plt.grid(True)
        plt.xlabel("t", fontsize=24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=24)


# Make an errorbar coefficient plot from the results of ensembling
def plot_ensemble_results(
    model, mean_ensemble, std_ensemble, mean_library_ensemble, std_library_ensemble
):
    # Plot results
    xticknames = model.get_feature_names()
    for i in range(10):
        xticknames[i] = "$" + xticknames[i] + "$"
    plt.figure(figsize=(18, 4))
    colors = ["b", "r", "k"]
    plt.subplot(1, 2, 1)
    plt.xlabel("Candidate terms", fontsize=22)
    plt.ylabel("Coefficient values", fontsize=22)
    for i in range(3):
        plt.errorbar(
            range(10),
            mean_ensemble[i, :],
            yerr=std_ensemble[i, :],
            fmt="o",
            color=colors[i],
            label=r"Equation for $\dot{" + feature_names[i] + r"}$",
        )
    ax = plt.gca()
    plt.grid(True)
    ax.set_xticks(range(10))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.set_xticklabels(xticknames, verticalalignment="top")
    plt.subplot(1, 2, 2)
    plt.xlabel("Candidate terms", fontsize=22)
    for i in range(3):
        plt.errorbar(
            range(10),
            mean_library_ensemble[i, :],
            yerr=std_library_ensemble[i, :],
            fmt="o",
            color=colors[i],
            label=r"Equation for $\dot{" + feature_names[i] + r"}$",
        )
    ax = plt.gca()
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=16, loc="upper right")
    ax.set_xticks(range(10))
    ax.set_xticklabels(xticknames, verticalalignment="top")


# Make energy-preserving quadratic constraints for model of size r
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
            constraint_matrix[
                q, r * (r + j - 1) + j + r * int(i * (2 * r - i - 3) / 2.0)
            ] = 1.0
            q = q + 1
    for i in range(r):
        for j in range(0, i):
            constraint_matrix[q, r * (N - r + j) + i] = 1.0
            constraint_matrix[
                q, r * (r + i - 1) + j + r * int(j * (2 * r - j - 3) / 2.0)
            ] = 1.0
            q = q + 1

    # Set coefficients adorning terms like a_ia_ja_k to be antisymmetric
    for i in range(r):
        for j in range(i + 1, r):
            for k in range(j + 1, r):
                constraint_matrix[
                    q, r * (r + k - 1) + i + r * int(j * (2 * r - j - 3) / 2.0)
                ] = 1.0
                constraint_matrix[
                    q, r * (r + k - 1) + j + r * int(i * (2 * r - i - 3) / 2.0)
                ] = 1.0
                constraint_matrix[
                    q, r * (r + j - 1) + k + r * int(i * (2 * r - i - 3) / 2.0)
                ] = 1.0
                q = q + 1

    return constraint_zeros, constraint_matrix


# For Trapping SINDy, use optimal m, and calculate if identified model is stable
def check_stability(r, Xi, sindy_opt):
    N = int((r ** 2 + 3 * r) / 2.0)
    opt_m = sindy_opt.m_history_[-1]
    PL_tensor_unsym = sindy_opt.PL_unsym_
    PL_tensor = sindy_opt.PL_
    PQ_tensor = sindy_opt.PQ_
    mPQ = np.tensordot(opt_m, PQ_tensor, axes=([0], [0]))
    P_tensor = PL_tensor - mPQ
    As = np.tensordot(P_tensor, Xi, axes=([3, 2], [0, 1]))
    eigvals, eigvecs = np.linalg.eigh(As)
    print("optimal m: ", opt_m)
    print("As eigvals: ", np.sort(eigvals))
    print(
        "All As eigenvalues are < 0 and therefore system is globally stable? ",
        np.all(eigvals < 0),
    )
    max_eigval = np.sort(eigvals)[-1]
    min_eigval = np.sort(eigvals)[0]
    L = np.tensordot(PL_tensor_unsym, Xi, axes=([3, 2], [0, 1]))
    Q = np.tensordot(PQ_tensor, Xi, axes=([4, 3], [0, 1]))
    d = np.dot(L, opt_m) + np.dot(np.tensordot(Q, opt_m, axes=([2], [0])), opt_m)
    Rm = np.linalg.norm(d) / np.abs(max_eigval)
    print("Estimate of trapping region size, Rm = ", Rm)


# Plot Kuramoto-Sivashinsky data and its derivative
def plot_u_and_u_dot(t, x, u):
    dt = t[1] - t[0]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(t, x, u[:, :, 0])
    plt.xlabel("t", fontsize=16)
    plt.ylabel("x", fontsize=16)
    plt.title(r"$u(x, t)$", fontsize=16)
    u_dot = ps.FiniteDifference(axis=1)._differentiate(u, t=dt)
    plt.subplot(1, 2, 2)
    plt.pcolormesh(t, x, u_dot[:, :, 0])
    plt.xlabel("t", fontsize=16)
    plt.ylabel("x", fontsize=16)
    ax = plt.gca()
    ax.set_yticklabels([])
    plt.title(r"$\dot{u}(x, t)$", fontsize=16)
    return u_dot


# Make 3d plots comparing a test trajectory, 
# an associated model trajectory, and a second model trajectory.
def make_3d_plots(x_test, x_sim, constrained_x_sim, last_label):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    plt.plot(
        x_test[:, 0],
        x_test[:, 1],
        x_test[:, 2],
        "k",
        label="Validation Lorenz trajectory",
    )
    plt.plot(x_sim[:, 0], x_sim[:, 1], x_sim[:, 2], "r", 
             label="SR3, no constraints")
    plt.plot(
        constrained_x_sim[:, 0],
        constrained_x_sim[:, 1],
        constrained_x_sim[:, 2],
        "b",
        label=last_label,
    )
    ax.set_xlabel("x", fontsize=20)
    ax.set_ylabel("y", fontsize=20)
    ax.set_zlabel("z", fontsize=20)
    plt.legend(fontsize=16, framealpha=1.0)

## -------------------------------------------------------------------------

# Part 1: How to choose algorithm hyperparameters such as lambda

## -------------------------------------------------------------------------
# define the testing and training Lorenz data we will use for these examples
dt = 0.002

t_train = np.arange(0, 10, dt)
x0_train = [-8, 8, 27]
t_train_span = (t_train[0], t_train[-1])
x_train = solve_ivp(
    lorenz, t_train_span, x0_train, t_eval=t_train, **integrator_keywords
).y.T

t_test = np.arange(0, 15, dt)
t_test_span = (t_test[0], t_test[-1])
x0_test = np.array([8, 7, 15])
x_test = solve_ivp(
    lorenz, t_test_span, x0_test, t_eval=t_test, **integrator_keywords
).y.T

# define the testing and training data for the Lorenz system with control
def u_fun(t):
    return np.column_stack([np.sin(2 * t), t ** 2])


x_train_control = solve_ivp(
    lorenz_control,
    t_train_span,
    x0_train,
    t_eval=t_train,
    args=(u_fun,),
    **integrator_keywords
).y.T
u_train_control = u_fun(t_train)
x_test_control = solve_ivp(
    lorenz_control,
    t_test_span,
    x0_test,
    t_eval=t_test,
    args=(u_fun,),
    **integrator_keywords
).y.T
u_test_control = u_fun(t_test)

# Instantiate and fit the SINDy model
feature_names = ['x', 'y', 'z']

print("--- Part 1 ---")

threshold_scan = np.linspace(0, 1.0, 10) # default threshold is lambda = 0.1
coefs = []
rmse = mean_squared_error(x_train, np.zeros(x_train.shape), squared=False)
x_train_added_noise = x_train + np.random.normal(0, rmse / 10.0,  x_train.shape)
for i, threshold in enumerate(threshold_scan):
    sparse_regression_optimizer = ps.STLSQ(threshold=threshold)
    model = ps.SINDy(feature_names=feature_names, 
                    optimizer=sparse_regression_optimizer)
    model.fit(x_train_added_noise, t=dt, quiet=True)
    coefs.append(model.coefficients())
    
plot_pareto(coefs, sparse_regression_optimizer, model, 
            threshold_scan, x_test, t_test)
    
# Primary conclusions of part 1
# ------------------------------
# A type of pareto curve, generated by scanning over lambda, produces a systematic way to choose 
# the level of sparsity in a system. Moreover, there are sharp increases as lambda increases. 
# There are physical dropoffs because these correspond to important terms being truncated off the model!

## -------------------------------------------------------------------------

# Part 2: How to make SINDy more robust for system identification?

## -------------------------------------------------------------------------
# Part 2a: Differentiate the data with method other than finite differences
print("--- Part 2a ---")

rmse = mean_squared_error(x_train, np.zeros(x_train.shape), squared=False)
x_train_added_noise = x_train 
plot_data_and_derivative(x_train_added_noise, dt, 
                        ps.FiniteDifference()._differentiate)

# 2% added noise
rmse = mean_squared_error(x_train, np.zeros(x_train.shape), squared=False)
x_train_added_noise = x_train + np.random.normal(0, rmse / 50.0,  x_train.shape)
plot_data_and_derivative(x_train_added_noise, dt, 
                        ps.FiniteDifference()._differentiate)

# Repeat but with smoothed finite differences!
plot_data_and_derivative(x_train_added_noise, dt, 
                        ps.SmoothedFiniteDifference()._differentiate)

# Primary conclusions of part 2a
# ------------------------------
# More sophisticated numerical differentiation can avoid some of the noise amplification 
# associated with finite differences. While we only look at using a simple SmoothedFiniteDifference 
# class, there are many other differentiatiors available in PySINDy. See the Example 5 Jupyter 
# notebook for more details.

# Part 2b: Simply add more data!
print("--- Part 2b ---")

sparse_regression_optimizer = ps.STLSQ()
model = ps.SINDy(feature_names=feature_names, optimizer=sparse_regression_optimizer)
model.fit(x_train_added_noise, t=dt)
model.print()

n_trajectories = 40
x0s = (np.random.rand(n_trajectories, 3) - 0.5) * 20
x_train_multi = []
for i in range(n_trajectories):
    x_train_temp = solve_ivp(
        lorenz, t_train_span, x0s[i], t_eval=t_train, **integrator_keywords
    ).y.T
    rmse = mean_squared_error(x_train_temp, 
                            np.zeros(x_train_temp.shape), 
                            squared=False)
    x_train_multi.append(
        x_train_temp + np.random.normal(0, rmse / 50.0, x_train_temp.shape)
    )

sparse_regression_optimizer = ps.STLSQ()
model = ps.SINDy(feature_names=feature_names, optimizer=sparse_regression_optimizer)
model.fit(x_train_multi, t=dt, multiple_trajectories=True)
model.print()

# Primary conclusions of part 2b
# ------------------------------
# Even when all the data is noisy, often model improvements are available by simply adding 
# more data to the regression.

# Part 2c: Use ensemble methods
# Fit a regular SINDy model with 5% added Gaussian noise

rmse = mean_squared_error(x_train, np.zeros(x_train.shape), squared=False)
x_train_added_noise = x_train + np.random.normal(0, rmse / 20.0,  x_train.shape)
sparse_regression_optimizer = ps.STLSQ(threshold=0.5)

print("--- Part 2c ---")

model = ps.SINDy(feature_names=feature_names, optimizer=sparse_regression_optimizer)
model.fit(x_train_added_noise, t=dt)
model.print()

# Fit many SINDy models with 5% added Gaussian noise
model = ps.SINDy(feature_names=feature_names, optimizer=sparse_regression_optimizer)
model.fit(x_train_added_noise, t=dt, ensemble=True, quiet=True)
ensemble_coefs = model.coef_list

# Get average and standard deviation of the ensemble model coefficients
mean_ensemble = np.mean(ensemble_coefs, axis=0)
std_ensemble = np.std(ensemble_coefs, axis=0)

# Now we sub-sample the library, generating multiple models. 
# The default sampling omits a single library term.
model = ps.SINDy(feature_names=feature_names, optimizer=sparse_regression_optimizer)
model.fit(x_train, t=dt, library_ensemble=True, quiet=True)
library_ensemble_coefs = model.coef_list

# Get average and standard deviation of the library ensemble model coefficients
mean_library_ensemble = np.mean(library_ensemble_coefs, axis=0)
std_library_ensemble = np.std(library_ensemble_coefs, axis=0)

plot_ensemble_results(
    model, 
    mean_ensemble, 
    std_ensemble, 
    mean_library_ensemble, 
    std_library_ensemble
)

# Primary conclusions of part 2c
# ------------------------------
# Ensembling and library ensembling techniques generate many models, allowing for probabilistic 
# system identification and significant robustness against noise in the data.

# Part 2d: use prior physical knowledge to constrain the model
print("--- Part 2d ---")

# Fit a regular SINDy model with 5% added Gaussian noise
sparse_regression_optimizer = ps.SR3(threshold=0.5)
model = ps.SINDy(feature_names=feature_names, optimizer=sparse_regression_optimizer)
model.fit(x_train_added_noise, t=dt)
print("SR3 model, no constraints:")
model.print()
x_sim = model.simulate(x0_test, t=t_test)

# Figure out how many library features there will be
library = ps.PolynomialLibrary()
library.fit(x_train)
n_features = library.n_output_features_

# Set constraints
n_targets = x_train.shape[1]
constraint_rhs = np.array([0, 28])

# One row per constraint, one column per coefficient
constraint_lhs = np.zeros((2, n_targets * n_features))

# 1 * (x0 coefficient) + 1 * (x1 coefficient) = 0
constraint_lhs[0, 1] = 1
constraint_lhs[0, 2] = 1

# 1 * (x0 coefficient) = 28
constraint_lhs[1, 1 + n_features] = 1

optimizer = ps.ConstrainedSR3(
    constraint_rhs=constraint_rhs,
    constraint_lhs=constraint_lhs,
    threshold=0.5,
    thresholder="l1",
)
model = ps.SINDy(
    optimizer=optimizer, feature_library=library, feature_names=feature_names
)
model.fit(x_train_added_noise, t=dt)
print("ConstrainedSR3 model, equality constraints:")
model.print()
constrained_x_sim = model.simulate(x0_test, t=t_test)
make_3d_plots(x_test, x_sim, constrained_x_sim, 
            "ConstrainedSR3, equality constraints")

# Repeat with inequality constraints
eps = 0.5
constraint_rhs = np.array([eps, eps, 28])

# One row per constraint, one column per coefficient
constraint_lhs = np.zeros((3, n_targets * n_features))

# 1 * (x0 coefficient) + 1 * (x1 coefficient) <= eps
constraint_lhs[0, 1] = 1
constraint_lhs[0, 2] = 1

# -eps <= 1 * (x0 coefficient) + 1 * (x1 coefficient)
constraint_lhs[1, 1] = -1
constraint_lhs[1, 2] = -1

# 1 * (x0 coefficient) <= 28
constraint_lhs[2, 1 + n_features] = 1

print("ConstrainedSR3 wasnt working. Changed to TrappingSR3")
optimizer = ps.TrappingSR3(
    constraint_rhs=constraint_rhs,
    constraint_lhs=constraint_lhs,
    threshold=0.5,
    inequality_constraints=True,
    thresholder="l1",
)
model = ps.SINDy(
    optimizer=optimizer, feature_library=library, feature_names=feature_names
)
model.fit(x_train_added_noise, t=dt)
print("ConstrainedSR3 model, inequality constraints:")
model.print()
constrained_x_sim = model.simulate(x0_test, t=t_test)
make_3d_plots(
    x_test, x_sim, constrained_x_sim, 
    "ConstrainedSR3, inequality constraints"
)

# Primary conclusions of part 2d
# ------------------------------
# Physical priors can be built into data-driven models via equality or inequality constraints. 
# This restricts the possible solutions, which can improve robustness against noise. 
# However, there are some pitfalls, such as above where we restrict the coefficients of x and y 
# to be equal and opposite with equality constraints. Inequality constraints are often more suitable.

# Part 2e (advanced): use trapping SINDy for globally stable models
print("--- Part 2e ---")
# define hyperparameters
threshold = 0
max_iter = 20000
eta = 1.0e-2
constraint_zeros, constraint_matrix = make_constraints(3)

# run trapping SINDy
sindy_opt = ps.TrappingSR3(
    threshold=threshold,
    eta=eta,
    gamma=-1,
    max_iter=max_iter,
    constraint_lhs=constraint_matrix,
    constraint_rhs=constraint_zeros,
    constraint_order="feature",
)

# Initialize quadratic SINDy library, with custom ordering
library_functions = [lambda x: x, lambda x, y: x * y, lambda x: x ** 2]
library_function_names = [lambda x: x, lambda x, y: x + y, lambda x: x + x]
sindy_library = ps.CustomLibrary(
    library_functions=library_functions, 
    function_names=library_function_names
)

model = ps.SINDy(
    optimizer=sindy_opt,
    feature_library=sindy_library,
)
model.fit(x_train, t=dt, quiet=True)
model.print()

Xi = model.coefficients().T
check_stability(3, Xi, sindy_opt)

# show that new model trajectories are all stable
fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(111, projection="3d")
for i in range(10):
    x0_new = (np.random.rand(3) - 0.5) * 200
    x_test_new = solve_ivp(
        lorenz, t_test_span, x0_new, t_eval=t_test, **integrator_keywords
    ).y.T
    ax.plot(x_test_new[:, 0], x_test_new[:, 1], x_test_new[:, 2], "k")
    x_test_pred_new = model.simulate(x0_new, t=t_test, integrator="odeint")
    plt.plot(x_test_pred_new[:, 0], x_test_pred_new[:, 1], 
            x_test_pred_new[:, 2], "b")
    ax.set_xlabel("x", fontsize=20)
    ax.set_ylabel("y", fontsize=20)
    ax.set_zlabel("z", fontsize=20)
    plt.legend(
        ["Validation Lorenz trajectory", "TrappingSR3"], 
        fontsize=16, framealpha=1.0
    )

# Primary conclusions of part 2e
# ------------------------------
# Trapping SINDy can provide models that are provably globally stable for any new initial condition. 
# However, the system must have energy-preserving quadratic nonlinearities. These types of systems 
# are common in fluid and plasma flows.

# Last thing we haven't covered here... trimming outliers can really help for some noisy datasets. 
# The ConstrainedSR3 optimizer allows for automated trimming, but there are many methods for 
# pre-processing data to remove outliers before solving the SINDy optimization.

# Part 2f (advanced): use the weak formulation of SINDy
print("--- Part 2f ---")

ode_lib = ps.WeakPDELibrary(
    library_functions=library_functions,
    function_names=library_function_names,
    spatiotemporal_grid=t_train,
    include_bias=True,
)
rmse = mean_squared_error(x_train, np.zeros(x_train.shape), squared=False)
x_train_added_noise = x_train + np.random.normal(0, rmse / 10.0, x_train.shape)

# Fit a normal SINDy model
optimizer = ps.STLSQ()
model = ps.SINDy(feature_names=feature_names, optimizer=optimizer)
model.fit(x_train_added_noise, t=dt, ensemble=True)

print("Normal SINDy result on 10% Lorenz noise: ")
model.print()
regular_models = model.coef_list
regular_mean = np.mean(regular_models, axis=0)
regular_std = np.std(regular_models, axis=0)

# Instantiate and fit a weak formulation SINDy model
optimizer = ps.STLSQ()
model = ps.SINDy(
    feature_library=ode_lib, 
    feature_names=feature_names, 
    optimizer=optimizer
)
model.fit(x_train_added_noise, t=dt, ensemble=True)
print("Weak form result on 10% Lorenz noise: ")
model.print()
weak_form_models = model.coef_list
weak_form_mean = np.mean(weak_form_models, axis=0)
weak_form_std = np.std(weak_form_models, axis=0)

plot_ensemble_results(model, regular_mean, regular_std, 
                    weak_form_mean, weak_form_std)

# Primary conclusions of part 2f
# ------------------------------
# The weak formulation of SINDy drastically improves the quality of the models when noise is present, 
# and can be used with ensembling methods for extra robustness against high noise levels.

## -------------------------------------------------------------------------

# Part 3: Implicit ODEs
# Skipping part a because Navier-Stoke is not an ODE

## -------------------------------------------------------------------------
# Part 3b: Identifying partial differential equations (PDEs)
print("--- Part 3b ---")

from scipy.io import loadmat

# Load data from .mat file
data = loadmat('C:/Users/blagn771/Downloads/kuramoto_sivishinky.mat')
t = np.ravel(data['tt'])
dt = t[1] - t[0]
x = np.ravel(data['x'])
u = data['uu']
u = u.reshape(len(x), len(t), 1)
u_dot = plot_u_and_u_dot(t, x, u)

# Define PDE library that is quadratic in u, 
# and fourth-order in spatial derivatives of u.
library_functions = [lambda x: x, lambda x: x * x]
library_function_names = [lambda x: x, lambda x: x + x]
pde_lib = ps.PDELibrary(
    library_functions=library_functions,
    function_names=library_function_names,
    derivative_order=4,
    spatial_grid=x,
    is_uniform=True,
)

# Again, loop through all the optimizers
print('STLSQ model: ')
optimizer = ps.STLSQ(threshold=30, normalize_columns=True)
model = ps.SINDy(feature_library=pde_lib, 
                 feature_names=['u'], 
                 optimizer=optimizer)
model.fit(u, t=dt)
model.print()

plt.show()

# Primary conclusions of part 3b

# PDE identification can be done straightforwardly with PDELibrary. The primary changes are 
# flattening the data, passing x_dot to model.fit, and defining a spatial grid for the PDE.

## -------------------------------------------------------------------------

# Part 4: How to choose a regularizer and a sparse regression algorithm?

## -------------------------------------------------------------------------
# Part 4a: Choosing an algorithm to solve the l0 problem

# Primary conclusions of part 4a

# The STLSQ and SSR optimizers are more robust against noisy data than the FROLs, OMP, or LARS 
# optimizers because the latter algorithms rely on choosing terms by computing correlations 
# with the target data. SSR should be used if avoiding the user wants to avoid hyperparameters 
# or if the user wants a specific technique for chopping off coefficients at each algorithm iteration. 
# For instance, STLSQ also zeros out the coefficients below the threshold value, and then refits 
# the coefficients. SSR can zero out coefficients with more complex criteria.

# Part 4b: Choosing an algorithm to solve the l1 problem

# Primary conclusions of part 4b

# All the optimizers work pretty well, but the l1 norm tends to produce a few very small "extra" 
# terms that are hard to truncate unless the hyperparameter lambda is tuned. The PySINDy SR3 
# optimizer works very well, can be easily updated to allow for some l2 norm regularization, 
# has robust convergence guarantees, and allows for equality and inequality constraints, so we 
# strongly recommend the use of this optimizer with the l1 norm. However, depending on the 
# dynamical system, one may also need to tune the additional hyperparameter nu (usually nu = 1 is fine),
#  and this is the primary downside of this method.

## -------------------------------------------------------------------------

# Part 5: How to build complex candidate libraries

## -------------------------------------------------------------------------