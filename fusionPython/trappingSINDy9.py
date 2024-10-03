# Import libraries. Note that the neksuite (pymech)
# package is required for visualization of the 
# vortex shedding example
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pysindyOld as ps 
from scipy.integrate import solve_ivp
import matplotlib.gridspec as gridspec
import scipy.io as sio
from pysindyOld.utils import meanfield
from pysindyOld.utils import oscillator
from pysindyOld.utils import lorenz
from pysindyOld.utils import mhd
from pysindyOld.utils import burgers_galerkin

# ignore user warnings
import warnings
warnings.filterwarnings("ignore")

# Comment out below lines if not doing the vortex shedding
import sys
sys.path.append('fusionPython/data/vonKarman_pod/')
import pymech.neksuite as nek
from scipy.interpolate import griddata


# Define some setup and plotting functions
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
    
    
# Use optimal m, and calculate eigenvalues(PW) to see if identified model is stable
def check_stability(r, Xi, sindy_opt, mean_val):
    N = int((r ** 2 + 3 * r) / 2.0)
    opt_m = sindy_opt.m_history_[-1]
    PL_tensor_unsym = sindy_opt.PL_unsym_
    PL_tensor = sindy_opt.PL_
    PQ_tensor = sindy_opt.PQ_
    mPQ = np.tensordot(opt_m, PQ_tensor, axes=([0], [0]))
    P_tensor = PL_tensor - mPQ
    As = np.tensordot(P_tensor, Xi, axes=([3, 2], [0, 1]))
    eigvals, eigvecs = np.linalg.eigh(As)
    print('optimal m: ', opt_m)
    print('As eigvals: ', np.sort(eigvals))
    max_eigval = np.sort(eigvals)[-1]
    min_eigval = np.sort(eigvals)[0]
    L = np.tensordot(PL_tensor_unsym, Xi, axes=([3, 2], [0, 1]))
    Q = np.tensordot(PQ_tensor, Xi, axes=([4, 3], [0, 1]))
    d = np.dot(L, opt_m) + np.dot(np.tensordot(Q, opt_m, axes=([2], [0])), opt_m)
    Rm = np.linalg.norm(d) / np.abs(max_eigval)
    Reff = Rm / mean_val
    print('Estimate of trapping region size, Rm = ', Rm)
    print('Normalized trapping region size, Reff = ', Reff)
    

# Plot the SINDy trajectory, trapping region, and ellipsoid where Kdot >= 0
def trapping_region(r, x_test_pred, Xi, sindy_opt, filename):
    
    # Need to compute A^S from the optimal m obtained from SINDy algorithm
    N = int((r ** 2 + 3 * r) / 2.0)
    opt_m = sindy_opt.m_history_[-1]
    PL_tensor_unsym = sindy_opt.PL_unsym_
    PL_tensor = sindy_opt.PL_
    PQ_tensor = sindy_opt.PQ_
    mPQ = np.tensordot(opt_m, PQ_tensor, axes=([0], [0]))
    P_tensor = PL_tensor - mPQ
    As = np.tensordot(P_tensor, Xi, axes=([3, 2], [0, 1]))
    eigvals, eigvecs = np.linalg.eigh(As)
    print('optimal m: ', opt_m)
    print('As eigvals: ', eigvals)

    # Extract maximum and minimum eigenvalues, and compute radius of the trapping region
    max_eigval = np.sort(eigvals)[-1]
    min_eigval = np.sort(eigvals)[0]

    # Should be using the unsymmetrized L
    L = np.tensordot(PL_tensor_unsym, Xi, axes=([3, 2], [0, 1]))
    Q = np.tensordot(PQ_tensor, Xi, axes=([4, 3], [0, 1]))
    d = np.dot(L, opt_m) + np.dot(np.tensordot(Q, opt_m, axes=([2], [0])), opt_m)
    Rm = np.linalg.norm(d) / np.abs(max_eigval)

    # Make 3D plot illustrating the trapping region
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(8, 8))
    Y = np.zeros(x_test_pred.shape)
    Y = x_test_pred - opt_m * np.ones(x_test_pred.shape)
        
    Y = np.dot(eigvecs, Y.T).T
    plt.plot(Y[:, 0], Y[:, 1], Y[:, -1], 'k', 
             label='SINDy model prediction with new initial condition', 
             alpha=1.0, linewidth=3)
    h = np.dot(eigvecs, d)

    alpha = np.zeros(r)
    for i in range(r):
        if filename == 'Von Karman' and (i == 2 or i == 3):
            h[i] = 0
        alpha[i] = np.sqrt(0.5) * np.sqrt(np.sum(h ** 2 / eigvals) / eigvals[i])

    shift_orig = h / (4.0 * eigvals)

    # draw sphere in eigencoordinate space, centered at 0
    u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
    x = Rm * np.cos(u) * np.sin(v)
    y = Rm * np.sin(u) * np.sin(v)
    z = Rm * np.cos(v)
    
    ax.plot_wireframe(x, y, z, color="b",
                      label=r'Trapping region estimate, $B(m, R_m)$', 
                      alpha=0.5, linewidth=0.5)
    ax.plot_surface(x, y, z, color="b", alpha=0.05)
    ax.view_init(elev=0., azim=30)

    # define ellipsoid
    rx, ry, rz = np.asarray([alpha[0], alpha[1], alpha[-1]])

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Add this piece so we can compare with the analytic Lorenz ellipsoid,
    # which is typically defined only with a shift in the "y" direction here.
    if filename == 'Lorenz Attractor':
        shift_orig[0] = 0
        shift_orig[-1] = 0
    x = rx * np.outer(np.cos(u), np.sin(v)) - shift_orig[0]
    y = ry * np.outer(np.sin(u), np.sin(v)) - shift_orig[1]
    z = rz * np.outer(np.ones_like(u), np.cos(v)) - shift_orig[-1]

    # Plot ellipsoid
    ax.plot_wireframe(x, y, z, rstride=5, cstride=5, color='r', 
                      label='Ellipsoid of positive energy growth', 
                      alpha=1.0, linewidth=0.5)
    
    if filename == 'Lorenz Attractor':
        sigma = 10.0
        rho = 28.0
        beta = 8.0 / 3.0
    
        # define analytic ellipsoid in original Lorenz state space
        rx, ry, rz = [np.sqrt(beta * rho), np.sqrt(beta * rho ** 2), rho]

        # Set of all spherical angles:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        # ellipsoid in (x, y, z) coordinate to -> shifted by m
        x = rx * np.outer(np.cos(u), np.sin(v)) - opt_m[0]
        y = ry * np.outer(np.sin(u), np.sin(v)) - opt_m[1]
        z = rz * np.outer(np.ones_like(u), np.cos(v)) + rho - opt_m[-1]

        # Transform into eigencoordinate space
        xyz = np.tensordot(eigvecs, np.asarray([x, y, z]), axes=[1, 0])
        x = xyz[0, :, :]
        y = xyz[1, :, :]
        z = xyz[2, :, :]

        # Plot ellipsoid
        ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color='g', 
                          label=r'Lorenz analytic ellipsoid', 
                          alpha=1.0, linewidth=1.5)
    
    # Adjust plot features and save
    plt.legend(fontsize=16, loc='upper left')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_axis_off()
    plt.show()


# Plot errors between m_{k+1} and m_k and similarly for the model coefficients
def make_progress_plots(r, sindy_opt):
    W = np.asarray(sindy_opt.history_)
    M = np.asarray(sindy_opt.m_history_)
    dW = np.zeros(W.shape[0])
    dM = np.zeros(M.shape[0])
    for i in range(1,W.shape[0]):
        dW[i] = np.sum((W[i, :, :] - W[i - 1, :, :]) ** 2)
        dM[i] = np.sum((M[i, :] - M[i - 1, :]) ** 2)
    plt.figure()
    print(dW.shape, dM.shape)
    plt.semilogy(dW, label=r'Coefficient progress, $\|\xi_{k+1} - \xi_k\|_2^2$')
    plt.semilogy(dM, label=r'Vector m progress, $\|m_{k+1} - m_k\|_2^2$')
    plt.xlabel('Algorithm iterations', fontsize=16)
    plt.ylabel('Errors', fontsize=16)
    plt.legend(fontsize=14)
    PWeigs = np.asarray(sindy_opt.PWeigs_history_)
    plt.figure()
    for j in range(r):
        if np.all(PWeigs[:, j] > 0.0):
            plt.semilogy(PWeigs[:, j], 
                         label=r'diag($P\xi)_{' + str(j) + str(j) + '}$')
        else:
            plt.plot(PWeigs[:, j], 
                     label=r'diag($P\xi)_{' + str(j) + str(j) + '}$')
        plt.xlabel('Algorithm iterations', fontsize=16)
        plt.legend(fontsize=12)
        plt.ylabel(r'Eigenvalues of $P\xi$', fontsize=16)
    

# Plot first three modes in 3D for ground truth and SINDy prediction
def make_3d_plots(x_test, x_test_pred, filename):
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(8, 8))
    if filename == 'VonKarman':
        ind = -1
    else:
        ind = 2
    plt.plot(x_test[:, 0], x_test[:, 1], x_test[:, ind], 
             'r', label='true x')
    plt.plot(x_test_pred[:, 0], x_test_pred[:, 1], x_test_pred[:, ind], 
             'k', label='pred x')
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_axis_off()
    plt.legend(fontsize=14)
    plt.show()


# Plot the SINDy fits of X and Xdot against the ground truth
def make_fits(r, t, xdot_test, xdot_test_pred, x_test, x_test_pred, filename):
    fig = plt.figure(figsize=(16, 8))
    spec = gridspec.GridSpec(ncols=2, nrows=r, figure=fig, hspace=0.0, wspace=0.0)
    for i in range(r):
        plt.subplot(spec[i, 0]) #r, 2, 2 * i + 2)
        plt.plot(t, xdot_test[:, i], 'r', 
                 label=r'true $\dot{x}_' + str(i) + '$')
        plt.plot(t, xdot_test_pred[:, i], 'k--', 
                 label=r'pred $\dot{x}_' + str(i) + '$')
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.legend(fontsize=12)
        if i == r - 1:
            plt.xlabel('t', fontsize=18)
        plt.subplot(spec[i, 1])
        plt.plot(t, x_test[:, i], 'r', label=r'true $x_' + str(i) + '$')
        plt.plot(t, x_test_pred[:, i], 'k--', label=r'pred $x_' + str(i) + '$')
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.legend(fontsize=12)
        if i == r - 1:
            plt.xlabel('t', fontsize=18)
        
    plt.show()
    

# Make a bar plot of the distribution of SINDy coefficients
# and distribution of Galerkin coefficients for the von Karman street
def make_bar(galerkin9, L, Q):
    bins = np.logspace(-11, 0, 50)
    plt.figure(figsize=(8, 4))
    plt.grid('True')
    galerkin_full = np.vstack((galerkin9['L'].reshape(r ** 2, 1), 
                               galerkin9['Q'].reshape(len(galerkin9['Q'].flatten()), 1)))
    plt.hist(np.abs(galerkin_full), bins=bins, label='POD-9 model')
    sindy_full = np.vstack((L.reshape(r ** 2, 1), 
                            Q.reshape(len(galerkin9['Q'].flatten()), 1)))
    plt.hist(np.abs(sindy_full.flatten()), bins=bins, color='k', 
             label='Trapping SINDy model')
    plt.xscale('log')
    plt.legend(fontsize=14)
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_yticks([0, 20, 40, 60, 80])
    plt.xlabel('Coefficient values', fontsize=20)
    plt.ylabel('Number of coefficients', fontsize=20)
    plt.title('Histogram of coefficient values', fontsize=20)


# Make Lissajou figures with ground truth and SINDy model
def make_lissajou(r, x_train, x_test, x_train_pred, x_test_pred, filename):
    fig = plt.figure(figsize=(8, 8))
    spec = gridspec.GridSpec(ncols=r, nrows=r, figure=fig, hspace=0.0, wspace=0.0)
    for i in range(r):
        for j in range(i, r):
            plt.subplot(spec[i, j])
            plt.plot(x_train[:, i], x_train[:, j],linewidth=1)
            plt.plot(x_train_pred[:, i], x_train_pred[:, j], 'k--', linewidth=1)
            ax = plt.gca()
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                plt.ylabel(r'$x_' + str(i) + r'$', fontsize=18)
            if i == r - 1:
                plt.xlabel(r'$x_' + str(j) + r'$', fontsize=18)
        for j in range(i):
            plt.subplot(spec[i, j])
            plt.plot(x_test[:, j], x_test[:, i], 'r', linewidth=1)
            plt.plot(x_test_pred[:, j], x_test_pred[:, i], 'k--', linewidth=1)
            ax = plt.gca()
            ax.set_xticks([])
            ax.set_yticks([])     
            if j == 0:
                plt.ylabel(r'$x_' + str(i) + r'$', fontsize=18)
            if i == r - 1:
                plt.xlabel(r'$x_' + str(j) + r'$', fontsize=18)
    plt.show()
    
    
# Helper function for reading and plotting the von Karman data
def get_velocity(file):
    global nel, nGLL
    field = nek.readnek(file)
    u = np.array([field.elem[i].vel[0, 0, j, k]
               for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
    v = np.array([field.elem[i].vel[1, 0, j, k]
               for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
    return np.concatenate((u, v))


# Helper function for reading and plotting the von Karman data
def get_vorticity(file):
    field = nek.readnek(file)
    vort = np.array([field.elem[i].temp[0, 0, j, k]
               for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
    return vort


# Define von Karman grid
nx = 400
ny = 200
xmesh = np.linspace(-5, 15, nx)
ymesh = np.linspace(-5, 5, ny)
XX, YY = np.meshgrid(xmesh, ymesh)


# Helper function for plotting the von Karman data
def interp(field, method='cubic', 
           mask=(np.sqrt(XX ** 2 + YY ** 2) < 0.5).flatten('C')):
    global Cx, Cy, XX, YY
    """
    field - 1D array of cell values
    Cx, Cy - cell x-y values
    X, Y - meshgrid x-y values
    grid - if exists, should be an ngrid-dim logical that will be set to zer
    """
    ngrid = len(XX.flatten())
    grid_field = np.squeeze( np.reshape( griddata((Cx, Cy), field, (XX, YY), 
                                                  method=method), (ngrid, 1)) )
    if mask is not None:
        grid_field[mask] = 0
    return grid_field


# Helper function for plotting the von Karman data
def plot_field(field, clim=[-5, 5], label=None):
    """Plot cylinder field with masked circle"""
    im = plt.imshow(field, cmap='RdBu', vmin=clim[0], vmax=clim[1], 
                    origin='lower', extent=[-5, 15, -5, 5], 
                    interpolation='gaussian', label=label)
    cyl = plt.Circle((0, 0), 0.5, edgecolor='k', facecolor='gray')
    plt.gcf().gca().add_artist(cyl)
    return im


# Initialize quadratic SINDy library, with custom ordering 
# to be consistent with the constraint
library_functions = [lambda x:x, lambda x, y:x * y, lambda x:x ** 2]
library_function_names = [lambda x:x, lambda x, y:x + y, lambda x:x + x]
sindy_library = ps.CustomLibrary(library_functions=library_functions, 
                                 function_names=library_function_names)

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-15
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-10


# define analytic galerkin model for quadratic nonlinear systems
def galerkin_model(a, L, Q):
    """RHS of POD-Galerkin model, for time integration"""
    return (L @ a) + np.einsum('ijk,j,k->i', Q, a, a)


# define parameters and load in POD modes obtained from DNS
a = np.loadtxt('fusionPython/data/vonKarman_pod/vonKarman_a.dat')
t = a[:, 0]
r = 5
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
dt = t[1] - t[0]

# define the POD-Galerkin models from Noack (2003)
galerkin3 = sio.loadmat('fusionPython/data/vonKarman_pod/galerkin3.mat')
model3 = lambda t, a: galerkin_model(a, galerkin3['L'], galerkin3['Q'])
galerkin9 = sio.loadmat('fusionPython/data/vonKarman_pod/galerkin9.mat')

# make the Galerkin model nonlinearity exactly energy-preserving 
# rather than just approximately energy-preserving
gQ = 0.5 * (galerkin9['Q'] + np.transpose(galerkin9['Q'], [0, 2, 1]))
galerkin9['Q'] = gQ - (gQ + np.transpose(gQ, [1, 0, 2]) + np.transpose(
    gQ, [2, 1, 0]) + np.transpose(
    gQ, [0, 2, 1]) + np.transpose(
    gQ, [2, 0, 1]) + np.transpose(
    gQ, [1, 2, 0])) / 6.0
model9 = lambda t, a: galerkin_model(a, galerkin9['L'], galerkin9['Q'])

# time base for simulating Galerkin models
t_sim = np.arange(0, 500, dt)

# Generate initial condition from unstable eigenvectors
lamb, Phi = np.linalg.eig(galerkin9['L'])
idx = np.argsort(-np.real(lamb))
lamb, Phi = lamb[idx], Phi[:, idx]
a0 = np.zeros(9)
a0[0] = 1e-3
# np.real( 1e-3 * Phi[:, :2] @ np.random.random((2)) )

# get the 5D POD-Galerkin coefficients
inds5 = np.ix_([0, 1, 2, 3, -1], [0, 1, 2, 3, -1])
galerkin5 = {}
galerkin5['L'] = galerkin9['L'][inds5]
inds5 = np.ix_([0, 1, 2, 3, -1], [0, 1, 2, 3, -1], [0, 1, 2, 3, -1])
galerkin5['Q'] = galerkin9['Q'][inds5]
model5 = lambda t, a: galerkin_model(a, galerkin5['L'], galerkin5['Q'])

# make the 3D, 5D, and 9D POD-Galerkin trajectories
t_span = (t[0], t[-1])
a_galerkin3 = solve_ivp(model3, t_span, a0[:3], t_eval=t, 
                        **integrator_keywords).y.T
a_galerkin5 = solve_ivp(model5, t_span, a0[:5], t_eval=t, 
                        **integrator_keywords).y.T
adot_galerkin5 = np.gradient(a_galerkin5, axis=0) / (t[1] - t[0])
a_galerkin9 = solve_ivp(model9, t_span, a0[:9], t_eval=t, 
                        **integrator_keywords).y.T
adot_galerkin9 = np.gradient(a_galerkin9, axis=0) / (t[1] - t[0])

# plot the first 4 POD modes + the shift mode
mode_numbers = [0, 1, 2, 3, -1]
plt.figure(figsize=(16, 16))
for i in range(r):
    plt.subplot(r, 1, i + 1)
    if i == 0:
        plt.title('DNS and POD-Galerkin models on first 4 POD modes + shift mode', 
                  fontsize=16)
    plt.plot(t, a[:, mode_numbers[i]], 'r', label='POD from DNS')
    plt.plot(t, a_galerkin9[:, mode_numbers[i]], 'b', label='POD-9 model')
    plt.plot(t, a_galerkin5[:, mode_numbers[i]], 'm', label='POD-5 model')
    ax = plt.gca()
    plt.ylabel(r'$a_{' + str(mode_numbers[i]) + '}$', fontsize=20)
    plt.grid(True)
    if i == r - 1:
        plt.xlabel('t', fontsize=18)
        plt.legend(loc='upper left', fontsize=16)
    else:
        ax.set_xticklabels([])
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
a0 = np.zeros(r)
a0[0] = 1e-3

# initial guesses for the 5D trapping SINDy models
A_guess = np.asarray([
       [-0.09549166,  0.06690551,  0.07648779,  0.06031227,  0.00381231],
       [ 0.06690551, -0.16215055,  0.06025283, -0.00239112,  0.0703583 ],
       [ 0.07648779,  0.06025283, -0.25048566, -0.11299157, -0.02998519],
       [ 0.06031227, -0.00239112, -0.11299157, -0.25995424, -0.04143834],
       [ 0.00381231,  0.0703583 , -0.02998519, -0.04143834, -0.20420856]
])
m_guess = np.asarray([-1.30339101, -0.62070622, -2.01935759, 
                      -1.44105823,  0.38102858])

# same test and train trajectory for simplicity here
x_train = a
x_test = a

# define hyperparameters
max_iter = 10
eta = 1.0
threshold = 0.1
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
model.fit(x_train, t=t)
Xi = model.coefficients().T
xdot_test = model.differentiate(x_test, t=t)
xdot_test_pred = model.predict(x_test)
PL_tensor = sindy_opt.PL_
PQ_tensor = sindy_opt.PQ_
L = np.tensordot(PL_tensor, Xi, axes=([3, 2], [0, 1]))
Q = np.tensordot(PQ_tensor, Xi, axes=([4, 3], [0, 1]))
x_train_pred = model.simulate(x_train[0, :], t)
x_test_pred = model.simulate(a0, t)

# plotting and analysis
make_fits(r, t, xdot_test, xdot_test_pred, x_test, x_test_pred, 'vonKarman')
make_lissajou(r, x_train, x_test, x_train_pred, x_test_pred, 'VonKarman')
mean_val = np.mean(x_test_pred, axis=0)
mean_val = np.sqrt(np.sum(mean_val ** 2))
check_stability(r, Xi, sindy_opt, mean_val)
make_progress_plots(r, sindy_opt)
A_guess = sindy_opt.A_history_[-1]
m_guess = sindy_opt.m_history_[-1]
E_pred = np.linalg.norm(x_test - x_test_pred) / np.linalg.norm(x_test)
print('Frobenius Error = ', E_pred)

# Compute time-averaged dX/dt error
deriv_error = np.zeros(xdot_test.shape[0])
for i in range(xdot_test.shape[0]):
    deriv_error[i] = np.dot(xdot_test[i, :] - xdot_test_pred[i, :], 
                            xdot_test[i, :] - xdot_test_pred[i, :])  / np.dot(
                            xdot_test[i, :], xdot_test[i, :])
print('Time-averaged derivative error = ', np.nanmean(deriv_error))

# Interpolate onto better time base
t_traj = np.linspace(t[0], t[-1], len(t) * 10)

# random initial condition near to origin
a0 = (np.random.rand(r) - 0.5) * 2.0

# simulate trapping SINDy results
xtraj = model.simulate(a0, t_traj)

# simulate and plot 9D POD-Galerkin results
a0_galerkin9 = np.zeros(9)
a0_galerkin9[:r-1] = a0[:-1]
a0_galerkin9[-1] = a0[-1]
t_span = (t_traj[0], t_traj[-1])
xtraj_pod9 = solve_ivp(model9, t_span, a0_galerkin9, t_eval=t_traj, 
                       **integrator_keywords).y.T

# Make awesome plot
fig, ax = plt.subplots(3, 2, subplot_kw={'projection': '3d'}, figsize=(16, 16))
data = [x_test[:, [0, 1, -1]], xtraj[:, [0, 1, -1]], 
        x_test[:, [0, 1, -1]], xtraj_pod9[:, [0, 1, -1]], 
        x_test[:, [0, 1, 2]], xtraj[:, [0, 1, 2]], 
        x_test[:, [0, 1, 2]], xtraj_pod9[:, [0, 1, 2]], 
        x_test[:, [2, 3, -1]], xtraj[:, [2, 3, -1]], 
        x_test[:, [2, 3, -1]], xtraj_pod9[:, [2, 3, -1]]]
data_labels = [[r'$a_1$', r'$a_2$', r'$a_{-1}$'], 
               [r'$a_1$', r'$a_2$', r'$a_3$'], 
               [r'$a_2$', r'$a_3$', r'$a_{-1}$']]
for i in range(3):
    ax[i, 0].plot(data[4 * i][:, 0], data[4 * i][:, 1], data[4 * i][:, 2], 
                  color='r', label='POD trajectory from DNS')
    ax[i, 0].plot(data[4 * i + 1][:, 0], data[4 * i + 1][:, 1], 
                  data[4 * i + 1][:, 2],
                  color='k', label='Trapping SINDy model')
    ax[i, 1].plot(data[4 * i + 2][:, 0], data[4 * i + 2][:, 1], 
                  data[4 * i + 2][:, 2],
                  color='r', label='POD trajectory from DNS')
    ax[i, 1].plot(data[4 * i + 3][:, 0], data[4 * i + 3][:, 1], 
                  data[4 * i + 3][:, 2],
                  color='b', label='(analytic) POD-Galerkin model')
    ax[i, 0].legend(fontsize=10)
    ax[i, 0].set_xticklabels([])
    ax[i, 0].set_yticklabels([])
    ax[i, 0].set_zticklabels([])
    ax[i, 0].set_xlabel(data_labels[i][0], fontsize=14)
    ax[i, 0].set_ylabel(data_labels[i][1], fontsize=14)
    ax[i, 0].set_zlabel(data_labels[i][2], fontsize=14)
    ax[i, 1].legend(fontsize=10)
    ax[i, 1].set_xticklabels([])
    ax[i, 1].set_yticklabels([])
    ax[i, 1].set_zticklabels([])
    ax[i, 1].set_xlabel(data_labels[i][0], fontsize=14)
    ax[i, 1].set_ylabel(data_labels[i][1], fontsize=14)
    ax[i, 1].set_zlabel(data_labels[i][2], fontsize=14)
plt.show()


# make 3D illustration of the trapping region
trapping_region(r, x_test_pred, Xi, sindy_opt, 'Von Karman')

# load in POD modes from DNS again
r = 9
a = np.loadtxt('fusionPython/data/vonKarman_pod/vonKarman_a.dat')
if r < 9:
    a_temp = a[:, 1:r]
    a_temp = np.hstack((a_temp, a[:, -1].reshape(3000, 1)))
    a = a_temp
else:
    a = a[:, 1:r + 1]
a = a[tbegin:tend:skip, :]
x_test9 = a
x_train9 = a
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
model.fit(x_train9, t=t)
Xi = model.coefficients().T
xdot_test9 = model.differentiate(x_test9, t=t)
xdot_test_pred9 = model.predict(x_test9)
PL_tensor = sindy_opt.PL_
PQ_tensor = sindy_opt.PQ_
L = np.tensordot(PL_tensor, Xi, axes=([3, 2], [0, 1]))
Q = np.tensordot(PQ_tensor, Xi, axes=([4, 3], [0, 1]))
x_train_pred9 = model.simulate(x_train9[0, :], t)
x_test_pred9 = model.simulate(a0, t)

# plotting and analysis
mean_val = np.mean(x_test_pred, axis=0)
mean_val = np.sqrt(np.sum(mean_val ** 2))
check_stability(r, Xi, sindy_opt, mean_val)
make_fits(r, t, xdot_test9, xdot_test_pred9, x_test9, x_test_pred9, 'vonKarman9')

# Find the current A and m
A_guess = sindy_opt.A_history_[-1]
m_guess = sindy_opt.m_history_[-1]
print("A_guess: ", A_guess)
print("m_guess: ", m_guess)

# define energies of the DNS, and both the 5D and 9D models 
# for POD-Galerkin and the trapping SINDy models
E = np.sum(a ** 2, axis=1)
E_galerkin5 = np.sum(a_galerkin5 ** 2, axis=1)
E_galerkin9 = np.sum(a_galerkin9 ** 2, axis=1)
E_sindy5 = np.sum(x_test_pred ** 2, axis=1)
E_sindy9 = np.sum(x_test_pred9 ** 2, axis=1)

# plot the energies
plt.figure(figsize=(16, 4))
plt.plot(t, E, 'r', label='DNS')
plt.plot(t, E_galerkin5, 'm', label='POD-5')
plt.plot(t, E_galerkin9, 'b', label='POD-9')
plt.plot(t, E_sindy5, 'k', label=r'SINDy-5')
plt.plot(t, E_sindy9, 'c', label=r'SINDy-9')

# do some formatting and save
plt.legend(fontsize=22, loc=2)
plt.grid()
plt.xlim([0, 300])
ax = plt.gca()
ax.set_yticks([0, 10, 20])
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.ylabel('Total energy', fontsize=20)
plt.xlabel('t', fontsize=20)
plt.show()

make_bar(galerkin9, L, Q)
min_sum = 100
max_sum = 0
for i in range(r):
    if np.abs(np.sum(np.diag(galerkin9['Q'][i, :, :]))) > max_sum:
         max_sum = np.abs(np.sum(np.diag(galerkin9['Q'][i, :, :])))
    if np.abs(np.sum(np.diag(galerkin9['Q'][i, :, :]))) < min_sum:
         min_sum = np.abs(np.sum(np.diag(galerkin9['Q'][i, :, :])))
print('POD-9 model, S_e = ', min_sum / max_sum)     
min_sum = 100
max_sum = 0
for i in range(r):
    if np.abs(np.sum(np.diag(Q[i, :, :]))) > max_sum:
         max_sum = np.abs(np.sum(np.diag(Q[i, :, :])))
    if np.abs(np.sum(np.diag(Q[i, :, :]))) < min_sum:
         min_sum = np.abs(np.sum(np.diag(Q[i, :, :])))
print('Trapping model, S_e = ', min_sum / max_sum)   

# path to POD mode files
field_path = 'fusionPython/data/vonKarman_pod/cyl0.snapshot'  
mode_path = 'fusionPython/data/vonKarman_pod/pod_modes/'  

# Read limit cycle flow field for grid points
field = nek.readnek(field_path)
nel = 2622  # Number of spectral elements
nGLL = 7  # Order of the spectral mesh
n = nel * nGLL ** 2

# define cell values needed for the vorticity interpolation
Cx = np.array([field.elem[i].pos[0, 0, j, k]
               for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
Cy = np.array([field.elem[i].pos[1, 0, j, k]
               for i in range(nel) for j in range(nGLL) for k in range(nGLL)])

filename = lambda t_idx: 'cyl0.f{0:05d}'.format(t_idx)

# plot mean + leading POD modes
clim = [-1, 1]
file_order = [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]
file_labels = ['Mean field', 'Shift mode', 'Mode 1', 'Mode 2', 'Mode 3', 
               'Mode 4', 'Mode 5', 'Mode 6', 'Mode 7', 'Mode 8']
fig = plt.figure(figsize=(10, 14))
spec = gridspec.GridSpec(ncols=2, nrows=5, figure=fig, hspace=0.0, wspace=0.0)
for i in range(len(file_order)):
    plt.subplot(spec[i])
    vort = interp( get_vorticity(mode_path + filename(file_order[i])) )
    plot_field(np.reshape(vort, [nx, ny], order='F').T, clim=clim, 
               label=file_labels[i])
    plt.xlim([-1, 9])
    plt.ylim([-2, 2])
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(file_labels[i], fontsize=16)

plt.show()