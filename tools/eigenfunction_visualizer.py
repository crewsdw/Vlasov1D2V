import numpy as np
import scipy.special as sp
# import matplotlib.pyplot as plt
import math
import pyvista as pv
# from pyvistaqt import BackgroundPlotter
# import time


#
# Purpose: Check out the kernel function for imaginary frequency

# Parameters
# L = 30.0 # 2.0 * np.pi / 1.4 # 10.0 # larmor radii
# k = 2.0 * np.pi / L # 1.4 # 1.11 # 0.628
# k = 1.4 # 1.5 # 0.7 # 0.7 # 0.4 # 0.2 # 1.11 # 1.4
# L = 2.0 * np.pi / k
k = 0.05  # 1.4  # 1.3
L = 2.0 * np.pi / k
print('Wave-number is ' + str(k))
v = np.linspace(1.0e-6, 8.5, num=75)
phi = np.linspace(0, 2.0*np.pi, num=75)
space = np.linspace(0, L, num=75)
b = - k * v  # 0.0 + 0.3110j
om_p = 1.414  # 3 + 4.0e-8  # 2.00093  # 1.414  # 1.2 - 0.1j  # 1.0524  # 4.912e-1j  # 0.3110j # 1.414 # 1.2 + 0.2j
om_n = -om_p  # -2.00093  # -1.414  # -1.2 - 0.1j  # -4.912e-1j  # -1.414 # -0.3110j # -1.0e-5 # -2.05 # -3.1 # -1.414

# Distribution
a = 1
ring_j = 0
x = 0.5 * (v/a) ** 2.0
f0 = 1/(2.0 * np.pi * (a ** 2.0) * math.factorial(ring_j)) * np.multiply(x ** ring_j, np.exp(-x))
dfdv = np.multiply(f0, (ring_j/x - 1.0)) / (a ** 2.0)


# Fourier series components
def inner_series(n, om):
    return n / (n - om) * np.tensordot(sp.jv(n, b), np.exp(-1j * n * phi), axes=0)


# Sum up to "terms" in angular part
terms_n = 20
factor = np.tensordot(b, np.sin(phi), axes=0)
# Positive frequency part
series = np.array([inner_series(n, om_p) for n in range(-terms_n, terms_n+1)]).sum(axis=0)
Gam_p = -1j * np.tensordot(np.exp(1j * k * space), np.multiply(np.exp(1j * factor), series), axes=0)
# Negative frequency part
series = np.array([inner_series(n, om_n) for n in range(-terms_n, terms_n+1)]).sum(axis=0)
Gam_n = -1j * np.tensordot(np.exp(1j * k * space), np.multiply(np.exp(1j * factor), series), axes=0)


# Convert from cylindrical to cartesian
vx = np.tensordot(v, np.cos(phi), axes=0)
vy = np.tensordot(v, np.sin(phi), axes=0)

scale = 0.1
x3 = np.tensordot(scale * space, np.ones_like(vx), axes=0)
vx3 = np.tensordot(np.ones_like(space), vx, axes=0)
vy3 = np.tensordot(np.ones_like(space), vy, axes=0)
grid = pv.StructuredGrid(x3, vx3, vy3)

Gam = np.real(Gam_p + Gam_n)
perturbation = np.multiply(Gam, dfdv[None, :, None])
low = np.amin(perturbation)
high = np.amax(perturbation)
contour_array = np.linspace(0.9 * low, 0.9 * high, num=6)

grid["vol"] = perturbation.transpose().flatten()
# slice0 = grid.slice_orthogonal(x=scale * L/4, y=0, z=0)
# slice1 = grid.slice_orthogonal(x=scale * 3*L/4, y=0, z=0)
contour = grid.contour(contour_array)
# contour = grid.contour([0.8*high])
clim = [low, high]

p = pv.Plotter()
# actor0 = p.add_mesh(slice0, clim=clim)
# actor1 = p.add_mesh(slice1, clim=clim)
actor = p.add_mesh(contour, clim=clim, opacity='linear')
p.show_grid()
p.show(auto_close=False)
p.open_movie('test_bernstein4.mp4', framerate=12)

# Real part and contour plot
t = np.linspace(0, 20, num=100)
for idx_t in range(t.shape[0]):
    idx = 10
    Gamr = np.real(Gam_p * np.exp(-1j * om_p * t[idx_t]) + Gam_n * np.exp(-1j * om_n * t[idx_t]))
    cb = np.linspace(np.amin(Gamr), np.amax(Gamr), num=100)
    
    perturbation = -np.multiply(Gamr, dfdv[None, :, None])
    cbp = np.linspace(np.amin(perturbation), np.amax(perturbation), num=100)
    
    # plt.figure()
    # plt.contourf(vx, vy, Gamr[idx, :, :], cb)
    # plt.xlabel(r'Velocity $v_x$')
    # plt.ylabel(r'Velocity $v_y$')
    # plt.colorbar()
    # plt.title(str(terms_n) + r' term approximation of $\Omega(k_0v, \varphi)$')
    # plt.tight_layout()
    
    # plt.figure()
    # plt.contourf(vx, vy, perturbation[idx, :, :], cbp)
    # plt.title(str(terms_n) + r' term approximation of $f_1(\omega = $' + str(om) + r'$\omega_{c})$ and $\lambda = $ ' + str(L) + r'$r_L$')
    # plt.xlabel(r'Velocity $v_x$')
    # plt.ylabel(r'Velocity $v_y$')
    # plt.colorbar()
    # plt.tight_layout()
    
    # plt.show()
    grid["vol"] = perturbation.transpose().flatten()
    contours = grid.contour(contour_array)
    # slice0 = grid.slice_orthogonal(x= scale * L/2, y=0, z=0)
    # 3D plotter
    p.remove_actor(actor)
    # actor = p.add_mesh(slice0, clim=clim)
    actor = p.add_mesh(contours, clim=clim, opacity='linear')
    p.write_frame()

p.close()

quit()
