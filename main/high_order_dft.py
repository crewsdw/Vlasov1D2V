import numpy as np
import cupy as cp
import basis as b
import grid as g
import elliptic as ell
import timestep as ts
import fluxes as flux
import reference as ref
import data_management

import scipy.special as sp

import matplotlib.pyplot as plt

order = 4

plt.figure(figsize=(8.09, 5)) # golden :P

for order in range(2, 9):
    time_order = 3
    res_x, res_u, res_v = 20, 30, 30

    print('\nInitializing reference values...')
    triplet = np.array([1.0e21, 1.0e3, 1.0])
    refs = ref.Reference(triplet=triplet, mass_fraction=1836.15267343)

    print('\nInitializing basis...')
    orders = np.array([order, order, order])
    basis = b.Basis3D(orders)

    print('\nInitializing grids...')
    L = 1.0
    print('Domain length is ' + str(L))
    lows = np.array([-L / 2.0, -10 * refs.vt_e, -10 * refs.vt_e])
    highs = np.array([L / 2.0, 10 * refs.vt_e, 10 * refs.vt_e])
    resolutions = np.array([res_x, res_u, res_v])
    resolutions_ghosts = np.array([res_x + 2, res_u + 2, res_v + 2])
    grids = g.Grid3D(basis=basis, lows=lows, highs=highs, resolutions=resolutions, fine_x=True)

    # Fourier tests with elliptic cosine
    k = 1 - 1.0e-6
    m = k**2.0
    K = sp.ellipk(m)  # Complete elliptic integral first kind
    E = sp.ellipe(m)  # Elliptic integral of second kind
    Kp = sp.ellipk(1-m)  # K prime
    arg = 4.0*K  # scaled function argument


    def c(n):
        return 1.0/np.cosh((1 + 2 * n) * 0.5 * np.pi * Kp / K)


    def c2(n):
        return 1.0 / np.cosh(n * 0.5 * np.pi * Kp / K)


    yf = sum(c(n) * np.cos(2.0 * np.pi * (1 + 2.0 * n) * grids.x.arr[1:-1, :]) for n in range(0, 30)) * np.pi / (K * k)

    # Get Fourier coefficients
    rhs_coefficients = grids.x.fourier_basis(cp.asarray(yf))

    coefficients_g_zero = rhs_coefficients[grids.x.wave_numbers >= 0]
    wvn_g_zero = grids.x.wave_numbers[grids.x.wave_numbers >= 0]

    # Coefficients
    coefficients = np.zeros(wvn_g_zero.shape[0])
    for i in range(coefficients.shape[0]):
        if i % 2 == 1:
            coefficients[i] = c2(i) * 0.5 * np.pi / (K * k)

    # plt.figure()
    # plt.plot(grids.x.arr[1:-1, :].flatten(), yf.flatten(), 'o--')
    # plt.grid(True)

    # Difference between coefficients
    dc = np.absolute(coefficients - np.real(coefficients_g_zero.get()))

    for i in range(dc.shape[0]):
        if i % 2 == 1:
            dc[i] = dc[i] / np.absolute(coefficients[i]+1.0e-16)

    cutoff = int(coefficients.shape[0])
    error = sum(dc[i] * dc[i] for i in range(cutoff)) ** 0.5 / cutoff
    # print(error)

    # plt.figure()
    # plt.plot(grids.x.wave_numbers[grids.x.wave_numbers > 0] / grids.x.k1,
    #          np.absolute(rhs_coefficients[grids.x.wave_numbers > 0].get()), 'o--')
    # plt.grid(True)
    # plt.plot(wvn_g_zero / grids.x.k1, np.real(coefficients_g_zero.get()), 'o--')
    # plt.plot(coefficients, 'o--')
    modes = np.arange(dc.shape[0])
    plt.semilogy(modes[1::2], dc[1::2], 'o--', label=str(order) + ' nodes')

plt.axis([0, dc.shape[0]-1, 1.0e-10, 1.0e2])
plt.xlabel(r'Mode number $p$')
plt.ylabel(r'Error normalized to the analytic coefficient $\epsilon = |\tilde{c}_p - c_p| / |c_p|$')
plt.grid(True)
plt.tight_layout()
# plt.legend(bbox_to_anchor=(1,1), loc="upper left")
plt.legend(loc='lower right')
# plt.legend(loc='best')
plt.savefig('..\\pictures\\dft_error\\errors_N' + str(res_x) + '_o' + str(orders[0]) + '.png')

plt.show()
