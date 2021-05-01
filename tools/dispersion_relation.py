import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import pyvista as pv
import cupy as cp

from numpy.polynomial import Laguerre as L

# GL Quad
# quad_arr = cp.array([[1,   0.1231760537267154,  0.0000000000000000],
#     [2,   0.1222424429903100,  -0.1228646926107104],
#     [3,   0.1222424429903100,  0.1228646926107104],
#     [4,   0.1194557635357848,  -0.2438668837209884],
#     [5,   0.1194557635357848,  0.2438668837209884],
#     [6,   0.1148582591457116,  -0.3611723058093879],
#     [7,   0.1148582591457116,  0.3611723058093879],
#     [8,   0.1085196244742637,  -0.4730027314457150],
#     [9,   0.1085196244742637,  0.4730027314457150],
#     [10,  0.1005359490670506,  -0.5776629302412229],
#     [11,  0.1005359490670506,  0.5776629302412229],
#     [12,  0.0910282619829637,  -0.6735663684734684],
#     [13,  0.0910282619829637,  0.6735663684734684],
#     [14,  0.0801407003350010,  -0.7592592630373576],
#     [15,  0.0801407003350010,  0.7592592630373576],
#     [16,  0.0680383338123569,  -0.8334426287608340],
#     [17,  0.0680383338123569,  0.8334426287608340],
#     [18,  0.0549046959758352,  -0.8949919978782753],
#     [19,  0.0549046959758352, 0.8949919978782753],
#     [20,  0.0409391567013063,  -0.9429745712289743],
#     [21,  0.0409391567013063,  0.9429745712289743],
#     [22,  0.0263549866150321,  -0.9766639214595175],
#     [23,  0.0263549866150321,  0.9766639214595175],
#     [24,  0.0113937985010263,  -0.9955569697904981],
#     [25,  0.0113937985010263,  0.9955569697904981]])
quad_arr = cp.array([[1, -0.9988664044200710501855,  0.002908622553155140958],
    [2,   -0.994031969432090712585,    0.0067597991957454015028],
    [3,   -0.985354084048005882309,   0.0105905483836509692636],
    [4,   -0.9728643851066920737133,   0.0143808227614855744194],
    [5,   -0.9566109552428079429978,   0.0181155607134893903513],
    [6,   -0.9366566189448779337809,   0.0217802431701247929816],
    [7,   -0.9130785566557918930897,   0.02536067357001239044],
    [8,   -0.8859679795236130486375,   0.0288429935805351980299],
    [9,   -0.8554297694299460846114,   0.0322137282235780166482],
    [10,  -0.821582070859335948356,    0.0354598356151461541607],
    [11,  -0.784555832900399263905,    0.0385687566125876752448],
    [12,  -0.744494302226068538261,    0.041528463090147697422],
    [13,  -0.70155246870682225109,    0.044327504338803275492],
    [14,  -0.6558964656854393607816,   0.0469550513039484329656],
    [15,  -0.6077029271849502391804,   0.0494009384494663149212],
    [16,  -0.5571583045146500543155,   0.0516557030695811384899],
    [17,  -0.5044581449074642016515,   0.0537106218889962465235],
    [18,  -0.449806334974038789147,   0.05555774480621251762357],
    [19,  -0.3934143118975651273942,   0.057189925647728383723],
    [20,  -0.335500245419437356837,    0.058600849813222445835],
    [21,  -0.2762881937795319903276,   0.05978505870426545751],
    [22,  -0.2160072368760417568473,   0.0607379708417702160318],
    [23,  -0.1548905899981459020716,   0.06145589959031666375641],
    [24,  -0.0931747015600861408545,   0.0619360674206832433841],
    [25,  -0.0310983383271888761123,   0.062176616655347262321],
    [26,  0.0310983383271888761123,    0.062176616655347262321],
    [27,  0.09317470156008614085445,   0.0619360674206832433841],
    [28,  0.154890589998145902072,    0.0614558995903166637564],
    [29,  0.2160072368760417568473,    0.0607379708417702160318],
    [30,  0.2762881937795319903276,    0.05978505870426545751],
    [31,  0.335500245419437356837,    0.058600849813222445835],
    [32,  0.3934143118975651273942,    0.057189925647728383723],
    [33,  0.4498063349740387891471,    0.055557744806212517624],
    [34,  0.5044581449074642016515,    0.0537106218889962465235],
    [35,  0.5571583045146500543155,    0.05165570306958113849],
    [36,  0.60770292718495023918,     0.049400938449466314921],
    [37,  0.6558964656854393607816,    0.046955051303948432966],
    [38,  0.7015524687068222510896,    0.044327504338803275492],
    [39,  0.7444943022260685382605,    0.0415284630901476974224],
    [40,  0.7845558329003992639053,    0.0385687566125876752448],
    [41,  0.8215820708593359483563,    0.0354598356151461541607],
    [42,  0.8554297694299460846114,    0.0322137282235780166482],
    [43,  0.8859679795236130486375,    0.02884299358053519803],
    [44,  0.9130785566557918930897,    0.02536067357001239044],
    [45,  0.9366566189448779337809,    0.0217802431701247929816],
    [46,  0.9566109552428079429978,    0.0181155607134893903513],
    [47,  0.9728643851066920737133,    0.0143808227614855744194],
    [48,  0.985354084048005882309,    0.010590548383650969264],
    [49,  0.9940319694320907125851,    0.0067597991957454015028],
    [50,  0.9988664044200710501855,    0.0029086225531551409584]])

# Parameters
a = 10.0  # 10.0  # 10.0  # 20.0 # omega_p / omega_c
j = 6  # 6
# Grids
k = np.linspace(0.01, 2.5, num=100)
fr = np.linspace(0.0, 3.0, num=100)
fi = np.linspace(0.0, 0.3, num=75)
fz = np.tensordot(fr, np.ones_like(fi), axes=0) + 1.0j*np.tensordot(np.ones_like(fr), fi, axes=0)

# Mesh-grids for plotting
K = np.tensordot(k, np.ones_like(fr), axes=0)
F = np.tensordot(np.ones_like(k), fr, axes=0)
FR = np.tensordot(fr, np.ones_like(fi), axes=0)
FI = np.tensordot(np.ones_like(fr), fi, axes=0)

KK = np.tensordot(k, np.ones_like(fi), axes=0)
FF = np.tensordot(np.ones_like(k), fi, axes=0)

# Get order j Laguerre polynomial
arr = np.zeros(j+1)
arr[-1] = 1.0
poly = L(arr)


# Build funky integrand
def integrand(x, frequency, wave_number):
    # Affine [-1,1]->[0,pi]
    frequency = cp.asarray(frequency)
    wave_number = cp.asarray(wave_number)
    theta = 0.5 * np.pi * (1.0 + x)
    # Build integrand, z=k^2*(1 + cos(theta)) 
    # z is shape (quad points, wave numbers, real freq, imag freq)
    z = cp.tensordot(cp.tensordot(1.0 + cp.cos(theta), wave_number ** 2.0, axes=0), cp.ones_like(frequency), axes=0)
    lag = cp.asarray(poly(z.get()))  # sad because slow part... laguerre poly is numpy-only
    # Compute product theta * frequency
    ft = cp.tensordot(theta, frequency, axes=0)
    # Compute product sin(x) * sin(omega * x)
    sine_ft = cp.multiply(cp.sin(theta)[:, None, None], cp.sin(ft))
    # Return integrand on quad points, sin(x) * sin(omega * x) * Laguerre(z) * exp(-z)
    return cp.multiply(sine_ft[:, None, :, :], lag * cp.exp(-z))


# Integrand
inner = cp.asarray(integrand(x=quad_arr[:, 1], frequency=fz, wave_number=k))
# Dispersion function doing 50-pt GL quad
# D = 1.0 + (a ** 2.0) * 0.5 * np.pi * cp.divide(cp.tensordot(quad_arr[:, 2], inner, axes=([0], [0])),
#                                                cp.sin(np.pi * cp.asarray(fz))[None, :, :]).get()
D = (cp.sin(np.pi * cp.asarray(fz))[None, :, :] +
    (a ** 2.0 * 0.5 * np.pi) * cp.tensordot(quad_arr[:, 2], inner, axes=([0], [0]))).get()

# idx_z = np.where(np.absolute(fr) == np.amin(np.absolute(fr)))
# print(fr[idx_z])
# idx_k = np.where(np.absolute(k - 1.7) < 5.0e-2)
# idx_k = 50
# print(k[idx_k])

# plt.figure()
# plt.contour(KK, FF, np.real(D[:, idx_z, :][:, 0, 0, :]), 0)
# plt.grid(True)
# plt.xlabel(r'$k$')
# plt.ylabel(r'$\omega_i$')
# plt.tight_layout()

plt.figure()
plt.contour(K, F, np.real(D[:, :, 0]), 0)
plt.grid(True)
plt.xlabel(r'$kr_L$')
plt.ylabel(r'$\omega_r/\omega_c$')
plt.tight_layout()

plt.show()

 #plt.figure()
# plt.contourf(FR, FI, np.imag(D[0, :, :]))
# plt.colorbar()
# plt.tight_layout()
# plt.xlabel(r'$k$')
# plt.ylabel(r'$\omega_r$')

# plt.figure()
# plt.contourf(FR, FI, np.absolute(D[25, :, :]))
# plt.colorbar()
# plt.contour(FR, FI, np.real(D[25, :, :]), 0)
# plt.contour(FR, FI, np.imag(D[25, :, :]), 0)
# plt.tight_layout()
# plt.xlabel(r'$\omega_r$')
# plt.ylabel(r'$\omega_i$')

# plt.show()

# Plot zero contours in full (k, omega_r, omega_i) space
grid = pv.UniformGrid()
grid.dimensions = D.shape
grid.origin = (k[0], fr[0], fi[0])
grid.spacing = (k[1]-k[0], fr[1]-fr[0], fi[1]-fi[0])
# Real zero contour
grid.point_arrays["values"] = np.real(D).flatten(order='F')
contours0 = grid.contour([0])
# Imag zero contour
grid.point_arrays["values"] = np.imag(D).flatten(order='F')
contours1 = grid.contour([0])
# Absolute zero contour
# grid.point_arrays["values"] = np.absolute(D).flatten(order='F')
# contours2 = grid.contour([5.0*np.amin(np.absolute(D))])
# grid.point_arrays["values"] = (np.real(D) - np.imag(D)).flatten(order='F')
# contours2 = grid.contour([0])

p = pv.Plotter()
p.add_mesh(grid.outline(), color='k')
p.add_mesh(contours0, color='r', label='Real zero')
p.add_mesh(contours1, color='g', label='Imaginary zero')
# p.add_mesh(contours2, color='k', label='Real = Imaginary')
p.add_legend()
p.show_grid(xlabel=r'Wavenumber (norm. to Larmor radius)', ylabel='Real frequency (norm. to cyclotron)', zlabel='Imaginary frequency (norm. to cyclotron)')
p.show()
