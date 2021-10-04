import data_management as dm
import reference
import grid as g
import basis as b

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv


def grid_flatten(arr, res, ord):
    return arr.reshape(res[0] * ord[0], res[1] * ord[1], res[2] * ord[2])


# Filename
folder = 'spring21\\'
# filename1 = 'test1'
# filename2 = 'test2'
# filename3 = 'test3'
# filename4 = 'test4'
# filename5 = 'test5'
# filename6 = 'test6'
# filename7 = 'test7'
# filename8 = 'test8'
# filename9 = 'test9'
filename1 = 'rk3_test_0'
filename2 = 'rk3_test_1'
filename3 = 'rk3_test_2'
filename4 = 'rk3_test_3'
filename5 = 'rk3_test_4'
filename6 = 'rk3_test_5'
filename7 = 'rk3_test_6'
filename8 = 'rk3_test_7'
filename9 = 'rk3_test_8'
filename10 = 'rk3_test_9'

# Read data files
save_file1 = dm.ReadData(folder, filename1)
save_file2 = dm.ReadData(folder, filename2)
save_file3 = dm.ReadData(folder, filename3)
save_file4 = dm.ReadData(folder, filename4)
save_file5 = dm.ReadData(folder, filename5)
save_file6 = dm.ReadData(folder, filename6)
save_file7 = dm.ReadData(folder, filename7)
save_file8 = dm.ReadData(folder, filename8)
save_file9 = dm.ReadData(folder, filename9)
save_file10 = dm.ReadData(folder, filename10)

time, distribution1, potential, density, field_energy = save_file1.read_data()
time, distribution2, potential, density, field_energy = save_file2.read_data()
time, distribution3, potential, density, field_energy = save_file3.read_data()
time, distribution4, potential, density, field_energy = save_file4.read_data()
time, distribution5, potential, density, field_energy = save_file5.read_data()
time, distribution6, potential, density, field_energy = save_file6.read_data()
time, distribution7, potential, density, field_energy = save_file7.read_data()
time, distribution8, potential, density, field_energy = save_file8.read_data()
time, distribution9, potential, density, field_energy = save_file9.read_data()
time, distribution10, potential, density, field_energy = save_file10.read_data()

# Run info
orders, resolutions, lows, highs, time_info, ref_values = save_file1.read_info()

# Set up reference and grid
refs = reference.Reference(triplet=ref_values, mass_fraction=1836.15267343)
basis = b.Basis3D(orders)
grids = g.Grid3D(basis=basis, lows=lows, highs=highs, resolutions=resolutions)

# Visualization
# plt.figure()
# plt.plot(time, field_energy, 'o--')
# plt.grid(True)
# plt.xlabel('Time t')
# plt.ylabel('Field energy')


# def l2_integral(d):
#     mass_matrix = np.tensordot(basis.b1.mass / grids.x.J, np.tensordot(basis.b2.mass / grids.u.J,
#                                                                        basis.b3.mass / grids.v.J, axes=0), axes=0)
#     # l2x_d = np.transpose(np.tensordot(basis.b1.mass, d, axes=([1], [1])), [1, 0, 2, 3, 4, 5])
#     # l2x_d = np.tensordot(l2x_d, d, axes=([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]))
#     # l2u_d = np.transpose(np.tensordot(basis.b1.mass, d, axes=([1], [3])), [1, 2, 3, 0, 4, 5])
#     # l2u_d = np.tensordot(l2u_d, d1, axes=([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]))
#     # l2v_d = np.transpose(np.tensordot(basis.b1.mass, d, axes=([1], [5])), [1, 2, 3, 4, 5, 0])
#     # l2v_d = np.tensordot(l2v_d, d1, axes=([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]))
#
#     l2_d = np.transpose(np.tensordot(mass_matrix, d, axes=([1, 3, 5], [1, 3, 5])), [3, 0, 4, 1, 5, 2])
#     l2_d = np.tensordot(l2_d, d, axes=([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]))
#     # print(np.sqrt(l2_d))
#     # quit()
#
#     # l2x_d = np.transpose(np.tensordot(basis.b1.mass, d, axes=([1], [1])), [1, 0, 2, 3, 4, 5])
#     # l2x_d = np.multiply(l2x_d, d)
#     # l2x_d = l2x_d.sum(axis=1)
#     # l2x_d = np.tensordot(l2x_d, d, axes=([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]))
#     # print(l2x_d.shape)
#     # print(d.shape)
#     # quit()
#     # l2u_d = np.transpose(np.tensordot(basis.b1.mass, l2x_d, axes=([1], [3])), [1, 2, 3, 0, 4, 5])
#     # l2u_d = np.tensordot(l2u_d, d1, axes=([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]))
#     # l2v_d = np.transpose(np.tensordot(basis.b1.mass, l2u_d, axes=([1], [5])), [1, 2, 3, 4, 5, 0])
#     # l2v_d = np.tensordot(l2v_d, d1, axes=([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]))
#
#     return np.sqrt(l2_d)
    # return np.sqrt(l2x_d + l2u_d + l2v_d)
    # return np.sqrt(l2v_d)

# Max error


# print(d1.shape)
# print(l2x_d1)
# print(l2u_d1)
# print(l2v_d1)
# l2_d1 = np.sqrt(l2x_d1 + l2u_d1 + l2v_d1)
# print(l2_d1)

# quit()
d1 = distribution2[-1, 1:-1, :, 1:-1, :, 1:-1, :] - distribution1[-1, 1:-1, :, 1:-1, :, 1:-1, :]
d2 = distribution3[-1, 1:-1, :, 1:-1, :, 1:-1, :] - distribution2[-1, 1:-1, :, 1:-1, :, 1:-1, :]
d3 = distribution4[-1, 1:-1, :, 1:-1, :, 1:-1, :] - distribution3[-1, 1:-1, :, 1:-1, :, 1:-1, :]
d4 = distribution5[-1, 1:-1, :, 1:-1, :, 1:-1, :] - distribution4[-1, 1:-1, :, 1:-1, :, 1:-1, :]
d5 = distribution6[-1, 1:-1, :, 1:-1, :, 1:-1, :] - distribution5[-1, 1:-1, :, 1:-1, :, 1:-1, :]
d6 = distribution7[-1, 1:-1, :, 1:-1, :, 1:-1, :] - distribution6[-1, 1:-1, :, 1:-1, :, 1:-1, :]
d7 = distribution8[-1, 1:-1, :, 1:-1, :, 1:-1, :] - distribution7[-1, 1:-1, :, 1:-1, :, 1:-1, :]
d8 = distribution9[-1, 1:-1, :, 1:-1, :, 1:-1, :] - distribution8[-1, 1:-1, :, 1:-1, :, 1:-1, :]
d9 = distribution10[-1, 1:-1, :, 1:-1, :, 1:-1, :] - distribution9[-1, 1:-1, :, 1:-1, :, 1:-1, :]

# dd = distribution7[-1, 1:-1, :, 1:-1, :, 1:-1, :] - distribution1[-1, 1:-1, :, 1:-1, :, 1:-1, :]

# e1 = l2_integral(d1)
# e2 = l2_integral(d2)
# e3 = l2_integral(d3)
# e4 = l2_integral(d4)
# e5 = l2_integral(d5)
# e6 = l2_integral(d6)
# e7 = l2_integral(d7)
# e8 = l2_integral(d8)
# e9 = l2_integral(d9)

o = 2
e1 = np.linalg.norm(d1.flatten(), ord=o)  # / d1.shape[0]  # np.inf)
e2 = np.linalg.norm(d2.flatten(), ord=o)  # / d2.shape[0]  # np.inf)
e3 = np.linalg.norm(d3.flatten(), ord=o)  # / d3.shape[0]  # np.inf)
e4 = np.linalg.norm(d4.flatten(), ord=o)  # / d4.shape[0]  # np.inf)
e5 = np.linalg.norm(d5.flatten(), ord=o)  # / d5.shape[0]  # np.inf)
e6 = np.linalg.norm(d6.flatten(), ord=o)  # / d6.shape[0]  # np.inf)
e7 = np.linalg.norm(d7.flatten(), ord=o)  # / d7.shape[0]
e8 = np.linalg.norm(d8.flatten(), ord=o)  # / d8.shape[0]
e9 = np.linalg.norm(d9.flatten(), ord=o)  # / d9.shape[0]

# e_all = np.linalg.norm(dd.flatten(), ord=o)
# e7 = np.linalg.norm(d7.flatten(), ord=np.inf)
# e8 = np.linalg.norm(d8.flatten(), ord=np.inf)
# e9 = np.linalg.norm(d9.flatten(), ord=np.inf)

# dt = np.array([1/2, 1/4, 1/8, 1/16])

print('\nL2 differences')
print(e1)
print(e2)
print(e3)
print(e4)
print(e5)
print(e6)
print(e7)
print(e8)
print(e9)
# print(e_all)

print('\nDiff ratios')
print(e2/e1)
print(e3/e2)
print(e4/e3)
print(e5/e4)
print(e6/e5)
print(e7/e6)
print(e8/e7)
print(e9/e8)

print('\nLog 2 maneuver')
print(np.log2(e1/e2))
print(np.log2(e2/e3))
print(np.log2(e3/e4))
print(np.log2(e4/e5))
print(np.log2(e5/e6))
print(np.log2(e6/e7))
print(np.log2(e7/e8))
print(np.log2(e8/e9))

# print('\nLog 2 ratios')
# print(np.log2(e3/e2) / np.log2(e2/e1))
# print(np.log2(e4/e3) / np.log2(e3/e2))
# print(np.log2(e5/e4) / np.log2(e4/e3))

# print(np.log2(e5/e6))
# print(np.log2(e6/e7))
# print(np.log2(e7/e8))
# print(np.log2(e8/e7))

u2 = np.tensordot(grids.u.arr[1:-1, :].flatten(), np.ones_like(grids.v.arr[1:-1, :].flatten()), axes=0)
v2 = np.tensordot(np.ones_like(grids.u.arr[1:-1, :].flatten()), grids.v.arr[1:-1, :].flatten(), axes=0)
d1f = d3.reshape(grids.x.res * grids.x.order, grids.u.res * grids.u.order, grids.v.res * grids.v.order)
cb = np.linspace(np.amin(d1f), np.amax(d1f), num=100)

plt.figure()
plt.contourf(u2, v2, d1f[10, :, :], cb)
plt.colorbar()
plt.show()

quit()

print(np.log(e2)/np.log(e1))

# 3D grid
shrink = 0.75
x3, u3, v3 = np.meshgrid(shrink * grids.x.arr[1:-1, :].flatten(), grids.u.arr[1:-1, :].flatten(),
                        grids.v.arr[1:-1, :].flatten(), indexing='ij')
grid = pv.StructuredGrid(x3, u3, v3)
# Add volume info
f03D = grid_flatten(d1, resolutions, orders)
grid["vol"] = f03D.transpose().flatten()
# Make contours / iso-surfaces
contours = grid.contour(isosurfaces=200)
# 3D plotter
p = pv.Plotter()
p.add_mesh(contours, opacity='sigmoid')
p.show_grid()
p.show()

