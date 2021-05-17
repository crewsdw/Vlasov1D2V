import data_management as dm
import reference
import grid as g
import basis as b

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib
import pyvista as pv

pv.set_plot_theme("document")

# RC
font = {'weight': 'normal',
        'size': 18}

matplotlib.rc('font', **font)


def grid_flatten(arr):
    # return arr.reshape(res[0] * order[0], res[1] * order[1], res[2] * order[2])
    return arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2] * arr.shape[3], arr.shape[4] * arr.shape[5])


# Filename
folder = '..\\data\\'
filename = 'ring_param6_poisson2_2'  # 'ring_param6_mode1'  #  #  # 'ring_param6_poisson2_2'

# Read data files
save_file = dm.ReadData(folder, filename)

# time, distribution, potential, density, field_energy = save_file.read_data()
time, distribution, potential = save_file.read_specific_time(idx=0)
print(time)
# potential = save_file.read_potential_only(idx=90)

# Run info
orders, resolutions, lows, highs, time_info, ref_values = save_file.read_info()

ring_j = 6
om_pc = 10.0

# Set up reference and grid
# Fix file
ref_values[2] = 1.0
lows[2] = -8.5
highs[2] = 8.5
# Now refs
refs = reference.Reference(triplet=ref_values, mass_fraction=1836.15267343)
basis = b.Basis3D(orders)
grids = g.Grid3D(basis=basis, lows=lows, highs=highs, resolutions=resolutions, fine_all=True)

plt.figure(figsize=(8.09, 5))
plt.plot(grids.x.arr[1:-1, :].flatten(), -potential.flatten(), 'o--')
plt.grid(True)
plt.xlabel(r'Position $x$')
plt.ylabel(r'Potential, $-\Phi(x)$')
plt.tight_layout()
plt.show()

# Build equilibrium distribution
f_no_pert = g.Distribution(vt=refs.vt_e, ring_j=ring_j, resolutions=resolutions, orders=orders, perturbation=False)
f_no_pert.initialize_gpu(grids)

# restrict limits
xlim = [1, resolutions[0]+1]
ulim = [15, 36]
vlim = [14, 38]  # 10, 40
# ulim = [10, 40]
# vlim = [10, 40]

# Smooth edges
distribution = g.smooth_edges(cp.asarray(distribution)).get()
low, high = np.amin(distribution), np.amax(distribution)

# Mask values down outside specified range
# u_max = grids.u.arr[ulim[1], -1]
#
# distribution[np.tensordot(np.ones_like(grids.x.arr),
#                           np.sqrt((np.tensordot(grids.u.arr ** 2.0, np.ones_like(grids.v.arr), axes=0)
#        + np.tensordot(np.ones_like(grids.u.arr), grids.v.arr ** 2.0, axes=0))), axes=0) >= u_max] = 1.0

# Interpolate distribution
distribution_fine = basis.interpolate_values(grids, distribution, limits=[xlim, ulim, vlim])
distribution_no_pert_fine = basis.interpolate_values(grids, f_no_pert.arr.get(), limits=[xlim, ulim, vlim])
# print(distribution_fine.shape)
# quit()

#
# u_max = grids.u.arr[ulim, -1]
# v2D = (np.tensordot(grids.u.arr[ulim[0]:ulim[1], 0] ** 2.0, np.ones_like(grids.v.arr[vlim[0]:vlim[1], 0]), axes=0)
#        + np.tensordot(np.ones_like(grids.u.arr[ulim[0]:ulim[1], 0]), grids.v.arr[vlim[0]:vlim[1], 0] ** 2.0, axes=0))
# distribution_fine[None, None, v2D >= u_max] = 0
# print(distribution_fine.shape)
# quit()

# print(distribution.shape)
# print(f_no_pert.arr.shape)
# quit()

# print(time)
# print(field_energy)
# print(density[0,:,:])
# print(distribution[0,0,0,:,:,:,:])
# quit()

# Visualization
# plt.figure(figsize=(8.09, 5))
# plt.semilogy(time, field_energy, 'o--')
# plt.grid(True)
# plt.xlabel('Time t')
# plt.ylabel('Field energy')
# plt.tight_layout()
# plt.show()



# Movie, render
shrink = 0.25
grow = 1.0


# Full domain
# x3, u3, v3 = np.meshgrid(shrink * grids.x.arr[1:-1, :].flatten(), grow * grids.u.arr[1:-1, :].flatten(),
#                          grow * grids.v.arr[1:-1, :].flatten(), indexing='ij')
# grid = pv.StructuredGrid(x3, u3, v3)
# # Add volume info
# df = distribution[1:-1, :, 1:-1, :, 1:-1, :]  # - f_no_pert.arr[1:-1, :, 1:-1, :, 1:-1, :].get()

# Restrict domain
# NODES
# x3, u3, v3 = np.meshgrid(shrink * grids.x.arr[xlim[0]:xlim[1], :].flatten(),
#                          grow * grids.u.arr[ulim[0]:ulim[1], :].flatten(),
#                          grow * grids.v.arr[vlim[0]:vlim[1], :].flatten(), indexing='ij')
# grid = pv.StructuredGrid(x3, u3, v3)
# Add volume info
# df = distribution[xlim[0]:xlim[1], :, ulim[0]:ulim[1], :, vlim[0]:vlim[1], :]
# INTERPOLATED FINE GRID
x3, u3, v3 = np.meshgrid(shrink * grids.x.arr_fine[xlim[0]:xlim[1], :].flatten(),
                         grow * grids.u.arr_fine[ulim[0]:ulim[1], :].flatten(),
                         grow * grids.v.arr_fine[vlim[0]:vlim[1], :].flatten(), indexing='ij')
grid = pv.StructuredGrid(x3, u3, v3)
# - f_no_pert.arr[1:-1, :, 1:-1, :, 1:-1, :].get()
# df = np.log(np.absolute(distribution[1:-1, :, 1:-1, :, 1:-1, :]
# # - f_no_pert.arr[1:-1, :, 1:-1, :, 1:-1, :].get()) + 1.0)

# f03D = grid_flatten(df)
# f03D = df.reshape(df.shape[0] * orders[0], df.shape[2] * orders[1], df.shape[4] * orders[2])
f03D = grid_flatten(distribution_fine)  # - distribution_no_pert_fine)
grid['.'] = f03D.transpose().flatten()
# low, high = np.amin(distribution_fine), np.amax(distribution_fine)
# clim = [low, 0.1 * high]
# Make contours / iso-surfaces
# contour_array = np.linspace(0.9 * low, 0.9 * high, num=10)
# contours = grid.contour(contour_array)
print(low)
print(high)
# contours = grid.contour([0.3 * low, 0.3 * high])
contours = grid.contour([0.05 * high, 0.1 * high, 0.15 * high])
# print(np.amax(distribution))
# opacity = np.array([np.linspace(1.0, 0, num=5), np.linspace(0, 1.0, num=5)]).flatten() * 255.0
# opacity = np.linspace(0, 1, num=10) * 255.0

# Make slices
# slices1 = grid.slice_orthogonal(x=0, y=0, z=0)
# slices2 = grid.slice_orthogonal(x=-5, y=-5, z=0)
# slices3 = grid.slice_orthogonal(x=-5, y=+5, z=0)

# 3D plotter
p = pv.Plotter()
actor = p.add_mesh(contours, cmap='summer', show_scalar_bar=False)
# , cmap='bone')  # , clim=clim)  # , opacity=opacity)
# actor = p.add_mesh(slices1)
# actor2 = p.add_mesh(slices2)
# actor3 = p.add_mesh(slices3)
p.show_grid()
p.show(auto_close=False)

# Make a movie
# path = p.generate_orbital_path(factor=2, n_points=36, viewup=[0, 1, 0], shift=0.2)
p.open_movie('..\\pictures\\gifs\\test_mode0.mp4')

for idx_t in range(185):
    time, distribution = save_file.read_specific_time(idx=idx_t)
    # df = distribution[1:-1, :, 1:-1, :, 1:-1, :] - f_no_pert.arr[1:-1, :, 1:-1, :, 1:-1, :].get()
    # np.log(np.absolute(distribution[1:-1, :, 1:-1, :, 1:-1, :] - f_no_pert.arr[1:-1, :, 1:-1, :, 1:-1, :].get()) + 1.0)
    # grid['.'] = grid_flatten(df, resolutions, orders).transpose().flatten()
    # Interpolate distribution
    distribution_fine = basis.interpolate_values(grids, distribution, limits=[xlim, ulim, vlim])
    grid['.'] = grid_flatten(distribution_fine).transpose().flatten()
    contours = grid.contour([0.05 * high, 0.1 * high, 0.2 * high])
    # Colors
    # low, high = np.amin(df), np.amax(df)
    # clim = [low, high]
    # Make contours / iso-surfaces
    # contour_array = np.linspace(0.9 * low, 0.9 * high, num=10)
    # contours = grid.contour(contour_array)
    # slices2 = grid.slice_orthogonal(x=-5, y=-5, z=0)
    # slices3 = grid.slice_orthogonal(x=-5, y=+5, z=0)
    # Draw actor
    p.remove_actor(actor)
    actor = p.add_mesh(contours, cmap='summer', show_scalar_bar=False) # clim=clim, opacity=opacity)
    # p.remove_actor(actor2)
    # p.remove_actor(actor3)
    # actor2 = p.add_mesh(slices2)
    # actor3 = p.add_mesh(slices3)
    print('\nWriting frame ' + str(idx_t))
    p.write_frame()

# p.orbit_on_path(path, write_frames=True)
p.close()

quit()
