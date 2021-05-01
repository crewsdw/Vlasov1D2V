import data_management as dm
import reference
import grid as g
import basis as b

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

pv.set_plot_theme("document")


def grid_flatten(arr, res, order):
    return arr.reshape(res[0] * order[0], res[1] * order[1], res[2] * order[2])


# Filename
folder = '..\\data\\'
filename = 'ring_param6_poisson2_2'

# Read data files
save_file = dm.ReadData(folder, filename)

# time, distribution, potential, density, field_energy = save_file.read_data()

time, distribution = save_file.read_specific_time(idx=160)

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
grids = g.Grid3D(basis=basis, lows=lows, highs=highs, resolutions=resolutions)

# Build equilibrium distribution
f_no_pert = g.Distribution(vt=refs.vt_e, ring_j=ring_j, resolutions=resolutions, orders=orders, perturbation=False)
f_no_pert.initialize_gpu(grids)

# Visualization
plt.figure()
plt.semilogy(time, field_energy, 'o--')
plt.grid(True)
plt.xlabel('Time t')
plt.ylabel('Field energy')
plt.tight_layout()
plt.show()

# Movie, render
shrink = 0.25
grow = 1.0
x3, u3, v3 = np.meshgrid(shrink * grids.x.arr[1:-1, :].flatten(), grow * grids.u.arr[1:-1, :].flatten(),
                         grow * grids.v.arr[1:-1, :].flatten(), indexing='ij')
grid = pv.StructuredGrid(x3, u3, v3)
# Add volume info
df = distribution[1:-1, :, 1:-1, :, 1:-1, :] - f_no_pert.arr[1:-1, :, 1:-1, :, 1:-1, :].get()
# df = np.log(np.absolute(distribution[1:-1, :, 1:-1, :, 1:-1, :]
# # - f_no_pert.arr[1:-1, :, 1:-1, :, 1:-1, :].get()) + 1.0)

f03D = grid_flatten(df, resolutions, orders)
grid['.'] = f03D.transpose().flatten()
low, high = np.amin(df), np.amax(df)
clim = [low, 0.1 * high]
print(high)
print(low)
# Make contours / iso-surfaces
contour_array = np.linspace(0.9 * low, 0.9 * high, num=10)
# contours = grid.contour(contour_array)
contours = grid.contour([-0.02, 0.02])
opacity = np.array([np.linspace(1.0, 0, num=5), np.linspace(0, 1.0, num=5)]).flatten() * 255.0
# opacity = np.linspace(0, 1, num=10) * 255.0

# Make slices
# slices1 = grid.slice_orthogonal(x=0, y=0, z=0)
# slices2 = grid.slice_orthogonal(x=-5, y=-5, z=0)
# slices3 = grid.slice_orthogonal(x=-5, y=+5, z=0)

# 3D plotter
p = pv.Plotter()
actor = p.add_mesh(contours, cmap='cool', show_scalar_bar=False)  # , cmap='bone')  # , clim=clim)  # , opacity=opacity)
# actor = p.add_mesh(slices1)
# actor2 = p.add_mesh(slices2)
# actor3 = p.add_mesh(slices3)
p.show_grid()
p.show(auto_close=False)

### Make a movie
# path = p.generate_orbital_path(factor=2, n_points=36, viewup=[0, 1, 0], shift=0.2)
p.open_movie('..\\pictures\\gifs\\test3.mp4')

for idx_t in range(188):
    # idx_t += 100
    time, distribution = save_file.read_specific_time(idx=idx_t)
    # df = distribution[1:-1, :, 1:-1, :, 1:-1, :] - f_no_pert.arr[1:-1, :, 1:-1, :, 1:-1, :].get()
    np.log(np.absolute(distribution[1:-1, :, 1:-1, :, 1:-1, :] - f_no_pert.arr[1:-1, :, 1:-1, :, 1:-1, :].get()) + 1.0)
    grid['.'] = grid_flatten(df, resolutions, orders).transpose().flatten()
    # Colors
    # low, high = np.amin(df), np.amax(df)
    # clim = [low, high]
    # Make contours / iso-surfaces
    # contour_array = np.linspace(0.9 * low, 0.9 * high, num=10)
    # contours = grid.contour(contour_array)
    slices2 = grid.slice_orthogonal(x=-5, y=-5, z=0)
    slices3 = grid.slice_orthogonal(x=-5, y=+5, z=0)
    # Draw actor
    # p.remove_actor(actor)
    # actor = p.add_mesh(contours, cmap='bone', clim=clim, opacity=opacity)
    p.remove_actor(actor2)
    p.remove_actor(actor3)
    actor2 = p.add_mesh(slices2)
    actor3 = p.add_mesh(slices3)
    print('Writing frame ' + str(idx_t))
    p.write_frame()

# p.orbit_on_path(path, write_frames=True)
p.close()

quit()
