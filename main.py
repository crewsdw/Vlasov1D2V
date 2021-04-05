import numpy as np
import cupy as cp
import basis as b
import grid as g
import elliptic as ell
import timestep as ts
import fluxes as flux
# import fluxes_numpy as flux_np
# import timestep_numpy as ts_np
import reference as ref
import data_management
# import os

import matplotlib.pyplot as plt

# import time

order = 8  # 4  # 8  # 4
time_order = 6
res_x, res_u, res_v = 30, 30, 30  # 6, 25, 25  # 30, 30, 30
folder = 'spring21\\'
filename = 'test'

#  Initialize reference normalization parameters
print('Initializing reference values...')
triplet = np.array([1.0e21, 1.0e3, 1.0])
refs = ref.Reference(triplet=triplet, mass_fraction=1836.0)
# B_z = (refs.omp_e_tau / refs.omc_e_tau) / 10.0
# print(B_z)
# print(refs.electron_acceleration_multiplier)
# quit()

#  Build basis
print('Initializing basis...')
orders = np.array([order, order, order])
basis = b.Basis3D(orders)

# print(basis.b1.xi)
# print(basis.b1.up)
# quit()

# Initialize grids
print('\nInitializing grids...')
lows = np.array([-70 / 2.0, -8.0 * refs.vt_e, -8.0 * refs.vt_e])
highs = np.array([70 / 2.0, 8.0 * refs.vt_e, 8.0 * refs.vt_e])

resolutions = np.array([res_x, res_u, res_v])
resolutions_ghosts = np.array([res_x + 2, res_u + 2, res_v + 2])
grids = g.Grid3D(basis=basis, lows=lows, highs=highs, resolutions=resolutions)
# print(grids.u.arr[8:14, :])
# quit()
# dt estimate
dt_est = grids.u.dx / highs[1]
final_time = 200.0  # 10 * dt_est
write_time = final_time / 10.0
print('Estimated dt is {:0.3e}'.format(dt_est))

# Build distribution
print('\nInitializing distribution function...')
# f0 = g.DistributionNumpy(vt=refs.vt_e, ring_j=6, resolutions=resolutions, orders=orders)
f0 = g.Distribution(vt=refs.vt_e, ring_j=6, resolutions=resolutions, orders=orders)
f0.initialize_quad_weights(grids)
f0.initialize_gpu(grids)

# Visualize / mesh grids
# XX, UU = np.meshgrid(grids.x.arr[1:-1, :].flatten(), grids.u.arr[1:-1, :].flatten(), indexing='ij')
# U, V = np.meshgrid(grids.u.arr[1:-1, :].flatten(), grids.v.arr[1:-1, :].flatten(), indexing='ij')
XX, UU = np.meshgrid(grids.x.arr[:, :].flatten(), grids.u.arr[:, :].flatten(), indexing='ij')
U, V = np.meshgrid(grids.u.arr[:, :].flatten(), grids.v.arr[:, :].flatten(), indexing='ij')
f0f0 = cp.asnumpy(f0.grid_flatten_gpu())
# f0f0 = f0.grid_flatten_gpu()
idx = 2
cb_x = np.linspace(np.amin(f0f0[:, :, (res_v * order) // idx]), np.amax(f0f0[:, :, (res_v * order) // idx]), num=100)
cb_f = np.linspace(np.amin(f0f0[(res_x * order) // idx, :, :]), np.amax(f0f0[(res_x * order) // idx, :, :]), num=100)

plt.figure()
plt.contourf(XX, UU, f0f0[:, :, (res_v * order) // idx], cb_x)
plt.xlabel('x')
plt.ylabel('u')
plt.colorbar()

plt.figure()
plt.contourf(U, V, f0f0[(res_x * order) // idx, :, :], cb_f)
plt.xlabel('u')
plt.ylabel('v')
plt.colorbar()
plt.show()

# Build elliptic operator
print('\nInitializing elliptic operator...')
e = ell.Elliptic(poisson_coefficient=refs.charge_density_multiplier)
e.build_central_flux_operator(grid=grids.x, basis=basis.b1)
e.invert()
# Zeroth moment
n0 = f0.moment_zero()
# Poisson problem
e.poisson(charge_density=cp.asarray(n0) - cp.mean(n0), grid=grids.x, basis=basis.b1)
e.set_magnetic_field(magnetic_field=(refs.omp_e_tau / refs.omc_e_tau) / 10.0)  # electrons
print('Magnetic field is {:0.3e}'.format(refs.electron_acceleration_multiplier * e.magnetic_field))

print(e.electric_field[1, 0])
print(e.electric_field[-2, -1])

plt.figure()
plt.plot(grids.x.arr[1:-1, :].flatten(), n0.get().flatten(), 'o--')
plt.title('Density')
plt.grid(True)

plt.figure()
plt.plot(grids.x.arr[1:-1, :].flatten(),
         refs.electron_acceleration_multiplier * e.electric_field[1:-1, :].get().flatten(), 'o--')
plt.xlabel('x')
plt.ylabel('Electric field')
plt.grid(True)
plt.tight_layout()
plt.show()

# Flux set-up
print('\nSetting up fluxes...') # flux_np
fluxes = flux.DGFlux(resolutions=resolutions_ghosts,
                     orders=orders,
                     flux_coefficients=refs.electron_acceleration_multiplier)

# Save initial condition
print('\nSetting up save file...')
save_file = data_management.RunData(folder=folder, filename=filename, shape=f0.arr.shape)
save_file.create_file(distribution=f0.arr.get(), elliptic=e, density=n0)
# save_file.create_file(distribution=f0.arr, elliptic=e, density=cp.asarray(n0)) # np mode

# Time-step
print('\nSetting up time-stepper...') # ts_np
stepper = ts.Stepper(time_order=time_order, space_order=orders[0], write_time=write_time, final_time=final_time)
# Loop
print('\nBeginning main loop...')
max_0 = cp.amax(f0.arr)
# max_0 = np.amax(f0.arr)
print('Array max before main loop: ' + str(max_0))
stepper.main_loop(distribution=f0, basis=basis, elliptic=e,
                  grids=grids, dg_flux=fluxes, refs=refs, save_file=save_file)
# max_f = cp.amax(f0.arr)
max_f = np.amax(f0.arr)
print('Array max after main loop: ' + str(max_f))
print('Difference is {:0.3e}'.format(np.abs(max_f - max_0).get()))

# Visualize / mesh grids
f0f = cp.asnumpy(f0.grid_flatten_gpu())
cb_x = np.linspace(np.amin(f0f[:, :, (res_v * order) // idx]), np.amax(f0f[:, :, (res_v * order) // idx]), num=100)
cb_f = np.linspace(np.amin(f0f[(res_x * order) // idx, :, :]), np.amax(f0f[(res_x * order) // idx, :, :]), num=100)

df0f = np.log(np.abs(f0f - f0f0) + 1.0)

cb_dx = np.linspace(np.amin(df0f[:, :, (res_v * order) // idx]), np.amax(df0f[:, :, (res_v * order) // idx]), num=100)
cb_df = np.linspace(np.amin(df0f[(res_x * order) // idx, :, :]), np.amax(df0f[(res_x * order) // idx, :, :]), num=100)

plt.figure()
plt.contourf(XX, UU, f0f[:, :, (res_v * order) // idx], cb_x)
plt.xlabel('x')
plt.ylabel('u')
plt.colorbar()

plt.figure()
plt.contourf(U, V, f0f[(res_x * order) // idx, :, :], cb_f)
plt.xlabel('u')
plt.ylabel('v')
plt.colorbar()

plt.figure()
plt.contourf(XX, UU, df0f[:, :, (res_v * order) // idx], cb_dx)
plt.xlabel('x')
plt.ylabel('u')
plt.colorbar()

plt.figure()
plt.contourf(U, V, df0f[(res_x * order) // idx, :, :], cb_df)
plt.xlabel('u')
plt.ylabel('v')
plt.colorbar()

plt.show()

quit()

# # Testing stuff
# b3 = basis
# gridx = grids.x  # g.Grid1D(low=-1, high=1, res=res, basis=b3.b1)
# gridu = grids.u  # g.Grid1D(low=-10, high=10, res=res, basis=b3.b2)
# gridv = grids.v  # g.Grid1D(low=-10, high=10, res=res, basis=b3.b3)
#
# plt.figure()
# for i in range(order):
#     plt.semilogy(gridx.wave_numbers.flatten() / gridx.k1,
#                  np.absolute(gridx.spectral_transform[0, i, :].get().flatten()), 'o', label='node ' + str(i))
# plt.xlabel('Fourier mode number $k_p$')
# plt.ylabel(r'Spectral coefficient $|c_p|$')
# plt.title('Spectral coefficients of LGL element, $n=$' + str(order))
# plt.legend(loc='best')
# plt.grid(True)
# plt.xlim([-30, 30])
# plt.ylim([1.0e-8, 100.0])
# plt.tight_layout()
# #  plt.savefig('lgl_spectral_element_' + str(order) + '.png')
# plt.show()
#
# # Build elliptic operator
# print('Building elliptic operator')
# e = ell.Elliptic()
# e.build_central_flux_operator(grid=gridx, basis=b3.b1)
# print('Inverting and transferring')
# e.invert()
# print('Done')
#
# plt.figure()
# plt.spy(e.central_flux_operator)
# plt.title('Stabilized central flux symmetric elliptic DG operator')
# plt.show()
#
# # Build distribution
# f0 = g.Distribution(vt=1, ring_j=5, resolutions=resolutions, orders=orders)
# f0.initialize_quad_weights(gridu, gridv)
#
# # Mesh grids
# U, V = np.meshgrid(gridu.arr[1:-1].flatten(), gridv.arr[1:-1].flatten(), indexing='ij')
#
# print('\n\nBeginning pdf init on GPU')
# t0 = time.time()
# f0.initialize_gpu(gridx, gridu, gridv)
# print(f0.arr.shape)
# print('Time to initialize on GPU is ' + str(time.time() - t0))
# print('Memory size is ' + str(f0.arr.flatten().size * f0.arr.flatten().itemsize / 1e6) + ' MB')
# #  print('\nFlattening')
# #  t1 = time.time()
# #  df0f = f0.grid_flatten_gpu()
# #  print('Time to flatten is ' + str(time.time() - t1))
# print('\nFlattening and transferring from GPU:')
# t2 = time.time()
# #  f0f = cp.asnumpy(df0f)
# f0f = cp.asnumpy(f0.grid_flatten_gpu())
# print('Transfer time was ' + str(time.time() - t2))
# print('\nComputing zeroth moment:')
# t3 = time.time()
# n0 = f0.moment_zero()
# np_n0 = n0.get()
# print('Time to take moment is ' + str(time.time() - t3))
#
# # Check fourier coefficients ... good to go!
# # sine = cp.sin(gridx.k1*cp.asarray(gridx.arr[1:-1, :])) + cp.sin(gridx.k1*cp.asarray(gridx.arr[1:-1,:]) + 2)
# # coefficients = gridx.fourier_basis(sine)
# #
# # re_sum = gridx.sum_fourier(coefficients)
# #
# # plt.figure()
# # plt.plot(gridx.wave_numbers / gridx.k1, np.real(coefficients.get()), 'o', label='real')
# # plt.plot(gridx.wave_numbers / gridx.k1, np.imag(coefficients.get()), 'o', label='imag')
# # plt.xlabel('Mode number')
# # plt.ylabel('Amplitude')
# # plt.legend(loc='best')
# # plt.show()
# #
# # plt.figure()
# # plt.plot(gridx.arr[1:-1, :].flatten(), (sine - re_sum).get().flatten(), 'o', label='fourier error')
# # #plt.plot(gridx.arr[1:-1, :].flatten(), sine.get().flatten(), 'o', label='original')
# # #plt.plot(gridx.arr[1:-1, :].flatten(), re_sum.get().flatten(), 'o', label='summation')
# # plt.legend(loc='best')
# # plt.show()
# #
# # plt.show()
# cb_hrm = np.linspace(np.amin(f0f), np.amax(f0f), num=100)
#
# plt.figure()
# plt.contourf(U, V, f0f[10, :, :], cb_hrm)
# plt.colorbar()
# plt.show()
#
# # Poisson problem
# e.poisson(charge_density=n0 - cp.mean(n0), grid=gridx, basis=b3.b1)
#
#
# # Fluxes
# # RHS = flux.DGFlux(resolutions=resolutions, orders=orders)
# # t3 = time.time()
# # rhs = RHS.semi_discrete_rhs(function=f0.arr[1:-1, :, 1:-1, :, 1:-1, :],
# #                            electric_field=e.electric_field, basis3=b3, grid_x=gridx, grid_u=gridu, grid_v=gridv)
# # print('\nTime to calculate RHS is ' + str(time.time() - t3))
# # print('Memory:')
# # print('Memory size of pdf is ' + str(f0.arr.flatten().size * f0.arr.flatten().itemsize / 1e6) + ' MB')
# # print('Memory size of rhs is ' + str(rhs.flatten().size * rhs.flatten().itemsize / 1e6) + ' MB')
# # print('Spectral transform array is ' + str(gridx.spectral_transform.flatten().size *
# #                                            gridx.spectral_transform.flatten().itemsize / 1e6) + ' MB')
# # time.sleep(10)
# # quit()
# #
# # print('max rhs is '
# #       '')
# # print(cp.amax(rhs))
#
# # plt.figure()
# # plt.plot(gridx.arr[1:-1,:].flatten(), n0.get().flatten() - 1)
# # plt.grid(True)
# # plt.title('Density')
# #
# # plt.figure()
# # plt.plot(gridx.arr[1:-1, :].flatten(), e.potential.get().flatten(), label='Approx')
# # plt.plot(gridx.arr[1:-1, :].flatten(), -(n0.get().flatten() - 1)/np.pi ** 2.0, label = 'Exact')
# # plt.grid(True)
# # plt.legend(loc='best')
# # plt.title('Potential')
#
# field = -0.01 * np.cos(np.pi * gridx.arr) / np.pi
#
# # plt.figure()
# # plt.plot(gridx.arr[1:-1, :].flatten(), e.electric_field.get().flatten(), label='Field apprx')
# # plt.plot(gridx.arr[1:-1, :].flatten(), electric_field[1:-1,:].flatten(), label='Field exact')
# # plt.grid(True)
# # plt.legend(loc='best')
# # plt.title('Field')
#
# error_potential = e.potential.get().flatten() + (n0.get().flatten() - 1) / np.pi ** 2.0
# error_field = e.electric_field.get().flatten() - field[1:-1, :].flatten()
#
# # Clean electric_field
# coefficients = gridx.fourier_basis(e.electric_field)
# re_sum = gridx.sum_fourier(coefficients)
#
# error_field_cleaned = re_sum.get() - field[1:-1, :]
#
# plt.figure()
# plt.plot(gridx.arr[1:-1, :].flatten(), error_potential)
# plt.grid(True)
# plt.title('Error in potential')
#
# plt.figure()
# plt.plot(gridx.arr[1:-1, :].flatten(), error_field, 'o--', label='raw DG solution')
# plt.plot(gridx.arr[1:-1, :].flatten(), error_field_cleaned.flatten(), 'o--', label='anti-aliased')
# plt.grid(True)
# plt.legend(loc='best')
# plt.title('Error in electric_field')
# plt.tight_layout()
# plt.savefig('antialias_sine_N' + str(res) + '_n' + str(order) + '.png')
#
# plt.show()
#
# quit()
#
# # print('\nGrids, e.g. x')
# # print(gridx.arr)
#
#
# # f = np.sin(np.pi*gridx.arr)
# #
# # plt.figure()
# # plt.plot(gridx.arr[1:-1,:].flatten(), f[1:-1,:].flatten(), 'o--')
# # plt.grid(True)
# # plt.show()
#
# # f0 = g.Distribution(vt=1, ring_j=1, orders=orders)
# # print('\nBeginning pdf init')
# # f0.initialize(gridx, gridu, gridv)
# # print(f0.arr.shape)
# # print(f0.arr.flatten().shape)
# # print('\nFlattening')
# # #flat_f0 = f0.gridflatten()
#
# #
# # def gridflatten(arr):
# #     rs = np.transpose(arr, (0, 3, 1, 4, 2, 5))
# #     return rs.reshape((gridx.res * orders[0], gridu.res * orders[1], gridv.res * orders[2]))
#
#
# # flat_f0_1 = gridflatten(f0.arr[1:-1, 1:-1, 1:-1, :, :, :])
#
#
# # RSQ = U ** 2.0 + V ** 2.0
# # cb = np.linspace(np.amin(RSQ), np.amax(RSQ), num=100)
# # print('\nDoing tensor products...')
# #
# # # Grid indicators
# # Ix = np.tensordot(np.ones(gridx.res), np.ones(gridx.order), axes=0)
# # Iu = np.tensordot(np.ones(gridu.res), np.ones(gridu.order), axes=0)
# # Iv = np.tensordot(np.ones(gridv.res), np.ones(gridv.order), axes=0)
#
# # params
# # ring_j = 1
# # therm = 1
# # # Build gaussian
# # rsq = (np.tensordot(gridu.arr[1:-1, :] ** 2.0, Iv, axes=0) +
# #        np.tensordot(Iu, gridv.arr[1:-1, :] ** 2.0, axes=0)) / (therm ** 2.0)
# # radj = rsq ** ring_j
# # gauss = np.exp(-0.5*rsq)
# # polhrm = radj * gauss
# # f0 = np.tensordot(Ix, polhrm, axes=0)
# # f0f = f0.reshape(gridx.res*gridx.order, gridu.res*gridu.order, gridv.res*gridv.order)
#
# f0 = g.Distribution(vt=1, ring_j=5, resolutions=resolutions, orders=orders)
# f0.initialize_quad_weights(gridu, gridv)
# # print('\nBeginning pdf init on CPU')
# # t0 = time.time()
# # f0.initialize_cpu(gridx, gridu, gridv)
# # print('Time to initialize on CPU is ' + str(time.time() - t0))
# # print(f0.arr.shape)
# # print(f0.arr.flatten().shape)
# # print('Memory size is ' + str(f0.arr.flatten().size * f0.arr.flatten().itemsize / 1e6) + ' MB')
# # print('\nFlattening')
# # t1 = time.time()
# # f0f = f0.grid_flatten_cpu()
# # print('Time to flatten is ' + str(time.time() - t1))
# # print('\nTransferring to GPU:')
# # t2 = time.time()
# # df0 = cp.asarray(f0f)
# # print('Transfer time was ' + str(time.time() - t2))
#
# print('\n\nBeginning pdf init on GPU')
# t0 = time.time()
# f0.initialize_gpu(gridx, gridu, gridv)
# print('Time to initialize on GPU is ' + str(time.time() - t0))
# print(f0.arr.shape)
# print(f0.arr.flatten().shape)
# print('Memory size is ' + str(f0.arr.flatten().size * f0.arr.flatten().itemsize / 1e6) + ' MB')
# print('\nFlattening')
# t1 = time.time()
# df0f = f0.grid_flatten_gpu()
# print('Time to flatten is ' + str(time.time() - t1))
# print('\nTransferring from GPU:')
# t2 = time.time()
# f0f = cp.asnumpy(df0f)
# print('Transfer time was ' + str(time.time() - t2))
# print('\nComputing zeroth moment:')
# t3 = time.time()
# n0 = f0.moment_zero()
# print('Time to take moment is ' + str(time.time() - t3))
#
# # plt.figure()
# # plt.contourf(U, V, RSQ, cb)
# # plt.colorbar()
# # plt.contourf(U, V, fpl_1, cb)
#
# plt.figure()
# plt.contourf(U, V, f0f[10, :, :], cb_hrm)
# plt.colorbar()
#
# plt.figure()
# plt.plot(gridx.arr[1:-1, :].flatten(), n0.get().flatten())
# plt.grid(True)
#
# plt.show()

# print(cp.amin(f0.arr))
# print(f0.arr.dtype)
#
# f0f0 = cp.asnumpy(f0.grid_flatten_gpu())
#
# print(np.amax(f0f0))
#
# corners = cp.zeros_like(f0.arr)
# corners[:, :, :, 0, :, :] = -1
# corners[:, :, :, -1, :, :] = +1
# corners[:, :, :, :, :, 0] = -1
# corners[:, :, :, :, :, -1] = +1
# corners_f = cp.asnumpy(corners[1:-1, :, 1:-1, :, 1:-1, :].reshape(
#                         (grids.x.res * grids.x.order, grids.u.res * grids.u.order, grids.v.res * grids.v.order)))
#
# plt.figure()
# plt.imshow(f0f0[11, :, :])
#
# with open('new_way.npy', 'rb') as file_var:
#     new_way = np.load(file_var).reshape(res_x*order, res_u*order, res_v*order)
#
# with open('old_way.npy', 'rb') as file_var:
#     old_way = np.load(file_var)
#
# plt.figure()
# plt.imshow(new_way[11, :, :])
# plt.colorbar()
# plt.title('New way')
#
# plt.figure()
# plt.imshow(old_way[11, :, :])
# plt.colorbar()
# plt.title('Old way')
#
# difference = new_way - old_way
#
# print(np.amax(difference))
# print(np.amin(difference))
#
# plt.figure()
# plt.imshow(difference[11, 78:105, 78:105])
# plt.colorbar()
# plt.title('Difference of new way and old way')
#
# plt.figure()
# plt.imshow(corners_f[11, 78:105, 78:105])
#
# plt.figure()
# plt.imshow(old_way[11, 78:105, 78:105])
# plt.title('Old way zoom')
#
# plt.figure()
# plt.imshow(new_way[11, 78:105, 78:105])
# plt.title('New way zoom')
#
# plt.figure()
# plt.imshow(difference[11, 78:105, 78:105])
# plt.title('Difference zoom')
#
# plt.show()

# print(f0.arr[3, 0, 11:15, :, 15, 0])
# for i in range(8):
#     plt.figure()
#     plt.imshow(f0.arr[3, i, 8:16, :7, 8:16, :7].reshape(8*7, 8*7).get())
#     plt.title('X node ' + str(i))
# plt.show()
# quit()
