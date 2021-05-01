import numpy as np
import cupy as cp
import basis as b
import grid as g
import elliptic as ell
import timestep as ts
import fluxes as flux
import reference as ref
import data_management

import matplotlib.pyplot as plt
import pyvista as pv

order = 8
time_order = 3
res_x, res_u, res_v = 10, 25, 25  # 25, 50, 50
folder = '..\\data\\'
filename = 'harmonics_testing'  # 'ring_param6_poisson2_2'
# Flags
plot_IC = True
study_poisson = False

ring_j = 6
om_pc = 5.0  # 10.0
delta_n = 1.0e-4

#  Initialize reference normalization parameters
print('Initializing reference values...')
triplet = np.array([1.0e21, 1.0e3, 1.0])
refs = ref.Reference(triplet=triplet, mass_fraction=1836.15267343)

#  Build basis
print('Initializing basis...')
orders = np.array([order, order, order])
basis = b.Basis3D(orders)

# Initialize grids
print('\nInitializing grids...')
k_est = 1.3 / om_pc  # 0.886 / om_pc  # k * larmor_r
L = 2.0 * np.pi / k_est  # 100.0  # 140  # 113.2
print('Domain length is ' + str(L))
lows = np.array([-L / 2.0, -8.5 * refs.vt_e, -8.5 * refs.vt_e])
highs = np.array([L / 2.0, 8.5 * refs.vt_e, 8.5 * refs.vt_e])
resolutions = np.array([res_x, res_u, res_v])
resolutions_ghosts = np.array([res_x + 2, res_u + 2, res_v + 2])
grids = g.Grid3D(basis=basis, lows=lows, highs=highs, resolutions=resolutions, fine_x=True)
geo_info = np.array([[lows[0], highs[0], resolutions[0], orders[0]],
                     [lows[1], highs[1], resolutions[1], orders[1]],
                     [lows[2], highs[2], resolutions[2], orders[2]]])

# Time information
final_time = 40.0  # 190.0  # 200.0
write_time = 1.0
# dt estimate
dt_est = grids.u.dx / highs[1]
print('Estimated dt is {:0.3e}'.format(dt_est))

# Build distribution
print('\nInitializing distribution function...')
om = 2.852  # 1.542  # 1.0524
f0 = g.Distribution(vt=refs.vt_e, ring_j=ring_j, resolutions=resolutions, orders=orders,
                    om=om, om_pc=om_pc, delta_n=delta_n)  # om = 1.05, -3.486e-1j
f0.initialize_quad_weights(grids)
f0.initialize_gpu(grids)

# Visualize / mesh grids
print('\nVisualizing initial condition...')
XX, UU = np.meshgrid(grids.x.arr[:, :].flatten(), grids.u.arr[:, :].flatten(), indexing='ij')
U, V = np.meshgrid(grids.u.arr[:, :].flatten(), grids.v.arr[:, :].flatten(), indexing='ij')
# Initial state, with and without perturbation
f_no_pert = g.Distribution(vt=refs.vt_e, ring_j=ring_j, resolutions=resolutions, orders=orders, perturbation=False)
f_no_pert.initialize_gpu(grids)
# Flat ones
f_no_pert_f = cp.asnumpy(f_no_pert.grid_flatten_gpu())
f0f0 = cp.asnumpy(f0.grid_flatten_gpu()) - f_no_pert_f
idx = 2
cb_x = np.linspace(np.amin(f0f0[:, :, (res_v * order) // idx]), np.amax(f0f0[:, :, (res_v * order) // idx]), num=100)
cb_f = np.linspace(np.amin(f0f0[(res_x * order) // idx, :, :]), np.amax(f0f0[(res_x * order) // idx, :, :]), num=100)

if plot_IC:
    plt.figure()
    plt.contourf(XX, UU, f0f0[:, :, (res_v * order) // idx], cb_x)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.colorbar()
    plt.title('xu slice of perturbation')
    plt.tight_layout()

    plt.figure()
    plt.contourf(U, V, f0f0[(res_x * order) // idx, :, :], cb_f)
    plt.xlabel('u')
    plt.ylabel('v')
    plt.colorbar()
    plt.title('uv slice of perturbation')
    plt.tight_layout()

    plt.show()

# Build elliptic operator
print('\nInitializing elliptic operator...')
e = ell.Elliptic(poisson_coefficient=refs.charge_density_multiplier)
e.build_central_flux_operator(grid=grids.x, basis=basis.b1)
e.invert()
# Zeroth moment
n0 = f0.moment_zero()

print('\nSetting magnetic field...')
e.set_magnetic_field(magnetic_field=(refs.omp_e_tau / refs.omc_e_tau) / om_pc)  # referenced to electrons
print('The magnetic field felt by electrons is {:0.3e}'.format(
     refs.electron_acceleration_multiplier * e.magnetic_field))
print('The charge density multiplier is {:0.3e}'.format(refs.charge_density_multiplier))
print('The electron acceleration multiplier is {:0.3e}'.format(refs.electron_acceleration_multiplier))
# print('The ratio omega_p / omega_c is ' + str(refs.omp_e_tau / refs.omc_e_tau))
# quit()

# Build field
e.poisson2(charge_density=cp.asarray(n0) - cp.mean(n0), grid=grids.x, basis=basis.b1)

if plot_IC:
    plt.figure()
    plt.plot(grids.x.arr[1:-1, :].flatten(), n0.get().flatten(), 'o--')
    plt.title('Density')
    plt.grid(True)

    plt.figure()
    plt.plot(grids.x.arr[1:-1, :].flatten(), e.electric_field[1:-1, :].get().flatten(), 'o--')
    plt.title('Field')
    plt.grid(True)

    plt.show()

# Flux set-up
print('\nSetting up fluxes...')
fluxes = flux.DGFlux(resolutions=resolutions_ghosts,
                     orders=orders,
                     flux_coefficients=refs.electron_acceleration_multiplier)

# Time-step set-up
print('\nSetting up time-stepper...')
stepper = ts.Stepper(time_order=time_order, space_order=orders[0], write_time=write_time, final_time=final_time)
time_info = np.array([final_time, write_time, stepper.courant, time_order])

# Save initial condition
print('\nSetting up save file...')
save_file = data_management.RunData(folder=folder, filename=filename, shape=f0.arr.shape,
                                    geometry=geo_info, time=time_info, refs=refs)
save_file.create_file(distribution=f0.arr.get(), elliptic=e, density=n0)

# Time-step loop
print('\nBeginning main loop...')
stepper.main_loop(distribution=f0, basis=basis, elliptic=e,
                  grids=grids, dg_flux=fluxes, refs=refs, save_file=save_file)

# Visualize / mesh grids
print('\nVisualizing final state...')
# Flat
f0f = cp.asnumpy(f0.grid_flatten_gpu())
cb_x = np.linspace(np.amin(f0f[:, :, (res_v * order) // idx]), np.amax(f0f[:, :, (res_v * order) // idx]), num=100)
cb_f = np.linspace(np.amin(f0f[(res_x * order) // idx, :, :]), np.amax(f0f[(res_x * order) // idx, :, :]), num=100)

df0f = f0f - f_no_pert_f  # np.log(np.abs(f0f - f0f0) + 1.0)

cb_dx = np.linspace(np.amin(df0f[:, :, (res_v * order) // idx]), np.amax(df0f[:, :, (res_v * order) // idx]), num=100)
cb_df = np.linspace(np.amin(df0f[(res_x * order) // idx, :, :]), np.amax(df0f[(res_x * order) // idx, :, :]), num=100)

plt.figure()
plt.plot(stepper.time_array, stepper.field_energy, 'o--')
plt.grid(True)
plt.xlabel('Time t')
plt.ylabel('Field energy')
plt.title('Logarithmic field energy time series')
plt.tight_layout()

plt.figure()
plt.contourf(XX, UU, df0f[:, :, (res_v * order) // idx], cb_dx)
plt.xlabel('x')
plt.ylabel('u')
plt.colorbar()
plt.title('xu slice, difference from equilibrium')
plt.tight_layout()

plt.figure()
plt.contourf(U, V, df0f[(res_x * order) // idx, :, :], cb_df)
plt.xlabel('u')
plt.ylabel('v')
plt.colorbar()
plt.title('uv slice, difference from equilibrium')
plt.tight_layout()

print('\nClose plots to continue to 3D render')
plt.show()

# 3D grid
shrink = 0.5
x3, u3, v3 = np.meshgrid(shrink * grids.x.arr[1:-1, :].flatten(), grids.u.arr[1:-1, :].flatten(),
                         grids.v.arr[1:-1, :].flatten(), indexing='ij')
grid = pv.StructuredGrid(x3, u3, v3)
# Add volume info
temp = f0.arr - f_no_pert.arr
f03D = cp.asnumpy(temp[1:-1, :, 1:-1, :, 1:-1, :].reshape(
    (grids.x.res * grids.x.order, grids.u.res * grids.u.order, grids.v.res * grids.v.order)))
grid["vol"] = f03D.transpose().flatten()
# Make contours / isosurfaces
low = np.amin(f03D)
high = np.amax(f03D)
contour_array = np.linspace(0.9 * low, 0.9 * high, num=6)
contours = grid.contour(contour_array)
# grid.plot(opacity='sigmoid')
# 3D plotter
p = pv.Plotter()
p.add_mesh(contours, opacity='linear')  # geom / sigmoid
p.show_grid()
p.show()
# Do volume opacity mesh
#
quit()
# Do slices
# slices = grid.slice_orthogonal(x=-15 * shrink, y=0, z=0)
# slices2 = grid.slice_orthogonal(x=0, y=0, z=0)
# slices3 = grid.slice_orthogonal(x=15 * shrink, y=0, z=0)
# slices4 = grid.slice_orthogonal(x=30 * shrink, y=0, z=0)
# slices5 = grid.slice_orthogonal(x=-30 * shrink, y=0, z=0)
# p.add_mesh(slices)
# p.add_mesh(slices2)
# p.add_mesh(slices3)
# p.add_mesh(slices4)
# p.add_mesh(slices5)

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

p.show_grid()
p.show()

quit()

if study_poisson:
    # Artificial density
    n0 = cp.sin(grids.x.k1 * grids.x.arr_cp[1:-1, :]) / refs.charge_density_multiplier
    n0 += 0.1 * cp.random.rand(grids.x.res * grids.x.order).reshape(grids.x.res, grids.x.order) / refs.charge_density_multiplier

    plt.figure()
    plt.plot(grids.x.arr[1:-1, :].flatten(), n0.flatten().get(), 'o--')
    plt.show()

    # Poisson problem, two ways
    e.poisson(charge_density=cp.asarray(n0) - cp.mean(n0), grid=grids.x, basis=basis.b1, anti_alias=False)
    pot1 = e.potential
    field1 = e.electric_field
    e.poisson(charge_density=cp.asarray(n0) - cp.mean(n0), grid=grids.x, basis=basis.b1, anti_alias=True)
    pot11 = e.potential
    field11 = e.electric_field
    e.poisson2(charge_density=cp.asarray(n0) - cp.mean(n0), grid=grids.x, basis=basis.b1)
    pot2 = e.potential
    field2 = e.electric_field
    # Exact solutions, -2.0 * delta_n *
    pot_exact = - refs.charge_density_multiplier * \
                np.sin(grids.x.k1 * grids.x.arr[1:-1, :]) / (grids.x.k1 ** 2.0) / refs.charge_density_multiplier
    field_exact = refs.charge_density_multiplier * \
                  np.cos(grids.x.k1 * grids.x.arr) / grids.x.k1 / refs.charge_density_multiplier
    fine_exact = -refs.charge_density_multiplier * \
                np.sin(grids.x.k1 * grids.x.arr_fine[1:-1, :]) / (grids.x.k1 ** 2.0) / refs.charge_density_multiplier
    field_fine_exact = refs.charge_density_multiplier * \
                  np.cos(grids.x.k1 * grids.x.arr_fine) / grids.x.k1 / refs.charge_density_multiplier
    # Errors
    potential_error1 = pot_exact - pot1.get()
    potential_error11 = pot_exact - pot11.get()
    potential_error2 = pot_exact - pot2.get()
    field_error1 = field_exact - field1.get()
    field_error11 = field_exact - field11.get()
    field_error2 = field_exact - field2.get()

    # Interpolate for trapz error
    interpolate1 = basis.b1.interpolate_values(grids.x, pot1.get())
    int1_error = fine_exact - interpolate1
    interpolate11 = basis.b1.interpolate_values(grids.x, pot11.get())
    interpolate2 = basis.b1.interpolate_values(grids.x, pot2.get())
    int2_error = fine_exact - interpolate2
    error1 = (np.trapz((fine_exact - interpolate1) ** 2.0, x=grids.x.arr_fine[1:-1, :], axis=1) ** 0.5).sum(axis=0) \
             / resolutions[0]
    error11 = (np.trapz((fine_exact - interpolate11) ** 2.0, x=grids.x.arr_fine[1:-1, :], axis=1) ** 0.5).sum(axis=0) \
              / resolutions[0]
    error2 = (np.trapz((fine_exact - interpolate2) ** 2.0, x=grids.x.arr_fine[1:-1, :], axis=1) ** 0.5).sum(axis=0) \
             / resolutions[0]

    field_int1 = basis.b1.interpolate_values(grids.x, field1[1:-1, :].get())
    field_int1_error = field_fine_exact[1:-1, :] - field_int1
    field_int2 = basis.b1.interpolate_values(grids.x, field2[1:-1, :].get())
    field_int2_error = field_fine_exact[1:-1, :] - field_int2

    field_error_num1 = (np.trapz(field_int1_error ** 2.0, x=grids.x.arr_fine[1:-1, :], axis=1) ** 0.5).sum(axis=0) \
             / resolutions[0]
    field_error_num2 = (np.trapz(field_int2_error ** 2.0, x=grids.x.arr_fine[1:-1, :], axis=1) ** 0.5).sum(axis=0) \
             / resolutions[0]
    print('Potential broken L2 errors are')
    print(error1)
    # print(error11)
    print(error2)
    print('Field broken L2 errors are')
    print(field_error_num1)
    print(field_error_num2)

    plt.figure()
    plt.plot(grids.x.arr[1:-1, :].flatten(), pot_exact.flatten(), 'o--', label='Exact solution')
    plt.plot(grids.x.arr[1:-1, :].flatten(), pot1.flatten().get(), 'o--', label='Stabilized central flux')
    plt.plot(grids.x.arr[1:-1, :].flatten(), pot2.flatten().get(), 'o--', label='Fourier spectral method')
    plt.plot(grids.x.arr_fine[1:-1, :].flatten(), interpolate1.flatten(), 'o--', label='Interpolated central flux')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel(r'Potential $\Phi$')
    plt.legend(loc='best')
    plt.tight_layout()

    plt.figure()
    plt.plot(grids.x.arr.flatten(), field_exact.flatten(), 'o--', label='Exact solution')
    # plt.plot(grids.x.arr[1:-1, :].flatten(),
    #          refs.electron_acceleration_multiplier * e.electric_field[1:-1, :].get().flatten(), 'o--')
    plt.plot(grids.x.arr.flatten(), field1.flatten().get(), 'o--', label='Stabilized central flux')
    plt.plot(grids.x.arr.flatten(), field2.flatten().get(), 'o--', label='Fourier spectral method')
    plt.xlabel('x')
    plt.ylabel(r'Field $E$')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    axs[1].semilogy(grids.x.arr[1:-1, :].flatten(), abs(potential_error1.flatten()), 'o--',
                    label='Nodal central flux')
    axs[1].semilogy(grids.x.arr_fine[1:-1, :].flatten(), abs(int1_error.flatten()), 'o',
                    label='Interpolated central flux')
    # axs[1].semilogy(grids.x.arr[1:-1, :].flatten(), abs(potential_error11.flatten()), 'o--',
    #                 label='Anti-aliased central flux error')
    axs[1].semilogy(grids.x.arr[1:-1, :].flatten(), abs(potential_error2.flatten()), 'o--',
                    label='Nodal Fourier method')
    axs[1].semilogy(grids.x.arr_fine[1:-1, :].flatten(), abs(int2_error.flatten()), 'o',
                    label='Interpolated Fourier method')
    axs[1].grid(True)
    axs[1].set_title('Potential')  # 'Potential error')
    # plt.legend(loc='best')
    # axs[0].legend(loc='best')
    axs[0].semilogy(grids.x.arr[1:-1, :].flatten(), abs(field_error1[1:-1, :].flatten()),
                    'o--', label='Nodal central flux error')
    axs[0].semilogy(grids.x.arr_fine[1:-1, :].flatten(), abs(field_int1_error.flatten()),
                    'o', label='Interpolated central flux error')
    # axs[0].semilogy(grids.x.arr.flatten(), abs(field_error11.flatten()),
    # 'o--', label='Anti-aliased central flux error')
    axs[0].semilogy(grids.x.arr[1:-1, :].flatten(), abs(field_error2[1:-1, :].flatten()), 'o--',
                    label='Fourier method error')
    axs[0].semilogy(grids.x.arr_fine[1:-1, :].flatten(), abs(field_int2_error.flatten()),
                    'o', label='Interpolated Fourier method error')
    axs[0].set_title('Field')
    axs[0].grid(True)
    plt.legend(loc='best')
    # Add labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel(r'Position $x$')
    plt.ylabel(r'Difference from no-noise exact solution')
    plt.tight_layout()

    plt.show()
