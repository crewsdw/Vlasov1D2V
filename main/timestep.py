import numpy as np
import cupy as cp
import time as timer

# import copy

# For debug
# import matplotlib.pyplot as plt
# import pyvista as pv

# Dictionaries
ssp_rk_switch = {
    1: [1],
    2: [1 / 2, 1 / 2],
    3: [1 / 3, 1 / 2, 1 / 6],
    4: [3 / 8, 1 / 3, 1 / 4, 1 / 24],
    5: [11 / 30, 3 / 8, 1 / 6, 1 / 12, 1 / 120],
    6: [53 / 144, 11 / 30, 3 / 16, 1 / 18, 1 / 48, 1 / 720],
    7: [103 / 280, 53 / 144, 11 / 60, 3 / 48, 1 / 72, 1 / 240, 1 / 5040],
    8: [2119 / 5760, 103 / 280, 53 / 288, 11 / 180, 1 / 64, 1 / 360, 1 / 1440, 1 / 40320]
}

# Courant numbers for RK-DG stability from Cockburn and Shu 2001, [time_order][space_order-1]
courant_numbers = {
    1: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    2: [1.0, 0.333],
    3: [1.256, 0.409, 0.209, 0.130, 0.089, 0.066, 0.051, 0.040, 0.033],
    4: [1.392, 0.464, 0.235, 0.145, 0.100, 0.073, 0.056, 0.045, 0.037],
    5: [1.608, 0.534, 0.271, 0.167, 0.115, 0.085, 0.065, 0.052, 0.042],
    6: [1.776, 0.592, 0.300, 0.185, 0.127, 0.093, 0.072, 0.057, 0.047],
    7: [1.977, 0.659, 0.333, 0.206, 0.142, 0.104, 0.080, 0.064, 0.052],
    8: [2.156, 0.718, 0.364, 0.225, 0.154, 0.114, 0.087, 0.070, 0.057]
}

nonlinear_ssp_rk_switch = {
    2: [[1 / 2, 1 / 2, 1 / 2]],
    3: [[3 / 4, 1 / 4, 1 / 4],
        [1 / 3, 2 / 3, 2 / 3]]
}


class Stepper:
    def __init__(self, time_order, space_order, write_time, final_time, linear=False):
        # Time-stepper order and SSP-RK coefficients
        self.time_order = time_order
        self.space_order = space_order
        if linear:
            self.coefficients = self.get_coefficients()
        else:
            self.coefficients = self.get_nonlinear_coefficients()

        # Courant number
        # self.test_number = 0
        self.courant = self.get_courant_number()  # / (2.0 ** self.test_number)  # 1.0 / (2.0 * self.space_order - 1)
        #   # / 32.0  # / 256.0  # / 64.0  # / 32.0  # / 16.0

        # Simulation time init
        self.time = 0
        self.dt = None
        self.steps_counter = 0
        self.write_counter = 1  # IC already written

        # Time between write-outs
        self.write_time = write_time
        # Final time to step to
        self.final_time = final_time

        # Field energy and time array
        self.time_array = np.array([self.time])
        self.field_energy = np.array([])

    def get_coefficients(self):
        return np.array([ssp_rk_switch.get(self.time_order, "nothing")][0])

    def get_nonlinear_coefficients(self):
        return np.array(nonlinear_ssp_rk_switch.get(self.time_order, "nothing"))

    def get_courant_number(self):
        return courant_numbers.get(self.time_order)[self.space_order - 1]

    def main_loop(self, distribution, basis, elliptic, grids, dg_flux, refs, save_file):
        # Loop while time is less than final time
        t0 = timer.time()
        # Initial field energy
        self.field_energy = np.append(self.field_energy, elliptic.electric_energy(grid=grids.x).get())
        # Look at ghost cells
        print('\nInitializing time-step...')
        # Adapt time-step
        self.adapt_time_step(max_speeds=get_max_speeds(grids, elliptic, refs),
                             dx=grids.x.dx, du=grids.u.dx, dv=grids.v.dx)
        while self.time < self.final_time:
            # Perform RK update
            self.nonlinear_ssp_rk(distribution=distribution, basis=basis, elliptic=elliptic,
                                  grids=grids, dg_flux=dg_flux, refs=refs)
            # self.ssp_rk_update(distribution=distribution, basis=basis, elliptic=elliptic,
            #                    grids=grids, dg_flux=dg_flux, refs=refs)
            # Update time and steps counter
            self.time += self.dt
            self.steps_counter += 1
            # Get field energy and time
            self.time_array = np.append(self.time_array, self.time)
            energy = elliptic.electric_energy(grid=grids.x).get()
            self.field_energy = np.append(self.field_energy, energy)
            # Do write-out sometimes
            if self.time > self.write_counter * self.write_time:
                print('\nI made it through step ' + str(self.steps_counter))
                self.write_counter += 1
                print('Saving data...')
                save_file.save_data(distribution=distribution.arr.get(),
                                    elliptic=elliptic,
                                    density=distribution.moment_zero(),
                                    time=self.time,
                                    field_energy=energy)
                print('Done.')
                print('The simulation time is {:0.3e}'.format(self.time))
                print('The time-step is {:0.3e}'.format(self.dt))
                print('Time since start is ' + str((timer.time() - t0) / 60.0) + ' minutes')
            # if cp.isnan(distribution.arr).any():
            #     print('\nThere is nan')
            #     print(self.steps_counter)
            #     quit()
            # if self.steps_counter == 10 * (2.0 ** self.test_number):
            #     self.write_counter += 1
            #     print('Saving data...')
            #     save_file.save_data(distribution=distribution.arr.get(),
            #                         elliptic=elliptic,
            #                         density=distribution.moment_zero(),
            #                         time=self.time,
            #                         field_energy=energy)
            #     print('\nAll done at step ' + str(self.steps_counter))
            #     print('The simulation time is {:0.3e}'.format(self.time))
            #     print('The time-step is {:0.3e}'.format(self.dt))
            #     print('Time since start is ' + str((timer.time() - t0) / 60.0) + ' minutes')
            #     break
                # quit()

        print('\nFinal time reached')
        print('Total steps were ' + str(self.steps_counter))

    def nonlinear_ssp_rk(self, distribution, basis, elliptic, grids, dg_flux, refs):
        # Sync ghost cells
        distribution.swap_ghost_cells_of_array()
        # Set up stages
        t_shape = tuple([self.time_order] + [size for size in distribution.arr.shape])
        arr_stages = cp.zeros(t_shape)
        # Set velocity boundary conditions
        arr_stages[:, :, :, 0, :, :, :] = distribution.arr[:, :, 0, :, :, :]
        arr_stages[:, :, :, -1, :, :, :] = distribution.arr[:, :, -1, :, :, :]
        arr_stages[:, :, :, :, :, 0, :] = distribution.arr[:, :, :, :, 0, :]
        arr_stages[:, :, :, :, :, -1, :] = distribution.arr[:, :, :, :, -1, :]
        # First stage, moment
        density = distribution.moment_zero()
        # Calculate electric field
        elliptic.poisson2(charge_density=cp.mean(density) - density, grid=grids.x, basis=basis.b1)
        # Compute first RK stage
        no_ghost_idx = tuple([0] + [idx for idx in grids.no_ghost_slice])
        arr_stages[no_ghost_idx] = distribution.arr[grids.no_ghost_slice] + (self.dt *
                                                                            dg_flux.semi_discrete_rhs(
                                                                                function=distribution.arr,
                                                                                elliptic=elliptic,
                                                                                basis=basis,
                                                                                grids=grids)[grids.no_ghost_slice])
        # Compute further stages
        for i in range(1, self.time_order):
            # Sync ghost-cells
            arr_stages[i-1, 0, :, :, :, :, :] = arr_stages[i-1, -2, :, :, :, :, :]
            arr_stages[i-1, -1, :, :, :, :, :] = arr_stages[i-1, 1, :, :, :, :, :]
            # Next stage
            density = distribution.moment_zero_of_arr(arr_stages[i-1, :, :, :, :, :, :])
            # Calculate electric field
            elliptic.poisson2(charge_density=cp.mean(density) - density, grid=grids.x, basis=basis.b1)
            # RK stage advance
            df_dt = dg_flux.semi_discrete_rhs(function=arr_stages[i-1, :, :, :, :, :, :],
                                              elliptic=elliptic,
                                              basis=basis,
                                              grids=grids)[grids.no_ghost_slice]
            g_idx_i = tuple([i] + [idx for idx in grids.no_ghost_slice])
            g_idx_i1 = tuple([i - 1] + [idx for idx in grids.no_ghost_slice])
            arr_stages[g_idx_i] = (self.coefficients[i - 1, 0] * distribution.arr[grids.no_ghost_slice] +
                                  self.coefficients[i - 1, 1] * arr_stages[g_idx_i1] +
                                  self.coefficients[i - 1, 2] * self.dt * df_dt)
        # Adapt time-step
        self.adapt_time_step(max_speeds=get_max_speeds(grids, elliptic, refs),
                            dx=grids.x.dx, du=grids.u.dx, dv=grids.v.dx)
        # Update distribution
        distribution.arr[grids.no_ghost_slice] = arr_stages[-1, 1:-1, :, 1:-1, :, 1:-1, :]

    def ssp_rk_update(self, distribution, basis, grids, elliptic, dg_flux, refs):
        # Sync ghost cells
        distribution.swap_ghost_cells_of_array()
        # Set up explicit advance array (including ghost cells in copy)
        distribution.copy_for_explicit_advance()
        # Zeroth stage update: coefficient zero of the current time-step
        distribution.arr[grids.no_ghost_slice] = self.coefficients[0] * distribution.arr[grids.no_ghost_slice]
        # Time-step loop
        for i in range(1, self.time_order):
            # Periodic BC: Sync ghost-cells
            distribution.swap_ghost_cells_of_copy()
            # Compute moment
            density = distribution.moment_zero_of_copy()
            # Calculate electric field
            elliptic.poisson(charge_density=cp.mean(density) - density, grid=grids.x, basis=basis.b1)
            # Make forward Euler advance
            distribution.arr_explicit_copy[grids.no_ghost_slice] += (self.dt *
                                                                     dg_flux.semi_discrete_rhs(
                                                                         function=distribution.arr_explicit_copy,
                                                                         elliptic=elliptic,
                                                                         basis=basis,
                                                                         grids=grids)[grids.no_ghost_slice])

            # Accumulate actual update with ssp-rk coefficients
            distribution.arr[grids.no_ghost_slice] += (self.coefficients[i] *
                                                       distribution.arr_explicit_copy[grids.no_ghost_slice])

        # Last evaluation?
        distribution.swap_ghost_cells_of_copy()
        # Compute moment
        density = distribution.moment_zero_of_copy()
        # Calculate electric field
        elliptic.poisson(charge_density=cp.mean(density) - density, grid=grids.x, basis=basis.b1)
        distribution.arr[grids.no_ghost_slice] += (self.coefficients[-1] * self.dt *
                                                   dg_flux.semi_discrete_rhs(function=distribution.arr_explicit_copy,
                                                                             elliptic=elliptic,
                                                                             basis=basis,
                                                                             grids=grids)[grids.no_ghost_slice])
        # Adapt time-step
        self.adapt_time_step(max_speeds=get_max_speeds(grids, elliptic, refs),
                             dx=grids.x.dx, du=grids.u.dx, dv=grids.v.dx)

    def adapt_time_step(self, max_speeds, dx, du, dv):
        self.dt = self.courant / ((max_speeds[0] / dx) + (max_speeds[1] / du) + (max_speeds[2] / dv))


def get_max_speeds(grids, elliptic, refs):
    return np.absolute(np.array([grids.u.arr_max,
                                 cp.amax(cp.absolute(refs.electron_acceleration_multiplier *
                                                     elliptic.electric_field)).get() + abs(
                                     refs.electron_acceleration_multiplier * elliptic.magnetic_field *
                                     grids.v.arr_max),
                                 refs.electron_acceleration_multiplier * abs(elliptic.magnetic_field *
                                                                             grids.u.arr_max)]))

# plt.figure()
#                 plt.plot(time_array, self.field_energy, 'o--')
#                 plt.grid(True)
#                 plt.xlabel('Time t')
#                 plt.ylabel('Field energy')
#                 xx, uu = np.meshgrid(grids.x.arr[1:-1, :].flatten(), grids.u.arr[1:-1, :].flatten(), indexing='ij')
#                 u, v = np.meshgrid(grids.u.arr[1:-1, :].flatten(), grids.v.arr[1:-1, :].flatten(), indexing='ij')
#                 shrink = 0.75
#                 x3, u3, v3 = np.meshgrid(shrink * grids.x.arr[1:-1, :].flatten(), grids.u.arr[1:-1, :].flatten(),
#                                          grids.v.arr[1:-1, :].flatten(), indexing='ij')
#                 # Plotter
#                 p = pv.Plotter()
#                 grid = pv.StructuredGrid(x3, u3, v3)
#                 idx = 2
#                 print('Time-step is ' + str(self.dt))
#                 f0f = cp.asnumpy(distribution.arr[1:-1, :, 1:-1, :, 1:-1, :].reshape(
#                     (grids.x.res * grids.x.order, grids.u.res * grids.u.order, grids.v.res * grids.v.order)))
#                 grid["vol"] = f0f.transpose().flatten()
#                 slices = grid.slice_orthogonal(x=-15 * shrink, y=0, z=0)
#                 slices2 = grid.slice_orthogonal(x=0, y=0, z=0)
#                 slices3 = grid.slice_orthogonal(x=15 * shrink, y=0, z=0)
#                 slices4 = grid.slice_orthogonal(x=30 * shrink, y=0, z=0)
#                 slices5 = grid.slice_orthogonal(x=-30 * shrink, y=0, z=0)
#                 p.add_mesh(slices)
#                 p.add_mesh(slices2)
#                 p.add_mesh(slices3)
#                 p.add_mesh(slices4)
#                 p.add_mesh(slices5)
#                 p.show_grid()
#                 p.show()
#                 v_idx = (grids.v.res * grids.v.order) // idx
#                 x_idx = (grids.x.res * grids.x.order) // idx
#                 cb = np.linspace(np.amin(f0f[:, :, v_idx]),
#                                  np.amax(f0f[:, :, v_idx]), num=100)
#                 cb2 = np.linspace(np.amin(f0f[x_idx, :, :]),
#                                   np.amax(f0f[x_idx, :, :]), num=100)
#                 plt.figure()
#                 plt.contourf(xx, uu, f0f[:, :, v_idx], cb, extend='both')
#                 plt.colorbar()
#
#                 plt.figure()
#                 plt.contourf(u, v, f0f[x_idx, :, :], cb2, extend='both')
#                 plt.colorbar()
#
#                 plt.figure()
#                 plt.imshow(f0f[30, :, :])
#                 plt.colorbar()
#
#                 plt.show()

# Comments and other stuff I might want
# f.arr[grids.no_ghost_slice] = cp.add(f.arr[grids.no_ghost_slice],
#                                      self.coefficients[i] * explicit_advance[grids.no_ghost_slice])
# explicit_advance = cp.add(explicit_advance, self.dt * dg_flux.semi_discrete_rhs(function=explicit_advance,
#                                                                                 electric_field=elliptic.electric_field,
#                                                                                 basis=basis,
#                                                                                 grids=grids))
# dfdt = dg_flux.semi_discrete_rhs(function=explicit_advance,
#                                  electric_field=elliptic.electric_field,
#                                  basis=basis,
#                                  grids=grids)
# XXX, VVV = np.meshgrid(grids.x.arr[1:-1, :].flatten(), grids.v.arr[1:-1, :].flatten(), indexing='ij')
# XX, UU = np.meshgrid(grids.x.arr[1:-1, :].flatten(), grids.u.arr[1:-1, :].flatten(), indexing='ij')
# U, V = np.meshgrid(grids.u.arr[1:-1, :].flatten(), grids.v.arr[1:-1, :].flatten(), indexing='ij')
# idx = 2
# res = grids.x.res
# order = grids.x.order
# f0f = cp.asnumpy(dfdt[1:-1, :, 1:-1, :, 1:-1, :].reshape(
#     (res * order, res * order, res * order)))
# cb_v = np.linspace(np.amin(f0f[:, (res * order) // idx, :]), np.amax(f0f[:, (res * order) // idx, :]),
#                    num=100)
# cb_f = np.linspace(np.amin(f0f[(res * order) // idx + 2, :, :]), np.amax(f0f[(res * order) // idx + 2, :, :]),
#                    num=100)
# cb_x = np.linspace(np.amin(f0f[:, :, (res * order) // idx]), np.amax(f0f[:, :, (res * order) // idx]),
#                    num=100)
#
# plt.figure()
# plt.contourf(XX, UU, f0f[:, :, (res * order) // idx], cb_x)
# plt.xlabel('x')
# plt.ylabel('u')
# plt.colorbar()
#
# plt.figure()
# plt.contourf(U, V, f0f[(res * order) // idx + 2, :, :], cb_f)
# plt.xlabel('u')
# plt.ylabel('v')
# plt.colorbar()
# plt.show()
# quit()
# print('\n' + str(i))
# print(cp.amax(abs(dfdt)))
# print(cp.amax(explicit_advance))

# f.arr_explicit_copy += self.dt * df_dt
# Copy f after
# copy_f_after = f.arr_explicit_copy.reshape(((grids.x.res_ghosts * grids.x.order,
#                                              grids.u.res_ghosts * grids.u.order,
#                                              grids.v.res_ghosts * grids.v.order)))
# plt.figure()
# plt.imshow(df_dt_f[11, :, :].get())
# plt.title('df_dt')
# plt.figure()
# plt.imshow(copy_f_before[11, :, :].get())
# plt.title('Copy before')
# plt.figure()
# plt.imshow(copy_f_after[11, :, :].get())
# plt.title('Copy after')
# plt.show()
# quit()
# f.arr_explicit_copy[
#     grids.no_ghost_slice] += (self.dt *
#                               dg_flux.semi_discrete_rhs(function=f.arr_explicit_copy,
#                                                         elliptic=elliptic,
#                                                         basis=basis,
#                                                         grids=grids)[grids.no_ghost_slice])
# cp.cuda.Stream.null.synchronize()

# # Flats
# print(df_dt.shape)
# df_dt_f = df_dt.reshape((grids.x.res * grids.x.order,
#                          grids.u.res * grids.u.order,
#                          grids.v.res * grids.v.order))
# # print(df_dt_f.shape)
# # print(cp.amax(df_dt_f))
# print('Stage is ' + str(i))
# print(cp.amax(df_dt))
# print(cp.amin(df_dt))
# plt.figure()
# plt.imshow(df_dt_f[11, 85:100, 85:100].get())
# plt.colorbar()
# plt.show()
# copy_f_before = f.arr_explicit_copy.reshape((grids.x.res_ghosts * grids.x.order,
#                                      grids.u.res_ghosts * grids.u.order,
#                                      grids.v.res_ghosts * grids.v.order))
# df_dt[0, :, :, :, :, :] = 0
# df_dt[-1, :, :, :, :, :] = 0
# df_dt[:, :, 0, :, :, :] = 0
# df_dt[:, :, -1, :, :, :] = 0
# df_dt[:, :, :, :, 0, :] = 0
# df_dt[:, :, :, :, -1, :] = 0
# with open('new_way.npy', 'wb') as file_var:
#     np.save(file_var, df_dt.get())
# quit()
# print(cp.amax(df_dt[grids.no_ghost_slice]))
# print(cp.amin(df_dt[grids.no_ghost_slice]))
# quit()
# Try cleaning
# df_dt = cp.where(cp.absolute(df_dt) < 1.0e-6, 0.0, df_dt)

# print(dg_flux.semi_discrete_rhs(function=f.arr_explicit_copy,
#                           elliptic=elliptic,
#                           basis=basis,
#                           grids=grids).dtype)
# print(self.dt.dtype)
# print(self.coefficients[i].dtype)
# print((self.coefficients[i] * f.arr_explicit_copy[grids.no_ghost_slice]).dtype)
# print(elliptic.electric_field.dtype)
# quit()
# Perform explicit forward step based on last step
# xx, uu = np.meshgrid(grids.x.arr[1:-1, :].flatten(), grids.u.arr[1:-1, :].flatten(), indexing='ij')
# u, v = np.meshgrid(grids.u.arr[1:-1, :].flatten(), grids.v.arr[1:-1, :].flatten(), indexing='ij')
# # f0f = cp.asnumpy(distribution.arr[1:-1, :, 1:-1, :, 1:-1, :].reshape(
# #    (grids.x.res * grids.x.order, grids.u.res * grids.u.order, grids.v.res * grids.v.order)))
# idx = 2
# # vidx_x = 3
# # vidx_u =
# dfdt = dg_flux.semi_discrete_rhs(function=f.arr_explicit_copy,
#                                  elliptic=elliptic,
#                                  basis=basis,
#                                  grids=grids)
# # dfdt = cp.asnumpy(dfdt[1:-1, :, 1:-1, :, 1:-1, :].reshape(
# #    (grids.x.res * grids.x.order, grids.u.res * grids.u.order, grids.v.res * grids.v.order)))
# # print(dfdt.shape)
# # print(dfdt[23, (91-5):(91+5), (91-5):(91+5)])
# # quit()
# # dfdt = distribution.arr - f_initial
# f0f = cp.asnumpy(dfdt[1:-1, :, 1:-1, :, 1:-1, :].reshape(
#     (grids.x.res * grids.x.order, grids.u.res * grids.u.order, grids.v.res * grids.v.order)))
# v_idx = (grids.v.res * grids.v.order) // idx
# x_idx = (grids.x.res * grids.x.order) // idx
# cb = np.linspace(np.amin(f0f[:, :, v_idx]),
#                  np.amax(f0f[:, :, v_idx]), num=100)
# cb2 = np.linspace(np.amin(f0f), np.amax(f0f), num=100)
# # cb2 = np.linspace(np.amin(f0f[x_idx, :, :]),
# #                   np.amax(f0f[x_idx, :, :]), num=100)
# plt.figure()
# plt.contourf(xx, uu, f0f[:, :, v_idx], cb, extend='both')
# plt.colorbar()
#
# l = 0  # 91 - 10
# h = -1  # 91 + 10
# x_idx += 2
# cb2 = np.linspace(np.amin(f0f[x_idx, l:h, l:h]), np.amax(f0f[x_idx, l:h, l:h]), num=100)
# # print(cb2[-1])
# plt.figure()
# plt.contourf(u[l:h, l:h], v[l:h, l:h], f0f[x_idx, l:h, l:h], cb2)  # , extend='both')
# plt.colorbar()
# # print(f0f[x_idx, :, :])
# plt.show()
# quit()

# Look at ghost cells
# print('\nStep i: ' + str(self.steps_counter))
# print('Ghost:')
# print(distribution.arr[2, :, 10, :, 0, -1])
# print('Non-ghost:')
# print(distribution.arr[2, :, 10, :, 1, 0])
# print('Ghost diff:')
# print(distribution.arr[2, :, 10, :, 1, 0] - distribution.arr[2, :, 10, :, 0, -1])
# if self.steps_counter == 100:
#     quit()
# print('\nFor step ' + str(self.steps_counter))
# print('Time-step is ' + str(self.dt))

# def ssp_rk_update(self, distribution, f_last, basis, grids, elliptic, dg_flux, f_initial):
#     stages = cp.zeros((self.time_order, grids.x.res_ghosts, grids.x.order,
#                        grids.u.res_ghosts, grids.u.order, grids.v.res_ghosts, grids.v.order))
#     # Load zeroth stage
#     stages[0, :, :, :, :, :, :] = f_last
#     for idx in range(1, self.time_order):
#         # Compute zeroth moment
#         density = distribution.moment_zero_of_arr(stages[idx - 1, :, :, :, :, :, :])
#         # Compute field
#         elliptic.poisson(charge_density=density - cp.mean(density), grid=grids.x, basis=basis.b1)
#         # Compute semi-discrete RHS
#         df_dt = dg_flux.semi_discrete_rhs(function=stages[idx - 1, :, :, :, :, :, :],
#                                           elliptic=elliptic, basis=basis, grids=grids)
#         # Zero ghost cells
#         df_dt[0, :, :, :, :, :] = 0
#         df_dt[-1, :, :, :, :, :] = 0
#         df_dt[:, :, 0, :, :, :] = 0
#         df_dt[:, :, -1, :, :, :] = 0
#         df_dt[:, :, :, :, 0, :] = 0
#         df_dt[:, :, :, :, -1, :] = 0
#         # Stage update
#         stages[idx, :, :, :, :, :, :] = stages[idx - 1, :, :, :, :, :, :] + self.dt * df_dt
#     # Final step
#     density = distribution.moment_zero_of_arr(stages[-1, :, :, :, :, :, :])
#     elliptic.poisson(charge_density=density - cp.mean(density), grid=grids.x, basis=basis.b1)
#     df_dt = dg_flux.semi_discrete_rhs(function=stages[-1, :, :, :, :, :, :],
#                                       elliptic=elliptic, basis=basis, grids=grids)
#     # Zero ghost cells
#     df_dt[0, :, :, :, :, :] = 0
#     df_dt[-1, :, :, :, :, :] = 0
#     df_dt[:, :, 0, :, :, :] = 0
#     df_dt[:, :, -1, :, :, :] = 0
#     df_dt[:, :, :, :, 0, :] = 0
#     df_dt[:, :, :, :, -1, :] = 0
#     # Compute SSP-RK update
#     f_next = cp.average(stages, axis=0, weights=self.coefficients) + self.coefficients[-1] * self.dt * df_dt
#     # Sync ghost cells
#     f_next[:, :, 0, :, :, :] = f_initial[:, :, 0, :, :, :]
#     f_next[:, :, -1, :, :, :] = f_initial[:, :, -1, :, :, :]
#     f_next[:, :, :, :, 0, :] = f_initial[:, :, :, :, 0, :]
#     f_next[:, :, :, :, -1, :] = f_initial[:, :, :, :, -1, :]
#     # Difference
#     # print(f_next[:, :, 0, :, :, :] - f_initial[:, :, 0, :, :, :])
#     # quit()
#     # df = f_next - f_initial
#     # dff = df.reshape(grids.x.res_ghosts * grids.x.order,
#     #                  grids.u.res_ghosts * grids.u.order, grids.v.res_ghosts * grids.v.order)
#     # plt.figure(figsize=(10, 10))
#     # plt.imshow(cp.log(cp.absolute(dff[40, :, :]) + 1.0).get(), vmax=1.0e-8)
#     # plt.colorbar()
#     #
#     # # corners = cp.zeros_like(f_next)
#     # # corners[:, :, :, 0, :, :] = -1
#     # # corners[:, :, :, -1, :, :] = +1
#     # # corners[:, :, :, :, :, 0] = -1
#     # # corners[:, :, :, :, :, -1] = +1
#     # # corners_f = cp.asnumpy(corners.reshape(
#     # #     (grids.x.res_ghosts * grids.x.order, grids.u.res_ghosts * grids.u.order,
#     # #      grids.v.res_ghosts * grids.v.order)))
#     # # plt.figure(figsize=(10, 10))
#     # # plt.imshow(corners_f[40, :, :])
#     # # plt.colorbar()
#     #
#     # plt.show()
#     # Try syncing f_next ghost layer to f_initial
#     # f_next[:, :, 1, 0, :, :] = f_initial[:, :, 0, -1, :, :]
#     # f_next[:, :, -2, -1, :, :] = f_initial[:, :, -1, 0, :, :]
#     # f_next[:, :, :, :, 1, 0] = f_initial[:, :, :, :, 0, -1]
#     # f_next[:, :, :, :, -2, -1] = f_initial[:, :, :, :, -1, 0]
#     # print('I am at ' + str(i))
#     # plt.pause(0.1)
#     # plt.show()
#     return f_next
