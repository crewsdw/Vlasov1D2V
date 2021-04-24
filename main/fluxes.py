import numpy as np
import cupy as cp

import matplotlib.pyplot as plt


def basis_product(flux, basis_arr, axis, permutation):
    return cp.transpose(cp.tensordot(flux, basis_arr, axes=([axis], [1])),
                        axes=permutation)


class DGFlux:
    def __init__(self, resolutions, orders, flux_coefficients):
        self.resolutions = resolutions
        self.orders = orders
        # Permutations (list of tuples)
        self.permutations = [(0, 5, 1, 2, 3, 4),  # For contraction with x nodes
                             (0, 1, 2, 5, 3, 4),  # For contraction with u nodes
                             (0, 1, 2, 3, 4, 5)]  # For contraction with v nodes
        # Boundary slices (list of lists of tuples)
        self.boundary_slices = [
            # x-directed face slices [(left), (right)]
            [(slice(resolutions[0]), 0,
              slice(resolutions[1]), slice(orders[1]),
              slice(resolutions[2]), slice(orders[2])),
             (slice(resolutions[0]), -1,
              slice(resolutions[1]), slice(orders[1]),
              slice(resolutions[2]), slice(orders[2]))],
            # u-directed face slices [(left), (right)]
            [(slice(resolutions[0]), slice(orders[0]),
              slice(resolutions[1]), 0,
              slice(resolutions[2]), slice(orders[2])),
             (slice(resolutions[0]), slice(orders[0]),
              slice(resolutions[1]), -1,
              slice(resolutions[2]), slice(orders[2]))],
            # v-directed face slices [(left), (right)]
            [(slice(resolutions[0]), slice(orders[0]),
              slice(resolutions[1]), slice(orders[1]),
              slice(resolutions[2]), 0),
             (slice(resolutions[0]), slice(orders[0]),
              slice(resolutions[1]), slice(orders[1]),
              slice(resolutions[2]), -1)]]
        # Speed slices
        self.speed_slices = [(None, slice(resolutions[1]), slice(orders[1]), None, None),
                             (slice(resolutions[0]), slice(orders[0]), None, slice(resolutions[2]), slice(orders[2])),
                             (None, None, slice(resolutions[1]), slice(orders[1]), None)]

        # Grid and sub-element axes
        self.grid_axis = np.array([0, 2, 4])
        self.sub_element_axis = np.array([1, 3, 5])
        # Acceleration coefficients
        self.acceleration_coefficient = flux_coefficients
        # Numerical flux allocation size arrays
        self.num_flux_sizes = [(resolutions[0], 2, resolutions[1], orders[1], resolutions[2], orders[2]),
                               (resolutions[0], orders[0], resolutions[1], 2, resolutions[2], orders[2]),
                               (resolutions[0], orders[0], resolutions[1], orders[1], resolutions[2], 2)]

    def semi_discrete_rhs(self, function, elliptic, basis, grids):
        """
        Calculate the right-hand side of semi-discrete equation
        """
        # Debug
        # df_dt_x = (self.x_flux(function=function, basis=basis.b1, grid_u=grids.u) * grids.x.J)
        # df_dt_u = (self.u_flux(function=function, basis=basis.b2, elliptic=elliptic, grid_v=grids.v) * grids.u.J)
        # df_dt_v = (self.v_flux(function=function, basis=basis.b3, elliptic=elliptic, grid_u=grids.u) * grids.v.J)
        # df_dt_x_f = df_dt_x.reshape(self.resolutions[0] * self.orders[0], self.resolutions[1] * self.orders[1],
        #                             self.resolutions[2] * self.orders[2])
        # df_dt_u_f = df_dt_u.reshape(self.resolutions[0] * self.orders[0], self.resolutions[1] * self.orders[1],
        #                             self.resolutions[2] * self.orders[2])
        # df_dt_v_f = df_dt_v.reshape(self.resolutions[0] * self.orders[0], self.resolutions[1] * self.orders[1],
        #                             self.resolutions[2] * self.orders[2])
        # plt.figure()
        # plt.imshow(df_dt_x_f[11, :, :].get())
        # plt.colorbar()
        # plt.figure()
        # plt.imshow(df_dt_u_f[11, :, :].get())
        # plt.colorbar()
        # plt.figure()
        # plt.imshow(df_dt_v_f[11, :, :].get())
        # plt.colorbar()
        # plt.show()
        return ((self.x_flux(function=function, basis=basis.b1, grid_u=grids.u) * grids.x.J) +
                (self.u_flux(function=function, basis=basis.b2, elliptic=elliptic, grid_v=grids.v) * grids.u.J) +
                (self.v_flux(function=function, basis=basis.b3, elliptic=elliptic, grid_u=grids.u) * grids.v.J))

    def x_flux(self, function, basis, grid_u):
        dim = 0
        # Advection: compute x-directed flux as element-wise multiply
        flux = cp.multiply(function, grid_u.arr_cp[None, None, :, :, None, None])
        # Compute internal and numerical fluxes
        return (basis_product(flux=flux, basis_arr=basis.up,
                              axis=self.sub_element_axis[dim],
                              permutation=self.permutations[dim])
                - self.spatial_flux(flux=flux, speed=grid_u, basis=basis, dim=dim))

    def u_flux(self, function, basis, elliptic, grid_v):
        dim = 1
        # Lorentz force: compute u-directed speed as 4-index array
        speed = self.acceleration_coefficient * (elliptic.electric_field[:, :, None, None] +
                                                 elliptic.magnetic_field * grid_v.arr_cp[None, None, :, :])
        # then flux as element-wise multiply (does this work? test)
        flux = cp.multiply(function, speed[:, :, None, None, :, :])
        # Debug
        # ff = flux.reshape(self.resolutions[0]*self.orders[0], self.resolutions[1]*self.orders[1],
        #                   self.resolutions[2]*self.orders[2])
        # plt.figure()
        # plt.imshow(ff[11, :, :].get())
        # plt.show()
        # flux = (self.acceleration_coefficient *
        #         (cp.multiply(function, elliptic.electric_field[:, :, None, None, None, None]) +
        #          cp.multiply(function, elliptic.magnetic_field * grid_v.arr_cp[None, None, None, None, :, :]))
        #         )
        # Compute internal and numerical fluxes
        return (basis_product(flux=flux, basis_arr=basis.up,
                              axis=self.sub_element_axis[dim],
                              permutation=self.permutations[dim])
                - self.velocity_flux(flux=flux, speed=speed, basis=basis, dim=1))

    def v_flux(self, function, basis, elliptic, grid_u):
        dim = 2
        # Lorentz force: compute v-directed speed as 2-index array
        speed = self.acceleration_coefficient * (-elliptic.magnetic_field * grid_u.arr_cp)
        # then flux as element-wise multiply
        flux = cp.multiply(function, speed[None, None, :, :, None, None])
        # flux = (self.acceleration_coefficient *
        #         cp.multiply(function, -elliptic.magnetic_field * grid_u.arr_cp[None, None, :, :, None, None])
        #         )
        # Compute internal and numerical fluxes
        return (basis_product(flux=flux, basis_arr=basis.up,
                              axis=self.sub_element_axis[dim],
                              permutation=self.permutations[dim])
                - self.velocity_flux(flux=flux, speed=speed, basis=basis, dim=dim))

    # noinspection PyTypeChecker
    def spatial_flux(self, flux, speed, basis, dim):
        # Allocate
        num_flux = cp.zeros(self.num_flux_sizes[dim])
        # Debug
        # print(speed.one_positives.shape)
        # print(flux[self.boundary_slices[dim][0]].shape)
        # quit()
        # Upwind flux, left face
        num_flux[self.boundary_slices[dim][0]] = -1.0 * (cp.multiply(cp.roll(flux[self.boundary_slices[dim][1]],
                                                                             shift=1, axis=self.grid_axis[dim]),
                                                                     speed.one_positives[self.speed_slices[dim]]) +
                                                         cp.multiply(flux[self.boundary_slices[dim][0]],
                                                                     speed.one_negatives[self.speed_slices[dim]]))
        # Upwind flux, right face
        num_flux[self.boundary_slices[dim][1]] = (cp.multiply(flux[self.boundary_slices[dim][1]],
                                                              speed.one_positives[self.speed_slices[dim]]) +
                                                  cp.multiply(cp.roll(flux[self.boundary_slices[dim][0]], shift=-1,
                                                                      axis=self.grid_axis[dim]),
                                                              speed.one_negatives[self.speed_slices[dim]]))

        return basis_product(flux=num_flux, basis_arr=basis.xi,
                             axis=self.sub_element_axis[dim],
                             permutation=self.permutations[dim])

    # noinspection PyTypeChecker
    def velocity_flux(self, flux, speed, basis, dim):
        # Allocate
        num_flux = cp.zeros(self.num_flux_sizes[dim])
        # Measure upwind directions
        one_negatives = cp.where(condition=speed < 0, x=1, y=0)
        one_positives = cp.where(condition=speed >= 0, x=1, y=0)
        # Debug
        # if dim == 2:
        #     print(one_negatives.shape)
        #     print(flux[self.boundary_slices[dim][0]].shape)
        #     quit()
        # Upwind flux, left face
        num_flux[self.boundary_slices[dim][0]] = -1.0 * (cp.multiply(cp.roll(flux[self.boundary_slices[dim][1]],
                                                                             shift=1, axis=self.grid_axis[dim]),
                                                                     one_positives[self.speed_slices[dim]]) +
                                                         cp.multiply(flux[self.boundary_slices[dim][0]],
                                                                     one_negatives[self.speed_slices[dim]]))
        # Upwind flux, right face
        num_flux[self.boundary_slices[dim][1]] = (cp.multiply(flux[self.boundary_slices[dim][1]],
                                                              one_positives[self.speed_slices[dim]]) +
                                                  cp.multiply(cp.roll(flux[self.boundary_slices[dim][0]], shift=-1,
                                                                      axis=self.grid_axis[dim]),
                                                              one_negatives[self.speed_slices[dim]]))

        return basis_product(flux=num_flux, basis_arr=basis.xi,
                             axis=self.sub_element_axis[dim],
                             permutation=self.permutations[dim])

    # def stabilized_flux(self, flux, basis, dim):
    #     # Stabilization parameter
    #     alpha = 1.0
    #     param = (1.0 - alpha) / 2.0
    #     # Allocate
    #     num_flux = cp.zeros(self.num_flux_sizes[dim])
    #     # Left face, average (central flux part)
    #     num_flux[self.boundary_slices[dim][0]] = -0.5 * (cp.add(flux[self.boundary_slices[dim][0]],
    #                                                             cp.roll(flux[self.boundary_slices[dim][1]],
    #                                                                     shift=1, axis=self.grid_axis[dim])))
    #     # Left face, jump
    #     # num_flux[self.boundary_slices[dim][0]] += -1.0 * param *
    #     # (-1.0 * cp.absolute(flux[self.boundary_slices[dim][0]]) +
    #     #                                                    cp.absolute(cp.roll(flux[self.boundary_slices[dim][1]],
    #     #                                                                       shift=1, axis=self.grid_axis[dim])))
    #     # cp.cuda.runtime.deviceSynchronize()
    #     # Right face, average
    #     num_flux[self.boundary_slices[dim][1]] = 0.5 * (cp.add(flux[self.boundary_slices[dim][1]],
    #                                                            cp.roll(flux[self.boundary_slices[dim][0]],
    #                                                                    shift=-1, axis=self.grid_axis[dim])))
    #     # Right face, jump
    #     # num_flux[self.boundary_slices[dim][1]] += param * (cp.absolute(flux[self.boundary_slices[dim][1]]) -
    #     #                                                    cp.absolute(cp.roll(flux[self.boundary_slices[dim][0]],
    #     #                                                                        shift=-1, axis=self.grid_axis[dim])))
    #     # Compute product
    #     # cp.cuda.runtime.deviceSynchronize()
    #     return numerical_flux_product(flux=num_flux, basis_arr=basis.xi,
    #                                   axis=self.sub_element_axis[dim], permutation=self.permutations[dim])
    #
    # # noinspection PyTypeChecker
    # def upwind_flux2(self, flux, basis, dim):
    #     # Allocate
    #     num_flux = cp.zeros(self.num_flux_sizes[dim])
    #     # Left face
    #     num_flux[self.boundary_slices[dim][0]] = -1.0 * (cp.where(condition=flux[self.boundary_slices[dim][0]] >= 0,
    #                                                               # Where the flux on left face (0) is positive
    #                                                               x=cp.roll(flux[self.boundary_slices[dim][1]],
    #                                                                         shift=1, axis=self.grid_axis[dim]),
    #                                                               # Then use the left neighbor (-1) right face (1)
    #                                                               y=0.0) +  # else zero
    #                                                      cp.where(condition=flux[self.boundary_slices[dim][0]] < 0,
    #                                                               # Where the flux on left face (0) is negative
    #                                                               x=flux[self.boundary_slices[dim][0]],
    #                                                               # Then keep local values, else zero
    #                                                               y=0.0))
    #     # Right face
    #     num_flux[self.boundary_slices[dim][1]] = (cp.where(condition=flux[self.boundary_slices[dim][1]] >= 0,
    #                                                        # Where the flux on right face (1) is positive
    #                                                        x=flux[self.boundary_slices[dim][1]],
    #                                                        # Then use the local value, else zero
    #                                                        y=0.0) +
    #                                               cp.where(condition=flux[self.boundary_slices[dim][1]] < 0,
    #                                                        # Where the flux on right face (1) is negative
    #                                                        x=cp.roll(flux[self.boundary_slices[dim][0]],
    #                                                                  shift=-1, axis=self.grid_axis[dim]),
    #                                                        # Then use the right neighbor (-1) left face (0)
    #                                                        y=0.0))
    #     # if dim ==
    #     #     flat = num_flux[self.boundary_slices[dim][0]].reshape(self.resolutions[0]*self.orders[0],
    #     #                                                       self.resolutions[1],
    #     #                                                       self.resolutions[2]*self.orders[2])
    #     #     plt.figure()
    #     #     plt.imshow(flat[11, :, :].get())
    #     #     plt.show()
    #     #     print(num_flux.shape)
    #     #     quit()
    #     return numerical_flux_product(flux=num_flux, basis_arr=basis.xi,
    #                                   axis=self.sub_element_axis[dim], permutation=self.permutations[dim])
    #
    # # noinspection PyTypeChecker
    # def upwind_flux(self, flux, basis, dim):
    #     # Using upwind flux scheme, check sign and keep local values or use neighbors
    #     return (numerical_flux_product(flux=(cp.where(condition=flux[self.boundary_slices[dim][1]] >= 0,
    #                                                   # Where the flux on right face (1) is positive
    #                                                   x=flux[self.boundary_slices[dim][1]],
    #                                                   # Then use the local value, else zero
    #                                                   y=0.0) +
    #                                          cp.where(condition=flux[self.boundary_slices[dim][1]] < 0,
    #                                                   # Where the flux on right face (1) is negative
    #                                                   x=cp.roll(flux[self.boundary_slices[dim][0]],
    #                                                             shift=-1, axis=self.grid_axis[dim]),
    #                                                   # Then use the right neighbor (-1) left face (0)
    #                                                   y=0.0)),  # else zero
    #                                    basis_arr=basis.xi,
    #                                    face=1,  # right
    #                                    permutation=self.permutations[dim]) -
    #
    #             numerical_flux_product(flux=(cp.where(condition=flux[self.boundary_slices[dim][0]] >= 0,
    #                                                   # Where the flux on left face (0) is positive
    #                                                   x=cp.roll(flux[self.boundary_slices[dim][1]],
    #                                                             shift=1, axis=self.grid_axis[dim]),
    #                                                   # Then use the left neighbor (-1) right face (1)
    #                                                   y=0.0) +  # else zero
    #                                          cp.where(condition=flux[self.boundary_slices[dim][0]] < 0,
    #                                                   # Where the flux on left face (0) is negative
    #                                                   x=flux[self.boundary_slices[dim][0]],
    #                                                   # Then keep local values, else zero
    #                                                   y=0.0)),
    #                                    basis_arr=basis.xi,
    #                                    face=0,  # left face
    #                                    permutation=self.permutations[dim]))

# Stuff I might want later
# flux_left_positive = cp.where(flux[:, 0, :, :, :, :] > 0,
#                               cp.roll(flux[:, -1, :, :, :, :], shift=1, axis=0), 0)
# flux_left_negative = cp.where(flux[:, 0, :, :, :, :] < 0,
#                               flux[:, 0, :, :, :, :], 0)
# flux_right_positive = cp.where(flux[:, -1, :, :, :, :] > 0,
#                                flux[:, -1, :, :, :, :], 0)
# flux_right_negative = cp.where(flux[:, -1, :, :, :, :] < 0,
#                                cp.roll(flux[:, 0, :, :, :, :], shift=-1, axis=0), 0)

# flux_left = flux_left_negative + flux_left_positive
# flux_right = flux_right_negative + flux_right_positive
#
# num_flux = cp.zeros_like(flux[:, [0, -1], :, :, :, :])
# num_flux[:, 0, :, :, :, :] = -1.0 * flux_left
# num_flux[:, -1, :, :, :, :] = flux_right

# x flux
# flux = cp.multiply(function, grid_u.arr_cp[None, None, :, :, None, None])
# Compute internal and numerical fluxes
# internal = internal_flux_product(flux, basis3.b1.up, axis=1, permutation=(0, 5, 1, 2, 3, 4))

# Compute x-directed numerical flux
# sides = np.array([0, -1])
# flux_left_positive = cp.where(flux[:, 0, :, :, :, :] > 0,
#                               cp.roll(flux[:, -1, :, :, :, :], shift=1, axis=0), 0)
# flux_left_negative = cp.where(flux[:, 0, :, :, :, :] < 0,
#                               flux[:, 0, :, :, :, :], 0)
# flux_right_positive = cp.where(flux[:, -1, :, :, :, :] > 0,
#                                flux[:, -1, :, :, :, :], 0)
# flux_right_negative = cp.where(flux[:, -1, :, :, :, :] < 0,
#                                cp.roll(flux[:, 0, :, :, :, :], shift=-1, axis=0), 0)
#
# flux_left = flux_left_negative + flux_left_positive
# flux_right = flux_right_negative + flux_right_positive
#
# num_flux = cp.zeros_like(flux[:, [0, -1], :, :, :, :])
# num_flux[:, 0, :, :, :, :] = -1.0 * flux_left
# num_flux[:, -1, :, :, :, :] = flux_right
# print(num_flux.shape)

# boundary_flux = cp.transpose(cp.tensordot(num_flux,
#                                           basis3.b1.xi,
#                                           axes=([1], [1])),
#                              (0, 5, 1, 2, 3, 4))

# num_flux_left = -1.0 * cp.transpose(cp.tensordot(flux_left,
#                                                  basis3.b1.xi[:, 0], axes=0),
#                                     (0, 5, 1, 2, 3, 4))
# num_flux_right = cp.transpose(cp.tensordot(flux_right,
#                                            basis3.b1.xi[:, 1], axes=0),
#                               (0, 5, 1, 2, 3, 4))
# boundary_flux = num_flux_right + num_flux_left

# # # # Debug
# a = (cp.where(condition=flux[self.boundary_slices[dim][1]] >= 0,
#               # Where the flux on right face (1) is positive
#               x=flux[self.boundary_slices[dim][1]],
#               # Then use the local value, else zero
#               y=0.0) +
#      cp.where(condition=flux[self.boundary_slices[dim][1]] < 0,
#               # Where the flux on right face (1) is negative
#               x=cp.roll(flux[self.boundary_slices[dim][0]],
#                         shift=-1, axis=self.grid_axis[dim]),
#               # Then use the neighbor left face (0), else zero
#               y=0.0))
# b = (cp.where(condition=flux[self.boundary_slices[dim][0]] >= 0,
#               # Where the flux on left face (0) is positive
#               x=cp.roll(flux[self.boundary_slices[dim][1]],
#                         shift=1, axis=self.grid_axis[dim]),
#               # Then use the neighbor's right face (1), else zero
#               y=0.0) +
#      cp.where(condition=flux[self.boundary_slices[dim][0]] < 0,
#               # Where the flux on left face (0) is negative
#               x=flux[self.boundary_slices[dim][0]],
#               # Then keep local values, else zero
#               y=0.0))
# a1 = cp.zeros_like(flux)
# a2 = cp.zeros_like(flux)
# if dim == 1:
#     a1[self.boundary_slices[dim][1]] = (cp.where(flux[self.boundary_slices[dim][1]] > 0,
#                                                  # Where the flux on right face (1) is positive
#                                                  flux[self.boundary_slices[dim][1]],
#                                                  # Then use the local value, else zero
#                                                  0.0) +
#                                         cp.where(flux[self.boundary_slices[dim][1]] < 0,
#                                                  # Where the flux on right face (1) is negative
#                                                  cp.roll(flux[self.boundary_slices[dim][0]],
#                                                          shift=-1, axis=self.grid_axis[dim]),
#                                                  # Then use the neighbor left face (0), else zero
#                                                  0.0))
#
#     a2[self.boundary_slices[dim][0]] = (cp.where(flux[self.boundary_slices[dim][0]] >= 0,
#                                                  # Where the flux on left face (0) is positive
#                                                  cp.roll(flux[self.boundary_slices[dim][1]],
#                                                          shift=1, axis=self.grid_axis[dim]),
#                                                  # Then use the neighbor's right face (1), else zero
#                                                  0.0) +
#                                         cp.where(flux[self.boundary_slices[dim][0]] < 0,
#                                                  # Where the flux on left face (0) is negative
#                                                  flux[self.boundary_slices[dim][0]],
#                                                  # Then keep local values, else zero
#                                                  0.0))
#
#     plt.figure()
#     plt.contourf(a1.reshape(self.resolutions[0] * self.orders[0],
#                             self.resolutions[1] * self.orders[1],
#                             self.resolutions[2] * self.orders[2])[20, :, :], levels=200)
#     plt.colorbar()
#
#     plt.figure()
#     plt.contourf(a2.reshape(self.resolutions[0] * self.orders[0],
#                             self.resolutions[1] * self.orders[1],
#                             self.resolutions[2] * self.orders[2])[20, :, :], levels=200)
#     plt.colorbar()  # [20, 10:14, 90:110], levels=200)[20, 95:105, 95:105]
#     plt.show()

# flux = (self.acceleration_coefficient *
#         (cp.multiply(function, elliptic.electric_field[:, :, None, None, None, None]) +
#          cp.multiply(function, elliptic.magnetic_field * grid_v.arr_cp[None, None, None, None, :, :]))
#         )
# corners = cp.zeros_like(flux)
# corners[:, :, :, 0, :, :] = -1
# corners[:, :, :, -1, :, :] = +1
# corners[:, :, :, :, :, 0] = -1
# corners[:, :, :, :, :, -1] = +1
# flux_f = cp.asnumpy(flux[1:-1, :, 1:-1, :, 1:-1, :].reshape(6 * 8, 24 * 8, 24 * 8))
# corners_f = cp.asnumpy(corners[1:-1, :, 1:-1, :, 1:-1, :].reshape(
#     (6 * 8, 24 * 8, 24 * 8)))
# plt.figure()
# plt.imshow(flux_f[11, 90:102, 90:102])
# plt.colorbar()
# plt.figure()
# plt.imshow(corners_f[11, 90:102, 90:102])
# plt.colorbar()
# plt.show()
# cp.cuda.Stream.null.synchronize()

# Bin
#
# def internal_flux_product(flux, basis_arr, axis, permutation):
#     # Steps: contract flux with basis array, then permute indices back
#     return cp.transpose(cp.tensordot(flux, basis_arr,
#                                      axes=([axis], [1])),
#                         axes=permutation)


# def internal_flux_product(flux, basis_arr, axis, permutation):
#     # Steps: contract flux with basis array, then permute indices back
#     return cp.transpose(cp.tensordot(flux, basis_arr,
#                                      axes=([axis], [1])),
#                         axes=permutation)

# def numerical_flux_product(flux, basis_arr, face, permutation):
#     # Contract flux with basis array, then permute indices back
#     return cp.transpose(cp.tensordot(flux, basis_arr[:, face],
#                                      axes=0),
#                         axes=permutation)


# def numerical_flux_product(flux, basis_arr, axis, permutation):
#     # Contract flux with basis array and permute indices back
#     # print(flux.shape)
#     # print(basis_arr.shape)
#     # print(axis)
#     # quit()
#     return cp.transpose(cp.tensordot(flux, basis_arr, axes=([axis], [1])),
#                         axes=permutation)
