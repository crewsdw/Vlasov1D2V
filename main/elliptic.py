import numpy as np
import cupy as cp


# For debug
# import matplotlib.pyplot as plt


class Elliptic:
    def __init__(self, poisson_coefficient):
        # Operators
        self.central_flux_operator = None
        self.gradient_operator = None
        self.inv_op = None

        # Fields
        self.potential = None
        self.electric_field = None
        self.magnetic_field = None

        # Charge density coefficient in poisson equation
        self.poisson_coefficient = poisson_coefficient

    def build_central_flux_operator(self, grid, basis):
        # Build using indicating array
        indicator = np.zeros((grid.res, grid.order))
        # face differences for numerical flux
        face_diff0 = np.zeros((grid.res, 2))
        face_diff1 = np.zeros((grid.res, 2))
        num_flux = np.zeros_like(face_diff0)
        grad_num_flux = np.zeros_like(face_diff1)

        central_flux_operator = np.zeros((grid.res, grid.order, grid.res, grid.order))
        self.gradient_operator = np.zeros_like(central_flux_operator)

        for i in range(grid.res):
            for j in range(grid.order):
                # Choose node
                indicator[i, j] = 1.0

                # Compute strong form boundary flux (central)
                face_diff0[:, 0] = indicator[:, 0] - np.roll(indicator[:, -1], 1)
                face_diff0[:, 1] = indicator[:, -1] - np.roll(indicator[:, 0], -1)
                # face_diff0[:, 0] = (0.5 * (indicator[:, 0] + np.roll(indicator[:, -1], 1)) -
                #                    indicator[:, 0])
                # face_diff0[:, 1] = -1.0 * (0.5 * (indicator[:, -1] + np.roll(indicator[:, 0], -1)) -
                #                           indicator[:, -1])
                num_flux[:, 0] = 0.5 * face_diff0[:, 0]
                num_flux[:, 1] = -0.5 * face_diff0[:, 1]

                # Compute gradient of this node
                grad = (np.tensordot(basis.der, indicator, axes=([1], [1])) +
                        np.tensordot(basis.np_xi, num_flux, axes=([1], [1]))).T

                # Compute gradient's numerical flux (central)
                face_diff1[:, 0] = grad[:, 0] - np.roll(grad[:, -1], 1)
                face_diff1[:, 1] = grad[:, -1] - np.roll(grad[:, 0], -1)
                grad_num_flux[:, 0] = 0.5 * face_diff1[:, 0]
                grad_num_flux[:, 1] = -0.5 * face_diff1[:, 1]

                # Compute operator from gradient matrix
                operator = (np.tensordot(basis.stf, grad, axes=([1], [1])) +
                            np.tensordot(basis.face_mass, grad_num_flux + face_diff0, axes=([1], [1]))).T

                # place this operator in the global matrix
                central_flux_operator[i, j, :, :] = operator
                self.gradient_operator[i, j, :, :] = grad

                # reset nodal indicator
                indicator[i, j] = 0

        # Reshape to matrix and set gauge condition by fixing quadrature integral = 0 as extra equation in system
        op0 = np.hstack([central_flux_operator.reshape(grid.res * grid.order, grid.res * grid.order),
                         grid.quad_weights.get().reshape(grid.res * grid.order, 1)])
        self.central_flux_operator = np.vstack([op0, np.append(grid.quad_weights.get().flatten(), 0)])
        # Clear machine errors
        self.central_flux_operator[np.abs(self.central_flux_operator) < 1.0e-15] = 0

        # Send gradient operator to device
        self.gradient_operator = cp.asarray(self.gradient_operator)

    def invert(self):
        self.inv_op = cp.asarray(np.linalg.inv(self.central_flux_operator))

    def poisson(self, charge_density, grid, basis, anti_alias=True):
        """
        Poisson solve in 1D using stabilized central flux
        """
        # Preprocess (last entry is average value)
        rhs = cp.zeros((grid.res * grid.order + 1))
        rhs[:-1] = self.poisson_coefficient * cp.tensordot(charge_density, basis.d_mass, axes=([1], [1])).flatten()

        # Compute solution and remove last entry
        sol = cp.matmul(self.inv_op, rhs)[:-1] / (grid.J ** 2.0)
        self.potential = sol.reshape(grid.res, grid.order)

        # Clean solution (anti-alias)
        if anti_alias:
            coefficients = grid.fourier_basis(self.potential)
            self.potential = grid.sum_fourier(coefficients)

        # Compute field as negative potential gradient
        self.electric_field = cp.zeros_like(grid.arr_cp)
        self.electric_field[1:-1, :] = -1.0*(grid.J *
                                             cp.tensordot(self.gradient_operator,
                                                          self.potential, axes=([0, 1], [0, 1])))

        # Clean solution (anti-alias)
        if anti_alias:
            coefficients = grid.fourier_basis(self.electric_field[1:-1, :])
            self.electric_field[1:-1, :] = grid.sum_fourier(coefficients)

        # Set ghost cells
        self.electric_field[0, :] = self.electric_field[-2, :]
        self.electric_field[-1, :] = self.electric_field[1, :]

    def poisson2(self, charge_density, grid, basis):
        """ Alternative Poisson solve using Fourier spectral basis """
        # Pre-process charge density
        rhs = self.poisson_coefficient * charge_density
        # Get Fourier spectral basis
        rhs_coefficients = grid.fourier_basis(rhs)
        # Compute spectral solution coefficients
        field_coefficients = -cp.divide(rhs_coefficients, (1j * grid.d_wave_numbers))  # where=grid.wave_numbers != 0)
        potential_coefficients = -cp.divide(rhs_coefficients, grid.d_wave_numbers ** 2.0)
        # Clear "inf"
        field_coefficients = cp.nan_to_num(field_coefficients) # field_coefficients[field_coefficients == cp.inf] = 0
        potential_coefficients = cp.nan_to_num(potential_coefficients)
        # potential_coefficients[potential_coefficients == cp.inf] = 0

        # Compute field and potential as inverse fourier transform
        self.electric_field = cp.zeros_like(grid.arr_cp)
        self.electric_field[1:-1, :] = grid.sum_fourier(field_coefficients)
        self.potential = grid.sum_fourier(potential_coefficients)

        # Debug
        # print(field_coefficients)
        # print(self.electric_field[2, :])
        # plt.figure()
        # plt.plot(grid.arr.flatten(), self.electric_field.flatten().get(), 'o--')
        # plt.show()

        # Set ghost cells
        self.electric_field[0, :] = self.electric_field[-2, :]
        self.electric_field[-1, :] = self.electric_field[1, :]

    def set_magnetic_field(self, magnetic_field):
        self.magnetic_field = magnetic_field

    def electric_energy(self, grid):
        return cp.tensordot(self.electric_field[1:-1, :] ** 2.0, grid.quad_weights, axes=([0, 1], [0, 1]))

        # Compare for debug
        # true_p = -charge_density / np.pi ** 2.0
        # true_f = cp.asarray(-0.01 * np.cos(np.pi * grid.arr[1:-1, :]) / np.pi)
        # err_p_r = (potential - true_p)
        # err_p_c = (self.potential - true_p)
        # err_f_r = (electric_field - true_f)
        # err_f_c = (self.electric_field - true_f)
        #
        # plt.figure()
        # plt.plot(grid.arr[1:-1, :].flatten(), err_p_r.get().flatten(), 'o--', label='raw potential')
        # plt.plot(grid.arr[1:-1, :].flatten(), err_p_c.get().flatten(), 'o--', label='cleaned potential')
        # plt.legend(loc='best')
        # plt.ylabel('Error')
        # plt.xlabel('Position')
        #
        # plt.figure()
        # plt.plot(grid.arr[1:-1, :].flatten(), err_f_r.get().flatten(), 'o--', label='raw electric_field')
        # plt.plot(grid.arr[1:-1, :].flatten(), err_f_c.get().flatten(), 'o--', label='cleaned electric_field')
        # plt.legend(loc='best')
        # plt.ylabel('Error')
        # plt.xlabel('Position')
        #
        # plt.show()
