import numpy as np
import cupy as cp
import copy

# For debug
# import matplotlib.pyplot as plt


# noinspection PyTypeChecker
class Grid1D:
    def __init__(self, low, high, res, basis, spectrum=False):
        self.low = low
        self.high = high
        self.res = int(res)  # somehow gets non-int...
        self.res_ghosts = int(res + 2)  # resolution including ghosts
        self.order = basis.order

        # domain and element widths
        self.length = self.high - self.low
        self.dx = self.length / self.res

        # element Jacobian
        self.J = 2.0 / self.dx

        # The grid does not have a basis but does have quad weights
        self.quad_weights = cp.tensordot(cp.ones(self.res), cp.asarray(basis.weights), axes=0)
        # arrays
        self.arr = self.create_grid(basis.nodes)
        self.arr_cp = cp.asarray(self.arr)
        self.midpoints = np.array([(self.arr[i, -1] + self.arr[i, 0]) / 2.0 for i in range(1, self.res_ghosts - 1)])
        self.arr_max = np.amax(abs(self.arr))

        # velocity axis gets a positive/negative indexing slice
        self.one_negatives = cp.where(condition=self.arr_cp < 0, x=1, y=0)
        self.one_positives = cp.where(condition=self.arr_cp >= 0, x=1, y=0)

        # spectral coefficients
        if spectrum:
            self.nyquist_number = self.length // self.dx  # mode number of nyquist frequency
            self.k1 = 2.0 * np.pi / self.length  # fundamental mode
            self.wave_numbers = self.k1 * np.arange(1 - self.nyquist_number, self.nyquist_number)
            self.grid_phases = cp.asarray(np.exp(1j * np.tensordot(self.wave_numbers, self.arr[1:-1, :], axes=0)))

            # Spectral matrices
            self.spectral_transform = basis.fourier_transform_array(self.midpoints, self.J, self.wave_numbers)

    def create_grid(self, nodes):
        """
        Initialize array of global coordinates (including ghost elements).
        """
        # shift to include ghost cells
        min_gs = self.low - self.dx
        max_gs = self.high  # + self.dx
        # nodes (iso-parametric)
        nodes = (np.array(nodes) + 1) / 2

        # element left boundaries (including ghost elements)
        xl = np.linspace(min_gs, max_gs, num=self.res_ghosts)

        # construct coordinates
        self.arr = np.zeros((self.res_ghosts, self.order))
        for i in range(self.res_ghosts):
            self.arr[i, :] = xl[i] + self.dx * nodes

        return self.arr

    def grid2cp(self):
        self.arr = cp.asarray(self.arr)

    def grid2np(self):
        self.arr = self.arr.get()

    def fourier_basis(self, function):
        """
        On GPU, compute Fourier coefficients on the LGL grid of the given grid function
        """
        return cp.tensordot(function, self.spectral_transform, axes=([0, 1], [0, 1])) * self.dx / self.length

    def sum_fourier(self, coefficients):
        """
        On GPU, re-sum Fourier coefficients up to pre-set cutoff
        """
        return cp.real(cp.tensordot(coefficients, self.grid_phases, axes=([0], [0])))


class Grid3D:
    def __init__(self, basis, lows, highs, resolutions):
        # Grids
        self.x = Grid1D(low=lows[0], high=highs[0], res=resolutions[0], basis=basis.b1, spectrum=True)
        self.u = Grid1D(low=lows[1], high=highs[1], res=resolutions[1], basis=basis.b2)
        self.v = Grid1D(low=lows[2], high=highs[2], res=resolutions[2], basis=basis.b3)
        # No ghost slice
        self.no_ghost_slice = (slice(1, self.x.res_ghosts - 1), slice(self.x.order),
                               slice(1, self.u.res_ghosts - 1), slice(self.u.order),
                               slice(1, self.v.res_ghosts - 1), slice(self.v.order))
        # self.no_ghost_slice = (slice(1, self.x.res_ghosts - 1), slice(self.x.order),
        #                        slice(2, self.u.res_ghosts - 2), slice(self.u.order),
        #                        slice(2, self.v.res_ghosts - 2), slice(self.v.order))
        # self.no_ghost_slice = (slice(self.x.res_ghosts), slice(self.x.order),
        #                        slice(self.u.res_ghosts), slice(self.u.order),
        #                        slice(self.v.res_ghosts), slice(self.v.order))


class Distribution:
    def __init__(self, vt, ring_j, resolutions, orders):
        # parameters
        self.ring_j = ring_j  # ring parameter
        self.vt = vt  # thermal velocity
        # array, initialize later
        self.arr = None
        self.arr_explicit_copy = None
        self.arr_next = None

        # resolutions (no ghosts)
        self.x_res, self.u_res, self.v_res = resolutions[0], resolutions[1], resolutions[2]

        # orders
        self.x_ord, self.u_ord, self.v_ord = orders[0], orders[1], orders[2]

        # velocity-space quad weights init
        self.quad_weights = None

        # size0 = slice(1, resolutions[0] + 1)
        # size1 = slice(1, resolutions[1] + 1)
        # size2 = slice(1, resolutions[2] + 1)
        size0 = slice(resolutions[0] + 2)
        size1 = slice(resolutions[1] + 2)
        size2 = slice(resolutions[2] + 2)

        self.boundary_slices = [
            # x-directed face slices [(left), (right)]
            [(size0, 0, size1, slice(orders[1]), size2, slice(orders[2])),
             (size0, -1, size1, slice(orders[1]), size2, slice(orders[2]))],
            [(size0, slice(orders[0]), size1, 0, size2, slice(orders[2])),
             (size0, slice(orders[0]), size1, -1, size2, slice(orders[2]))],
            [(size0, slice(orders[0]), size1, slice(orders[1]), size2, 0),
             (size0, slice(orders[0]), size1, slice(orders[1]), size2, -1)]]
        # Grid and sub-element axes
        self.grid_axis = np.array([0, 2, 4])
        self.sub_element_axis = np.array([1, 3, 5])
    # def initialize_cpu(self, gridx, gridu, gridv):
    #     """
    #     Initialize distribution function as polar eigenfunction on CPU
    #     """
    #     # Normalization factor
    #     factor = 1.0 / (2 ** self.ring_j * (2.0 * np.pi) * (self.vt ** 2.0) * np.math.factorial(self.ring_j))
    #
    #     # Build distribution through tensor products
    #     # Grid indicators
    #     ix = np.tensordot(np.ones(gridx.res_ghosts), np.ones(gridx.order), axes=0)
    #     iu = np.tensordot(np.ones(gridu.res_ghosts), np.ones(gridu.order), axes=0)
    #     iv = np.tensordot(np.ones(gridv.res_ghosts), np.ones(gridv.order), axes=0)
    #
    #     # Build gaussian
    #     rsq = (np.tensordot(gridu.arr ** 2.0, iv, axes=0) +
    #            np.tensordot(iu, gridv.arr ** 2.0, axes=0)) / (self.vt ** 2.0)
    #     radj = rsq ** self.ring_j  # parameter-weighted radius
    #     gauss = np.exp(-0.5 * rsq)  # gaussian
    #     poleig = radj * gauss  # polar eigenstates
    #     self.arr = np.transpose(factor * np.tensordot(ix, poleig, axes=0), (0, 2, 4, 1, 3, 5))

    def initialize_next(self):
        self.arr_next = cp.zeros_like(self.arr)

    def reset_array(self, grids):
        # self.arr[grids.no_ghost_slice] = copy.deepcopy(self.arr_next)[grids.no_ghost_slice]
        # self.arr[grids.no_ghost_slice] = cp.copy(self.arr_next)[grids.no_ghost_slice]
        self.arr[grids.no_ghost_slice] = cp.ascontiguousarray(self.arr_next[grids.no_ghost_slice])
        # self.arr[self.arr < 0] = 0.0

    def swap_ghost_cells_of_array(self):
        self.arr[0, :, :, :, :, :] = cp.copy(self.arr[-2, :, :, :, :, :])
        self.arr[-1, :, :, :, :, :] = cp.copy(self.arr[1, :, :, :, :, :])

    def swap_ghost_cells_of_copy(self):
        self.arr_explicit_copy[0, :, :, :, :, :] = cp.copy(self.arr_explicit_copy[-2, :, :, :, :, :])
        self.arr_explicit_copy[-1, :, :, :, :, :] = cp.copy(self.arr_explicit_copy[1, :, :, :, :, :])

    # def clean_copy(self):
    #     self.arr_explicit_copy = cp.zeros_like(self.arr)

    def copy_for_explicit_advance(self):
        self.arr_explicit_copy = cp.copy(self.arr)  # copy.deepcopy(self.arr)  # cp.copy(self.arr)

    def initialize_gpu(self, grids):
        """
        Initialize distribution function as polar eigenfunction on GPU
        """
        # As CuPy arrays
        grids.x.grid2cp()
        grids.u.grid2cp()
        grids.v.grid2cp()
        # Normalization factor
        factor = 1.0 / ((2.0 ** self.ring_j) * (2.0 * np.pi) * (self.vt ** 2.0) * np.math.factorial(self.ring_j))

        # Build distribution through tensor products
        # Grid indicators
        ix = cp.tensordot(cp.ones(grids.x.res_ghosts), cp.ones(grids.x.order), axes=0)
        iu = cp.tensordot(cp.ones(grids.u.res_ghosts), cp.ones(grids.u.order), axes=0)
        iv = cp.tensordot(cp.ones(grids.v.res_ghosts), cp.ones(grids.v.order), axes=0)

        # add perturbation
        ix += 0.01 * cp.sin(grids.x.k1 * grids.x.arr)

        # Build gaussian
        rsq = (cp.tensordot(grids.u.arr ** 2.0, iv, axes=0) +
               cp.tensordot(iu, grids.v.arr ** 2.0, axes=0)) / (self.vt ** 2.0)

        # plt.figure()
        # plt.imshow(rsq.reshape(iu.shape[0]*iu.shape[1], iv.shape[0]*iv.shape[1]).get())
        # plt.show()
        # rsq = (cp.tensordot(grids.u.arr.flatten() ** 2.0, iv.flatten(), axes=0) +
        #        cp.tensordot(iu.flatten(), grids.v.arr.flatten() ** 2.0, axes=0))
        # print(rsq[:25, :25])
        #
        # plt.figure()
        # plt.imshow(rsq.get())
        # plt.show()

        rad_j = rsq ** self.ring_j  # parameter-weighted radius
        gauss = cp.exp(-0.5 * rsq)  # gaussian
        pol_eig = cp.multiply(rad_j, gauss)  # polar eigenstates
        self.arr = factor * cp.tensordot(ix, pol_eig, axes=0)
        # self.grid_order_gpu()

        # Send grids back to host
        grids.x.grid2np()
        grids.u.grid2np()
        grids.v.grid2np()

    def grid_order_gpu(self):
        """
        Permute from (Nx, x_ord, Nu, u_ord, Nv, v_ord) --> (Nx, Nu, Nv, x_ord, u_ord, v_ord)
        """
        self.arr = cp.transpose(self.arr, (0, 2, 4, 1, 3, 5))

    def tensor_product_order_gpu(self):
        """
        Permute from (Nx, Nu, Nv, x_ord, u_ord, v_ord) --> (Nx, x_ord, Nu, u_ord, Nv, v_ord)
        """
        self.arr = cp.transpose(self.arr, (0, 3, 1, 4, 2, 5))

    def grid_flatten_cpu(self):
        rs = np.transpose(self.arr[1:-1, 1:-1, 1:-1, :, :, :], (0, 3, 1, 4, 2, 5))
        return rs.reshape((self.x_res * self.x_ord, self.u_res * self.u_ord, self.v_res * self.v_ord))

    def grid_flatten_gpu(self):
        # rs = cp.transpose(self.arr[1:-1, 1:-1, 1:-1, :, :, :], (0, 3, 1, 4, 2, 5))
        # return self.arr[1:-1, :, 1:-1, :, 1:-1, :].reshape(
        #     (self.x_res * self.x_ord, self.u_res * self.u_ord, self.v_res * self.v_ord))
        return self.arr[:, :, :, :, :, :].reshape(
            ((self.x_res + 2) * self.x_ord, (self.u_res + 2) * self.u_ord, (self.v_res + 2) * self.v_ord))

    def initialize_quad_weights(self, grids):
        """
        Initialize the velocity-space quadrature weights
        """
        self.quad_weights = cp.tensordot(grids.u.quad_weights, grids.v.quad_weights, axes=0) / (grids.u.J * grids.v.J)

    def moment_zero(self):
        """
        Compute zeroth moment on gpu
        """
        # Permute pdf array to natural tensor product order
        # self.tensor_product_order_gpu()
        # Compute quadrature as tensor contraction on index pairs, avoiding ghost cells ([1:-1] etc.)
        moment = cp.tensordot(self.arr[1:-1, :, 1:-1, :, 1:-1, :],
                              self.quad_weights, axes=([2, 3, 4, 5], [0, 1, 2, 3]))
        # Permute pdf array back to grid order
        # self.grid_order_gpu()

        # Return zeroth moment
        return moment  # / 1.0e12

    def moment_zero_of_copy(self):
        # Compute quadrature as tensor contraction on index pairs, avoiding ghost cells ([1:-1] etc.)
        return cp.tensordot(self.arr_explicit_copy[1:-1, :, 1:-1, :, 1:-1, :],
                            self.quad_weights, axes=([2, 3, 4, 5], [0, 1, 2, 3]))  # / 1.0e12

    def moment_zero_of_arr(self, arr):
        return cp.tensordot(arr[1:-1, :, 1:-1, :, 1:-1, :],
                            self.quad_weights, axes=([2, 3, 4, 5], [0, 1, 2, 3]))

    # A desperate repair function (wtf...?!)
    # def sync_edges(self):
        # print('I synced it...!')

        # for dim in range(3):
        #     self.arr[self.boundary_slices[dim][1]] = 0.5 * (self.arr[self.boundary_slices[dim][1]] +
        #                                                 cp.roll(self.arr,
        #                                             shift=-1, axis=self.grid_axis[dim])[self.boundary_slices[dim][0]])
        #     self.arr[self.boundary_slices[dim][0]] = 0.5 * (self.arr[self.boundary_slices[dim][0]] +
        #                                                 cp.roll(self.arr, shift=1,
        #                                             axis=self.grid_axis[dim])[self.boundary_slices[dim][1]])


class DistributionNumpy:
    def __init__(self, vt, ring_j, resolutions, orders):
        # parameters
        self.ring_j = ring_j  # ring parameter
        self.vt = vt  # thermal velocity
        # array, initialize later
        self.arr = None
        self.arr_explicit_copy = None

        # resolutions (no ghosts)
        self.x_res, self.u_res, self.v_res = resolutions[0], resolutions[1], resolutions[2]

        # orders
        self.x_ord, self.u_ord, self.v_ord = orders[0], orders[1], orders[2]

        # velocity-space quad weights init
        self.quad_weights = None

    def swap_ghost_cells_of_array(self):
        self.arr[0, :, 1:-1, :, 1:-1, :] = copy.deepcopy(self.arr[-2, :, 1:-1, :, 1:-1, :])
        self.arr[-1, :, 1:-1, :, 1:-1, :] = copy.deepcopy(self.arr[1, :, 1:-1, :, 1:-1, :])

    def swap_ghost_cells_of_copy(self):
        self.arr_explicit_copy[0, :, 1:-1, :, 1:-1, :] = copy.deepcopy(self.arr_explicit_copy[-2, :, 1:-1, :, 1:-1, :])
        self.arr_explicit_copy[-1, :, 1:-1, :, 1:-1, :] = copy.deepcopy(self.arr_explicit_copy[1, :, 1:-1, :, 1:-1, :])

    def clean_copy(self):
        self.arr_explicit_copy = np.zeros_like(self.arr)

    def copy_for_explicit_advance(self):
        self.arr_explicit_copy = copy.deepcopy(self.arr)  # cp.copy(self.arr)

    def initialize_gpu(self, grids):
        """
        Initialize distribution function as polar eigenfunction on GPU
        """
        # As CuPy arrays
        grids.x.grid2cp()
        grids.u.grid2cp()
        grids.v.grid2cp()
        # Normalization factor
        factor = 1.0 / (2 ** self.ring_j * (2.0 * np.pi) * (self.vt ** 2.0) * np.math.factorial(self.ring_j))

        # Build distribution through tensor products
        # Grid indicators
        ix = cp.tensordot(cp.ones(grids.x.res_ghosts), cp.ones(grids.x.order), axes=0)
        iu = cp.tensordot(cp.ones(grids.u.res_ghosts), cp.ones(grids.u.order), axes=0)
        iv = cp.tensordot(cp.ones(grids.v.res_ghosts), cp.ones(grids.v.order), axes=0)

        # add perturbation
        # ix += 0.01 * cp.sin(grids.x.k1 * grids.x.arr)

        # Build gaussian
        rsq = (cp.tensordot(grids.u.arr ** 2.0, iv, axes=0) +
               cp.tensordot(iu, grids.v.arr ** 2.0, axes=0)) / (self.vt ** 2.0)
        rad_j = rsq ** self.ring_j  # parameter-weighted radius
        gauss = cp.exp(-0.5 * rsq)  # gaussian
        pol_eig = rad_j * gauss  # polar eigenstates
        self.arr = factor * cp.tensordot(ix, pol_eig, axes=0).get()
        # self.grid_order_gpu()

        # Send grids back to host
        grids.x.grid2np()
        grids.u.grid2np()
        grids.v.grid2np()

    def grid_order_gpu(self):
        """
        Permute from (Nx, nord, Nu, u_ord, Nv, v_ord) --> (Nx, Nu, Nv, nord, u_ord, v_ord)
        """
        self.arr = cp.transpose(self.arr, (0, 2, 4, 1, 3, 5))

    def tensor_product_order_gpu(self):
        """
        Permute from (Nx, Nu, Nv, nord, u_ord, v_ord) --> (Nx, nord, Nu, u_ord, Nv, v_ord)
        """
        self.arr = cp.transpose(self.arr, (0, 3, 1, 4, 2, 5))

    def grid_flatten_cpu(self):
        rs = np.transpose(self.arr[1:-1, 1:-1, 1:-1, :, :, :], (0, 3, 1, 4, 2, 5))
        return rs.reshape((self.x_res * self.x_ord, self.u_res * self.u_ord, self.v_res * self.v_ord))

    def grid_flatten_gpu(self):
        # rs = cp.transpose(self.arr[1:-1, 1:-1, 1:-1, :, :, :], (0, 3, 1, 4, 2, 5))
        # return self.arr[1:-1, :, 1:-1, :, 1:-1, :].reshape(
        #     (self.x_res * self.x_ord, self.u_res * self.u_ord, self.v_res * self.v_ord))
        return self.arr[:, :, :, :, :, :].reshape(
            ((self.x_res + 2) * self.x_ord, (self.u_res + 2) * self.u_ord, (self.v_res + 2) * self.v_ord))

    def initialize_quad_weights(self, grids):
        """
        Initialize the velocity-space quadrature weights
        """
        self.quad_weights = cp.tensordot(grids.u.quad_weights,
                                         grids.v.quad_weights, axes=0).get() / (grids.u.J * grids.v.J)

    def moment_zero(self):
        """
        Compute zeroth moment on gpu
        """
        # Permute pdf array to natural tensor product order
        # self.tensor_product_order_gpu()
        # Compute quadrature as tensor contraction on index pairs, avoiding ghost cells ([1:-1] etc.)
        moment = np.tensordot(self.arr[1:-1, :, 1:-1, :, 1:-1, :],
                              self.quad_weights, axes=([2, 3, 4, 5], [0, 1, 2, 3]))
        # Permute pdf array back to grid order
        # self.grid_order_gpu()

        # Return zeroth moment
        return moment  # / 1.0e12

    def moment_zero_of_copy(self):
        # Compute quadrature as tensor contraction on index pairs, avoiding ghost cells ([1:-1] etc.)
        return np.tensordot(self.arr_explicit_copy[1:-1, :, 1:-1, :, 1:-1, :],
                            self.quad_weights, axes=([2, 3, 4, 5], [0, 1, 2, 3]))  # / 1.0e12

    def moment_zero_of_arr(self, arr):
        return np.tensordot(arr[1:-1, :, 1:-1, :, 1:-1, :],
                            self.quad_weights, axes=([2, 3, 4, 5], [0, 1, 2, 3]))