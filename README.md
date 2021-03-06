# [Vlasov1D2V](https://github.com/crewsdw/Vlasov1D2V/)

This collection of Python codes solves the Vlasov-Poisson system in 1D+2V using the Runge-Kutta Discontinuous Galerkin finite element method using CUDA-accelerated libraries, namely [CuPy](https://github.com/cupy/cupy). The solver is capable of up to 9th order spatial and 3rd order temporal accuracy.

[Some example solutions are found here](https://students.washington.edu/dcrews/gallery.html)

## Abstract:
> The discontinuous Galerkin (DG) operators in the Lobatto basis are constructed to
arbitrary order for tensor-product n-cube elements, for which only the one-dimensional
gradient operators are needed. This is done by characterizing interpolating polynomials
on Lobatto quadrature points as a Fourier-Legendre approximation to the quadrature-weighted identity of that point.
The DG gradient operators are then shown to be
approximations to the derivative of the identity. Interpreting the nodal basis in this
way leads to useful and interpretable operations such as integration with the exponential kernel for Fourier analysis, leading to a high-order discrete Fourier transform.
In this work the high-order transform is shown to solve the Poisson equation with
O((∆x)^n+1/2)-accuracy. Efficient calculation of high-dimensional flows is illustrated
by solving the Vlasov-Poisson system in 1D+2V phase space to simulate the instability of a loss-cone distribution to perpendicular-propagating cyclotron harmonic waves.

## Use
You'll need a CUDA-compatible GPU and an install of the CuPy library.

To use the scipts: download the files, make a data folder, and adjust the run parameters (resolutions, domain length, etc.) at the beginning of the "main.py" file.
To play around with different initial conditions, adjust the eigenvalue parameter "om" (for frequency, omega) during initialization of the distribution function.
