# [Vlasov1D2V](https://github.com/crewsdw/Vlasov1D2V/)

This collection of Python codes solves the Vlasov-Poisson system in 1D+2V with up to 9th order spatial accuracy and 3rd order temporal accuracy using CUDA-accelerated libraries, namely [CuPy](https://github.com/cupy/cupy).

## Use
To use the scipts: download the files, make a data folder, and adjust the run parameters (resolutions, domain length, etc.) at the beginning of the "main.py" file.
To play around with different initial conditions, adjust the eigenvalue parameter "om" (for frequency, omega) during initialization of the distribution function.
