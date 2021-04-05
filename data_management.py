import h5py
import numpy as np


# import datetime


class RunData:
    def __init__(self, folder, filename, shape):
        self.write_filename = folder + filename + '.hdf5'
        self.shape = shape

    def create_file(self, distribution, elliptic, density):
        # Open file for writing
        with h5py.File(self.write_filename, 'w') as f:
            # Create datasets, dataset_distribution =
            f.create_dataset('pdf', data=np.array([distribution]),
                             chunks=True,
                             maxshape=(None, self.shape[0], self.shape[1], self.shape[2],
                                       self.shape[3], self.shape[4], self.shape[5]),
                             dtype='f')
            # space types, dataset_potential = , dataset_density =
            f.create_dataset('potential', data=np.array([elliptic.potential.get()]),
                             chunks=True,
                             maxshape=(None, self.shape[0], self.shape[1]),
                             dtype='f')
            f.create_dataset('density', data=np.array([density.get()]),
                             chunks=True,
                             maxshape=(None, self.shape[0], self.shape[1]),
                             dtype='f')
            # time, dataset_time =
            f.create_dataset('time', data=[0], chunks=True, maxshape=(None,))

    def save_data(self, distribution, elliptic, density, time):
        # Open for appending
        with h5py.File(self.write_filename, 'a') as f:
            # Add new time line
            f['pdf'].resize((f['pdf'].shape[0] + 1), axis=0)
            f['potential'].resize((f['potential'].shape[0] + 1), axis=0)
            f['density'].resize((f['density'].shape[0] + 1), axis=0)
            f['time'].resize((f['time'].shape[0] + 1), axis=0)
            # Save data
            f['pdf'][-1] = distribution
            f['potential'][-1] = elliptic.potential.get()
            f['density'][-1] = density.get()
            f['time'][-1] = time
