import h5py
import numpy as np


class RunData:
    def __init__(self, folder, filename, shape, geometry, time, refs):
        self.write_filename = folder + filename + '.hdf5'
        self.info_name = folder + filename + '_info.txt'
        self.shape = shape
        # Recreate run-file
        refs.info_file(self.info_name, geometry, time)

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
            f.create_dataset('energy', data=[0], chunks=True, maxshape=(None,))

    def save_data(self, distribution, elliptic, density, time, field_energy):
        # Open for appending
        with h5py.File(self.write_filename, 'a') as f:
            # Add new time line
            f['pdf'].resize((f['pdf'].shape[0] + 1), axis=0)
            f['potential'].resize((f['potential'].shape[0] + 1), axis=0)
            f['density'].resize((f['density'].shape[0] + 1), axis=0)
            f['time'].resize((f['time'].shape[0] + 1), axis=0)
            f['energy'].resize((f['energy'].shape[0] + 1), axis=0)
            # Save data
            f['pdf'][-1] = distribution
            f['potential'][-1] = elliptic.potential.get()
            f['density'][-1] = density.get()
            f['time'][-1] = time
            f['energy'][-1] = field_energy


class ReadData:
    def __init__(self, folder, filename):
        self.write_filename = folder + filename + '.hdf5'
        self.info_name = folder + filename + '_info.txt'

    def read_data(self):
        # Open for reading
        with h5py.File(self.write_filename, 'r') as f:
            t = f['time'][()]
            pdf = f['pdf'][()]
            pot = f['potential'][()]
            den = f['density'][()]
            eng = f['energy'][()]
        return t, pdf, pot, den, eng

    def read_specific_time(self, idx):
        # Open for reading
        with h5py.File(self.write_filename, 'r') as f:
            t = f['time'][idx]
            pdf = f['pdf'][idx]
        return t, pdf

    # Read data file
    def read_info(self):
        # Open info file
        params = open(self.info_name, 'r')
        # Read lines
        p = params.readlines()

        ### Geometry
        # Spatial
        order_x = int(float(p[46].split()[3]))
        res_x = int(float(p[34].split()[2]))
        x_low = float(p[37].split()[3])
        x_high = float(p[38].split()[3])
        # Velocity x
        order_u = int(float(p[47].split()[3]))
        res_u = int(float(p[35].split()[2]))
        u_low = float(p[39].split()[3])
        u_high = float(p[40].split()[3])
        # Velocity y
        order_v = int(float(p[48].split()[3]))
        res_v = int(float(p[36].split()[2]))
        v_low = int(float(p[41].split()[3]))
        v_high = int(float(p[42].split()[3]))
        # Final time
        final_time = float(p[43].split()[3])
        write_time = float(p[44].split()[4])
        order_time = float(p[49].split()[3])

        ### Reference values
        n0 = float(p[6].split()[2])
        T0 = float(p[7].split()[2])
        L0 = float(p[8].split()[3])

        # pack
        # geometry_info = np.array([[res_x, x_low, x_high, order_x],
        #                           [res_u, u_low, u_high, order_u],
        #                           [res_v, v_low, v_high, order_v]])
        orders = np.array([order_x, order_u, order_v])
        resolutions = np.array([res_x, res_u, res_v])
        lows = np.array([x_low, u_low, v_low])
        highs = np.array([x_high, u_high, v_high])
        time_info = np.array([final_time, write_time, order_time])
        refs = np.array([n0, T0, L0])

        return orders, resolutions, lows, highs, time_info, refs
