import numpy as np
import os.path



class RawBinary(object):
    '''TODO add doc...'''

    def __init__(self, path, nb_channels, length, sampling_rate):
        self.path = path
        self.nb_channels = nb_channels
        self.length = length
        self.shape = (self.length, self.nb_channels)
        self.sampling_rate = sampling_rate
        self.data_available = (0 < self.length)
        self.f = np.memmap(self.path, dtype='float32', mode='r', shape=self.shape)
        self.buffer_size = 1024
        self.head = 0

    def load(self):
        data = self.f[:, :]
        return data

    def load_data(self):
        i_start = self.head
        i_end = self.head + self.buffer_size
        data = self.f[i_start:i_end, :]
        if i_end < self.length:
            self.head = i_end
        else:
            self.data_available = False
        return data


def raw_binary(path, nb_channels, length, sampling_rate):
    path = os.path.expanduser(path)
    raw_binary = RawBinary(path, nb_channels, length, sampling_rate)
    return raw_binary
    # data = raw.load()
    # return data
