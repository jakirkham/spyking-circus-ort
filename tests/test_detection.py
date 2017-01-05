import matplotlib.pyplot as plt
import numpy as np

import circusort



path = "/data/tmp/synthetic.raw"

# Generate data
desc = circusort.io.generate.synthetic_grid(path)

# Load dataset
dataset = circusort.io.load.raw_binary(path, desc.nb_channels, desc.length, desc.sampling_rate)


class RingBuffer():
    '''Simple ring buffer'''
    def __init__(self, shape):
        self.shape = shape
        self.buffer_size = self.shape[0]
        self.data = np.zeros(self.shape)
        self.size = 0

    @property
    def start_index(self):
        return self.size - self.buffer_size

    @property
    def end_index(self):
        return self.size - 1

    def append(self, chunk_data):
        chunk_size = chunk_data.shape[0]
        self.data[:-chunk_size] = self.data[chunk_size:]
        self.data[-chunk_size:] = chunk_data
        self.size += chunk_size
        return

def detect_peaks(x, thresh=None, left=None, right=None, negative=True, edge='rising'):
    ''' Detect peaks
    x: array like
        Data.
    thresh: None, float (default None)
        Detect peaks that are greater than this threshold.
    ...
    '''
    if 1 == x.ndim:
        ind_dtype = 'uint32'
        zero = np.array([0], dtype=ind_dtype)
        # Reverse data
        if negative:
            x = - x
        # First find all the peak indices
        dx = x[1:] - x[:-1]
        ## Preallocation
        ine, ire, ife = np.array([[], [], []], dtype=ind_dtype)
        if not edge:
            flags_1 = np.hstack((dx, zero)) < 0
            flags_2 = np.hstack((zero, dx)) > 0
            flags = np.logical_and(flags_1, flags_2)
            ine = np.where(flags)[0]
        else:
            if edge.lower() in ['rising', 'both']:
                flags_1 = np.hstack((dx, zero)) <= 0
                flags_2 = np.hstack((zero, dx)) > 0
                flags = np.logical_and(flags_1, flags_2)
                ire = np.where(flags)[0]
            if edge.lower() in ['falling', 'both']:
                flags_1 = np.hstack((dx, zero)) < 0
                flags_2 = np.hstack((zero, dx)) >= 0
                flags = np.logical_and(flags_1, flags_2)
                ife = np.where(flags)[0]
        ind = np.hstack((ine, ire, ife))
        ind = np.unique(ind)
        # Sort peak indices
        ind = np.sort(ind)
        # Remove first index which cannot be a peak index
        if 0 < ind.size and ind[0] == 0:
            ind = ind[1:]
        # Remove last index which cannot be a peak index
        if 0 < ind.size and ind[-1] == x.size - 1:
            ind = ind[:-1]
        # Remove small peaks
        if 0 < ind.size and thresh is not None:
            idel = x[ind] < thresh
            ind = ind[~idel]
        # Remove peaks near higher peak
        if 0 < ind.size and (0 < left or 0 < right):
            ind = ind[np.argsort(x[ind])] # sort indices by peak height
            ind = ind[::-1] # flip array (decreasing peak height)
            idel = np.zeros(ind.size, dtype='bool')
            # For each potential peak index
            for i in range(0, ind.size):
                if not idel[i]:
                    # idel = idel | (ind[i] - 40 <= ind) & (ind <= ind[i] + 40)
                    idel = idel | (ind[i] - left <= ind) & (ind <= ind[i] + right)
                    idel[i] = False # keep peak index
                    # TODO remove...
                    # if i < 2:
                    #     idel[i] = False
                    # else:
                    #     idel[i] = True
                else:
                    pass
            ind = ind[~idel]
        # Remove left indices which may not be peak indices
        if 0 < ind.size and 0 < left:
            ind = ind[left <= ind]
        # Remove right indices which may not be peak indices
        if 0 < ind.size and 0 < right:
            ind = ind[ind < x.size - right]
        # Sort peak indices
        ind = np.sort(ind)
        # Change indices reference
        ind = ind - left
        return ind
    elif 2 == x.ndim:
        size = x.shape[1]
        inds = [None] * size
        for k in range(0, size):
            inds[k] = detect_peaks(x[:, k], thresh=thresh, left=left, right=right, negative=negative, edge=edge)
        return inds
    else:
        assert(False)


left_padding = 20
right_padding = 40

w_start = left_padding
w_end = dataset.buffer_size + left_padding - right_padding
w_size = w_end - w_start

rb_size = dataset.buffer_size + left_padding
shape = (rb_size, desc.nb_channels)
rb = RingBuffer(shape)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
line1, = ax.plot([0.0] * rb_size)
_ = ax.axvline(x=w_start, color='k', linestyle='dashed')
_ = ax.axvline(x=w_end, color='k', linestyle='dashed')
line2, = ax.plot(range(w_start, w_end), [0.0] * w_size)
line3, = ax.plot(range(w_start, w_end), [0.0] * w_size)
line4, = ax.plot(range(w_start, w_end), [0.0] * w_size)
ax.set_xlim(left=0, right=rb_size-1)
ax.set_ylim(bottom=-0.01, top=0.01)


# Load data online
count = 0
mu = np.zeros(desc.nb_channels)
mad = np.zeros(desc.nb_channels)
while dataset.data_available:
    rb.append(dataset.load_data())
    buf = rb.data[left_padding:dataset.buffer_size+left_padding]
    line1.set_ydata(rb.data[:, 0])
    mu = float(count) / float(count + 1) * mu + 1.0 / float(count + 1) * np.median(buf, axis=0)
    mad = float(count) / float(count + 1) * mad + 1.0 / float(count + 1) * 1.4826 * np.median(np.abs(buf - mu), axis=0)
    count += 1
    line2.set_ydata([mu[0]] * w_size)
    line3.set_ydata([mu[0] - 6.0 * mad[0]] * w_size)
    # Center data
    data = rb.data - mu
    # Reduce data
    data = data / mad # TODO check if divide by zero...
    i = detect_peaks(data, thresh=6.0, left=left_padding, right=right_padding)
    print(i[0])
    mask = np.zeros(w_size)
    mask[i[0]] = -5.0e-3
    line4.set_ydata(mask)
    fig.canvas.draw()
    plt.pause(0.3)
    # TODO: detect threshold crossing...

while True:
    plt.pause(0.05)
