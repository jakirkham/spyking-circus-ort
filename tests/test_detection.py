import matplotlib.pyplot as plt
import numpy as np

import circusort



path = "/data/tmp/synthetic.raw"

# Generate data
desc = circusort.io.generate.synthetic_grid(path)

# Load dataset
dataset = circusort.io.load.raw_binary(path, desc.nb_channels, desc.length, desc.sampling_rate)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
line1, = ax.plot([0] * dataset.buffer_size)
line2, = ax.plot([0] * dataset.buffer_size)
line3, = ax.plot([0] * dataset.buffer_size)
line4, = ax.plot([0] * dataset.buffer_size)
ax.set_xlim(left=0, right=1024-1)
ax.set_ylim(bottom=-0.01, top=0.01)


# Load data online
count = 0
mu = np.zeros(desc.nb_channels)
mad = np.zeros(desc.nb_channels)
while dataset.data_available:
    buf = dataset.load_data()
    line1.set_ydata(buf[:, 0])
    mu = float(count) / float(count + 1) * mu + 1.0 / float(count + 1) * np.median(buf, axis=0)
    mad = float(count) / float(count + 1) * mad + 1.0 / float(count + 1) * 1.4826 * np.median(np.abs(buf - mu), axis=0)
    count = count + 1
    line2.set_ydata([mu[0]] * dataset.buffer_size)
    line2.set_ydata([mu[0] - 6.0 * mad[0]] * dataset.buffer_size)
    # Remove median
    buf = buf - mu
    mask = buf < -6.0 * mad
    line4.set_ydata(-5.0e-3 * mask[:, 0])
    print(buf)
    print(mad)
    print(mask)
    fig.canvas.draw()
    plt.pause(0.01)
    # TODO: detect threshold crossing...

while True:
    plt.pause(0.05)
