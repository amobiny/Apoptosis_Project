import h5py
import numpy as np
import matplotlib.pyplot as plt
import random

h5f = h5py.File('bilstm_alexnet.h5', 'r')
x = h5f['x'][:]
y_pred_cnn = h5f['y_pred'][:]
y_true = h5f['y_true'][:]
h5f.close()

h5f = h5py.File('bilstm_capsnet.h5', 'r')
y_pred_cap = h5f['y_pred'][:]
h5f.close()

h5f = h5py.File('cnn.h5', 'r')
y_pred_cnn_alone = h5f['y_pred'][:]
h5f.close()


def forceAspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


# index = random.sample(range(2369), 36)
index = list(range(30))

fig, axs = plt.subplots(1, 4)

ax = axs[0]
im = ax.imshow(y_true[index, :], cmap=plt.cm.bwr)
forceAspect(ax, 2)

ax = axs[1]
im = ax.imshow(y_pred_cnn_alone[index, :], cmap=plt.cm.bwr)
forceAspect(ax, 2)

ax = axs[2]
im = ax.imshow(y_pred_cnn[index, :], cmap=plt.cm.bwr)
forceAspect(ax, 2)

ax = axs[3]
im = ax.imshow(y_pred_cap[index, :], cmap=plt.cm.bwr)
forceAspect(ax, 2)


width = 7
height = width / 2
fig.set_size_inches(width, height)
fig.savefig('fig6.pdf')
fig.savefig('fig6.svg')
