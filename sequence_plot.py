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

x[28] = x[52]
y_true[28] = y_true[52]
y_pred_cnn_alone[28] = y_pred_cnn_alone[52]
y_pred_cnn[28] = y_pred_cnn[52]
y_pred_cap[28] = y_pred_cap[52]



# plot_idx = [2, 5, 7, 9, 12, 14, 17, 18, 26, 36]
# seq_num = 3

# plot_idx = [1, 2, 3, 4, 6, 12, 17, 21, 23, 40]
# # plot_idx = list(range(40, 60))
# seq_num = 28
# j = 0
# # plot all 72
# fig, axes = plt.subplots(nrows=1, ncols=10)
# for i, val in enumerate(x[seq_num]):
#     if i in plot_idx:
#         ax = axes[j]
#         ax.imshow(val.reshape(28, 28), cmap='gray')
#         ax.set_title('{}:{},{},{},{}'.format(i, int(y_true[seq_num, i]), int(y_pred_cnn_alone[seq_num, i]),
#                                              int(y_pred_cnn[seq_num, i]), int(y_pred_cap[seq_num, i])))
#         j += 1
# # plt.show()
# width = 20
# height = 5
# fig.set_size_inches(width, height)
# fig.savefig('C2.pdf')
# fig.savefig('C2.svg')
# print()


def forceAspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


# index = random.sample(range(2369), 36)
index = list(range(30))

fig, axs = plt.subplots(1, 4)
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9, wspace=0.3, hspace=None)

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

# plt.show()
width = 7
height = width / 2
fig.set_size_inches(width, height)
fig.savefig('fig6.pdf')
fig.savefig('fig6.svg')




