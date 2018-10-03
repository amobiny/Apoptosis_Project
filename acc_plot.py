import matplotlib.pyplot as plt
import numpy as np
import h5py

x = np.flip(np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10,
                      9, 8, 7, 6, 5, 4, 3, 2, 1,
                      0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]), 0)

alex_acc = np.array([86, 85.7, 85, 83.7, 84, 83.2, 83.7, 83.2, 82.7, 81.2,
                     80.7, 79.6, 79.7, 78.6, 78, 78.1, 77.5, 77.5, 76.1,
                     75.7, 74, 73.4, 71.6, 70.3, 69.4, 68.5, 67.3, 67])
alex_std = np.array([0, 1, 1.2, 1.1, 1.1, 1.3, 1.4, 1.7, 2, 2.29,
                     2.1, 2.4, 2.3, 2.5, 2.5, 2.3, 2.8, 2.76, 2.78,
                     3.1, 3.2, 3.4, 3.5, 3.7, 4.6, 4.9, 5.1, 5.4])

res_acc = np.array([86.4, 86.1, 84.7, 84.5, 85.05, 84.9, 84.3, 83.5, 82.7, 82.4,
                    82.2, 82, 82.1, 81.3, 80.7, 79.7, 79.1, 79.0, 78.4,
                    78, 76.8, 75.4, 74.7, 73.4, 72.1, 71.7, 70.3, 69.3])
res_std = np.array([0, 1.1, 1.15, 1.17, 1.19, 1.42, 1.26, 1.4, 1.45, 1.42,
                    1.38, 1.38, 1.47, 1.76, 1.9, 2, 2.1, 2.3, 2.5,
                    3, 3.1, 3.34, 3.4, 3.9, 4.63, 4.5, 4.54, 4.7])

dense_acc = np.array([88, 87.9, 87, 86.6, 86.9, 86, 86.5, 86.1, 85.3, 84,
                      83.8, 83.5, 83.1, 82.7, 82.2, 81.6, 81.5, 81, 80.3,
                      79.6, 79.3, 78, 76.8, 75.2, 74.6, 73.3, 73.4, 73.1])
dense_std = np.array([0, 1.4, 1.3, 1.7, 1.65, 1.45, 1.73, 1.85, 1.95, 2.2,
                      2.3, 2.3, 2.4, 2.5, 2.6, 2.9, 2.8, 3, 3.09,
                      3.1, 3.2, 3.4, 3.7, 3.63, 4.1, 4.3, 4.5, 4.5])

cap_acc = np.array([87.2, 87.2, 87.1, 87.1, 87.0, 87.0, 86.8, 86.71, 86.67, 86.65,
                    86.42, 86.1, 85.9, 85.9, 85.5, 85.2, 84.2, 84, 83.8,
                    83.2, 82.6, 81.8, 81.4, 81.6, 81.2, 80., 78.8, 77.2])
cap_std = np.array([0, 1, 1.6, 1.7, 1.4, 1.6, 1.3, 1.4, 1.45, 1.4,
                    1.3, 1.27, 1.43, 1.54, 1.87, 1.99, 1.95, 1.98, 2.51,
                    2.82, 2.77, 2.99, 3.1, 3.8, 3.5, 3.7, 3.86, 3.8])

dcap_acc = np.array([88.5, 88.5, 88.46, 88.48, 88.4, 88.4, 88.3, 88.21, 88.17, 87.1,
                    87, 86.8, 86.5, 85.9, 86.1, 85.4, 84.8, 84.4, 84,
                    83.2, 83.6, 82.6, 82, 81.8, 81.7, 80.3, 79.5, 78.2])
dcap_std = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1.3, 1.27, 1.43, 1.54, 1.57, 1.79, 1.85, 1.98, 2.1,
                    2.2, 2.37, 2.59, 2.91, 3, 3.1, 3.4, 3.56, 3.6])

fig, axes = plt.subplots(nrows=1, ncols=2)
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9, wspace=0.3, hspace=None)
ax0 = axes[1]
line1, = ax0.plot(x, alex_acc, '^', lw=0.8, label='AlexNet', markersize=2, color='orange')
ax0.fill_between(x,
                 alex_acc - alex_std / 2,
                 alex_acc + alex_std / 2,
                 color='orange', alpha=0.3)
line1.set_dashes([1, 2, 1, 2])  # 2pt line, 2pt break, 10pt line, 2pt break

line2, = ax0.plot(x, res_acc, 'v', lw=0.8, label='ResNet', markersize=2, color='firebrick')
ax0.fill_between(x,
                 res_acc - res_std / 2,
                 res_acc + res_std / 2,
                 color='firebrick', alpha=0.3)
line2.set_dashes([1, 1, 2, 1])

line3, = ax0.plot(x, dense_acc, 's', lw=0.8, label='DenseNet-BC', markersize=2, color='cornflowerblue')
ax0.fill_between(x,
                 dense_acc - dense_std / 2,
                 dense_acc + dense_std / 2,
                 color='cornflowerblue', alpha=0.3)
line3.set_dashes([2, 2, 2, 2])

line5, = ax0.plot(x, cap_acc, 'D', lw=0.8, label='CapsNet', markersize=2, color='blueviolet')
ax0.fill_between(x,
                 cap_acc - cap_std / 2,
                 cap_acc + cap_std / 2,
                 color='blueviolet', alpha=0.3)
line5.set_dashes([1, 1, 1, 2])

line4, = ax0.plot(x, dcap_acc, 'o', lw=0.8, label='Deep-CapsNet', markersize=2, color='forestgreen')
ax0.fill_between(x,
                 dcap_acc - dcap_std / 2,
                 dcap_acc + dcap_std / 2,
                 color='forestgreen', alpha=0.3)
line4.set_dashes([0.5, 2, 0.5, 2])

# ax0.plot((0, 7000), (0.1, 50), 'k--', lw=1)
ax0.set_xscale("log", nonposx='clip')
ax0.set_ylim([65, 90])
ax0.set_xlim([0.1, 100])
ax0.set_ylabel('Prediction Accuracy')
ax0.set_xlabel('Percentage fo Samples')
ax0.legend(loc='upper center', ncol=3)

# Plot Precision-Recall curve
ax1 = axes[0]
# ax1.yaxis.tick_right()
# load the  data
h5f = h5py.File('/home/cougarnet.uh.edu/amobiny/Desktop/Apoptosis_Project/alexnet_results.h5', 'r')
Precision_alex = h5f['Precision'][:]
Recall_alex = h5f['Recall'][:]
h5f.close()

h5f = h5py.File('/home/cougarnet.uh.edu/amobiny/Desktop/Apoptosis_Project/resnet_results.h5', 'r')
Precision_res = h5f['Precision'][:]
Recall_res = h5f['Recall'][:]
h5f.close()

h5f = h5py.File('/home/cougarnet.uh.edu/amobiny/Desktop/Apoptosis_Project/densenet_269999.h5', 'r')
Precision_dense = h5f['Precision'][:]
Recall_dense = h5f['Recall'][:]
h5f.close()

h5f = h5py.File('/home/cougarnet.uh.edu/amobiny/Desktop/Apoptosis_Project/capsnet_results.h5', 'r')
Precision_caps = h5f['Precision'][:]
Recall_caps = h5f['Recall'][:]
h5f.close()

h5f = h5py.File('/home/cougarnet.uh.edu/amobiny/Desktop/Apoptosis_Project/densenet_620999.h5', 'r')
Precision_dcaps = h5f['Precision'][:]
Recall_dcaps = h5f['Recall'][:]
h5f.close()


ax1.plot(Recall_alex, Precision_alex, lw=1, color='orange', label='AlexNet')
ax1.plot(Recall_res, Precision_res, lw=1, color='firebrick', label='ResNet')
ax1.plot(Recall_dense, Precision_dense, lw=1, color='cornflowerblue', label='DenseNet-BC')
ax1.plot(Recall_caps, Precision_caps, lw=1, color='blueviolet', label='CapsNet')
ax1.plot(Recall_dcaps+0.003, Precision_dcaps+0.003, lw=1, color='forestgreen', label='Deep-CapsNet')


ax1.plot(0.8449, 0.8693, 'o', color='orange', markersize=3)
ax1.plot(0.8439, 0.8812, 'o', color='firebrick', markersize=3)
ax1.plot(0.8115, 0.9393, 'o', color='cornflowerblue', markersize=3)
ax1.plot(0.8530, 0.8859, 'o', color='blueviolet', markersize=3)
ax1.plot(0.839, 0.9298, 'o', color='forestgreen', markersize=3)


ax1.set_xlabel('Recall', size=10)
ax1.set_ylabel('Precision', size=10)
# ax1.tick_params(labelsize=18)
ax1.set_ylim([0.5, 1.])
ax1.set_xlim([0.5, 1.])
# plt.legend(loc=3, borderpad=1.5)

width = 7
height = width / 2
fig.set_size_inches(width, height)
fig.savefig('plot.pdf')
fig.savefig('plot.svg')
