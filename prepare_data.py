import numpy as np
import os
import h5py

data_dir = './data/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

X_train_51 = np.load('/home/cougarnet.uh.edu/amobiny/Desktop/DATASETS/apoptosis/CNN/TRAIN/X_train_Alex.npy')
X_train_28 = np.load('/home/cougarnet.uh.edu/amobiny/Desktop/DATASETS/apoptosis/CNN/TRAIN/X_train_LeNet.npy')
y_train = np.load('/home/cougarnet.uh.edu/amobiny/Desktop/DATASETS/apoptosis/CNN/TRAIN/y_train.npy')

X_test_51 = np.load('/home/cougarnet.uh.edu/amobiny/Desktop/DATASETS/apoptosis/CNN/TEST/X_test_Alex.npy')
X_test_28 = np.load('/home/cougarnet.uh.edu/amobiny/Desktop/DATASETS/apoptosis/CNN/TEST/X_test_LeNet.npy')
y_test = np.load('/home/cougarnet.uh.edu/amobiny/Desktop/DATASETS/apoptosis/CNN/TEST/y_test.npy')

h5f = h5py.File(data_dir+'data_51.h5', 'w')
h5f.create_dataset('X_train', data=X_train_51[:72000])
h5f.create_dataset('Y_train', data=y_train[:72000])
h5f.create_dataset('X_test', data=X_test_51)
h5f.create_dataset('Y_test', data=y_test)
h5f.close()

h5f = h5py.File(data_dir+'data_28.h5', 'w')
h5f.create_dataset('X_train', data=X_train_28[:72000])
h5f.create_dataset('Y_train', data=y_train[:72000])
h5f.create_dataset('X_test', data=X_test_28)
h5f.create_dataset('Y_test', data=y_test)
h5f.close()


