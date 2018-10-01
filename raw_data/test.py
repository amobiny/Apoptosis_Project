import h5py
import numpy as np

h5f = h5py.File('/home/cougarnet.uh.edu/amobiny/Desktop/Apoptosis_Project/data/data_28_0.01.h5', 'r')
X_train = h5f['X_train'][:]
Y_train = h5f['Y_train'][:]
X_test = h5f['X_test'][:]
Y_test = h5f['Y_test'][:]
h5f.close()

num = [648, 576, 504, 432, 360, 288, 216, 144, 72]
goh = [0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]

for i in range(len(num)):
    total_num_data = Y_train.shape[0]
    n = num[i]
    index = np.random.choice(total_num_data, n, replace=False)
    X_train = X_train[index]
    Y_train = Y_train[index]
    h5f = h5py.File('/home/cougarnet.uh.edu/amobiny/Desktop/Apoptosis_Project/data/data_28_' + str(goh[i]) + '.h5', 'w')
    h5f.create_dataset('X_train', data=X_train)
    h5f.create_dataset('Y_train', data=Y_train)
    h5f.create_dataset('X_test', data=X_test)
    h5f.create_dataset('Y_test', data=Y_test)
    h5f.close()



