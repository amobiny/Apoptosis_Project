import h5py
import numpy as np

h5f = h5py.File('/home/cougarnet.uh.edu/amobiny/Desktop/Apoptosis_Project/data/data_28.h5', 'r')
X_train = h5f['X_train'][:]
Y_train = h5f['Y_train'][:]
X_test = h5f['X_test'][:]
Y_test = h5f['Y_test'][:]
h5f.close()

percent = 1
total_num_data = Y_train.shape[0]
for _ in range(9):
    current_num_data = Y_train.shape[0]
    percent -= 0.1
    n = int(percent * total_num_data)
    index = np.random.choice(current_num_data, n, replace=False)
    X_train = X_train[index]
    Y_train = Y_train[index]
    print('data_{} is created using {} images out of {}'.format(percent, n, current_num_data))

    h5f = h5py.File('/home/cougarnet.uh.edu/amobiny/Desktop/Apoptosis_Project/data/data_28_' + str(percent) + '.h5', 'w')
    h5f.create_dataset('X_train', data=X_train)
    h5f.create_dataset('Y_train', data=Y_train)
    h5f.create_dataset('X_test', data=X_test)
    h5f.create_dataset('Y_test', data=Y_test)
    h5f.close()

