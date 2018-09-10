import h5py
import numpy as np
from py_utils import minor_augmentation

data_dir = '/home/cougarnet.uh.edu/amobiny/Desktop/Apoptosis_Project/data/'

h5f = h5py.File(data_dir+'2_class/effector_28.h5', 'r')
x_train_effector = h5f['X_train'][:]
y_train_effector = h5f['Y_train'][:]
x_valid_effector = h5f['X_valid'][:]
y_valid_effector = h5f['Y_valid'][:]
x_test_effector = h5f['X_test'][:]
y_test_effector = h5f['Y_test'][:]
h5f.close()

h5f = h5py.File(data_dir+'2_class/target_28.h5', 'r')
x_train_target = h5f['X_train'][:]
y_train_target = h5f['Y_train'][:]
x_valid_target = h5f['X_valid'][:]
y_valid_target = h5f['Y_valid'][:]
x_test_target = h5f['X_test'][:]
y_test_target = h5f['Y_test'][:]
h5f.close()

# Adding 12K samples to the effector training samples to make it balanced
x_train_effector, _, y_train_effector = minor_augmentation(x_train_effector,
                                                           x_train_effector,
                                                           y_train_effector,
                                                           num_per_class=6000)


def add_channel(x, value=1):
    if value == 1:
        extra = np.ones_like(x)
    elif value == 0:
        extra = np.zeros_like(x)
    x_new = np.concatenate((x, extra), axis=-1)
    return x_new


x_train_effector = add_channel(x_train_effector, value=0)
x_valid_effector = add_channel(x_valid_effector, value=0)
x_test_effector = add_channel(x_test_effector, value=0)

x_train_target = add_channel(x_train_target, value=1)
x_valid_target = add_channel(x_valid_target, value=1)
x_test_target = add_channel(x_test_target, value=1)

x_train = np.concatenate((x_train_target, x_train_effector), axis=0)
y_train = np.concatenate((y_train_target, y_train_effector), axis=0)
x_valid = np.concatenate((x_valid_target, x_valid_effector), axis=0)
y_valid = np.concatenate((y_valid_target, y_valid_effector), axis=0)
x_test = np.concatenate((x_test_target, x_test_effector), axis=0)
y_test = np.concatenate((y_test_target, y_test_effector), axis=0)

h5f = h5py.File(data_dir + '2_class/revised_data_28.h5', 'w')
h5f.create_dataset('X_train', data=x_train)
h5f.create_dataset('Y_train', data=y_train)
h5f.create_dataset('X_valid', data=x_valid)
h5f.create_dataset('Y_valid', data=y_valid)
h5f.create_dataset('X_test', data=x_test)
h5f.create_dataset('Y_test', data=y_test)
h5f.close()