import numpy as np
import os
import h5py
from py_utils import *

data_dir = '/home/cougarnet.uh.edu/amobiny/Desktop/Apoptosis_Project/data/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    os.makedirs(data_dir+'2_class')
    os.makedirs(data_dir+'4_class')


x_train_effector = np.load('/home/cougarnet.uh.edu/amobiny/Desktop/Apoptosis_Project/raw_data/Effector/X_Train.npy')
y_train_effector = np.load('/home/cougarnet.uh.edu/amobiny/Desktop/Apoptosis_Project/raw_data/Effector/Y_Train.npy')
x_test_effector = np.load('/home/cougarnet.uh.edu/amobiny/Desktop/Apoptosis_Project/raw_data/Effector/X_Test.npy')
y_test_effector = np.load('/home/cougarnet.uh.edu/amobiny/Desktop/Apoptosis_Project/raw_data/Effector/Y_Test.npy')

# these labels are already one-hot-encoded
x_train_target = np.load('/home/cougarnet.uh.edu/amobiny/Desktop/Apoptosis_Project/raw_data/Target/X_train.npy')[:72000]
y_train_target = np.load('/home/cougarnet.uh.edu/amobiny/Desktop/Apoptosis_Project/raw_data/Target/y_train.npy')[:72000]
x_test_target = np.load('/home/cougarnet.uh.edu/amobiny/Desktop/Apoptosis_Project/raw_data/Target/X_test.npy')
y_test_target = np.load('/home/cougarnet.uh.edu/amobiny/Desktop/Apoptosis_Project/raw_data/Target/y_test.npy')

# normalize
x_train_effector_51 = x_train_effector / 255.
x_test_effector_51 = x_test_effector / 255.
x_train_target_51 = x_train_target / 255.
x_test_target_51 = x_test_target / 255.

y_train_effector = one_hot(y_train_effector, num_class=2)
y_test_effector = one_hot(y_test_effector, num_class=2)

x_valid_target_51, y_valid_target, x_test_target_51, y_test_target = split_valid_test(x_test_target_51,
                                                                                      y_test_target,
                                                                                      num_valid_per_class=5000)
x_valid_effector_51, y_valid_effector, x_test_effector_51, y_test_effector = split_valid_test(x_test_effector_51,
                                                                                              y_test_effector,
                                                                                              num_valid_per_class=4000)

# Target
x_train_target_28 = resize(x_train_target_51, new_size=28)
x_valid_target_28 = resize(x_valid_target_51, new_size=28)
x_test_target_28 = resize(x_test_target_51, new_size=28)

# Effector
x_train_effector_28 = resize(x_train_effector_51, new_size=28)
x_valid_effector_28 = resize(x_valid_effector_51, new_size=28)
x_test_effector_28 = resize(x_test_effector_51, new_size=28)

h5f = h5py.File(data_dir + '2_class/target_51.h5', 'w')
h5f.create_dataset('X_train', data=x_train_target_51)
h5f.create_dataset('Y_train', data=y_train_target)
h5f.create_dataset('X_valid', data=x_valid_target_51)
h5f.create_dataset('Y_valid', data=y_valid_target)
h5f.create_dataset('X_test', data=x_test_target_51)
h5f.create_dataset('Y_test', data=y_test_target)
h5f.close()

h5f = h5py.File(data_dir + '2_class/effector_51.h5', 'w')
h5f.create_dataset('X_train', data=x_train_effector_51)
h5f.create_dataset('Y_train', data=y_train_effector)
h5f.create_dataset('X_valid', data=x_valid_effector_51)
h5f.create_dataset('Y_valid', data=y_valid_effector)
h5f.create_dataset('X_test', data=x_test_effector_51)
h5f.create_dataset('Y_test', data=y_test_effector)
h5f.close()

h5f = h5py.File(data_dir + '2_class/target_28.h5', 'w')
h5f.create_dataset('X_train', data=x_train_target_28)
h5f.create_dataset('Y_train', data=y_train_target)
h5f.create_dataset('X_valid', data=x_valid_target_28)
h5f.create_dataset('Y_valid', data=y_valid_target)
h5f.create_dataset('X_test', data=x_test_target_28)
h5f.create_dataset('Y_test', data=y_test_target)
h5f.close()

h5f = h5py.File(data_dir + '2_class/effector_28.h5', 'w')
h5f.create_dataset('X_train', data=x_train_effector_28)
h5f.create_dataset('Y_train', data=y_train_effector)
h5f.create_dataset('X_valid', data=x_valid_effector_28)
h5f.create_dataset('Y_valid', data=y_valid_effector)
h5f.create_dataset('X_test', data=x_test_effector_28)
h5f.create_dataset('Y_test', data=y_test_effector)
h5f.close()

# Adding 12K samples to the effector training samples to make it balanced
x_train_effector_51, x_train_effector_28, y_train_effector = minor_augmentation(x_train_effector_51,
                                                                                x_train_effector_28,
                                                                                y_train_effector,
                                                                                num_per_class=6000)

x_train_51, y_train_51 = merge_all_classes(x_train_effector_51, y_train_effector, x_train_target_51, y_train_target)
x_valid_51, y_valid_51 = merge_all_classes(x_valid_effector_51, y_valid_effector, x_valid_target_51, y_valid_target)
x_test_51, y_test_51 = merge_all_classes(x_test_effector_51, y_test_effector, x_test_target_51, y_test_target)

x_train_28, y_train_28 = merge_all_classes(x_train_effector_28, y_train_effector, x_train_target_28, y_train_target)
x_valid_28, y_valid_28 = merge_all_classes(x_valid_effector_28, y_valid_effector, x_valid_target_28, y_valid_target)
x_test_28, y_test_28 = merge_all_classes(x_test_effector_28, y_test_effector, x_test_target_28, y_test_target)

h5f = h5py.File(data_dir + '4_class/data_51.h5', 'w')
h5f.create_dataset('X_train', data=x_train_51)
h5f.create_dataset('Y_train', data=y_train_51)
h5f.create_dataset('X_valid', data=x_valid_51)
h5f.create_dataset('Y_valid', data=y_valid_51)
h5f.create_dataset('X_test', data=x_test_51)
h5f.create_dataset('Y_test', data=y_test_51)
h5f.close()

h5f = h5py.File(data_dir + '4_class/data_28.h5', 'w')
h5f.create_dataset('X_train', data=x_train_28)
h5f.create_dataset('Y_train', data=y_train_28)
h5f.create_dataset('X_valid', data=x_valid_28)
h5f.create_dataset('Y_valid', data=y_valid_28)
h5f.create_dataset('X_test', data=x_test_28)
h5f.create_dataset('Y_test', data=y_test_28)
h5f.close()
