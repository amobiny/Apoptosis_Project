import numpy as np
from keras.utils import to_categorical
import os
import h5py

data_dir = '/home/cougarnet.uh.edu/amobiny/Desktop/Apoptosis_Project/data/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


print("Loading Training Data ...")
X_train = np.load('/home/cougarnet.uh.edu/amobiny/Desktop/TIMING-Apoptosis-Dataset/CNN-LSTM/Target/Tx28_Combine_train.npy')
y_train = np.load('/home/cougarnet.uh.edu/amobiny/Desktop/TIMING-Apoptosis-Dataset/CNN-LSTM/Target/Ty28_Combine_train.npy')
print("Training Data loaded!")

print("Loading Test Data ...")
X_test = np.load('/home/cougarnet.uh.edu/amobiny/Desktop/TIMING-Apoptosis-Dataset/CNN-LSTM/Target/Tx28_Combine_test.npy')
y_test = np.load('/home/cougarnet.uh.edu/amobiny/Desktop/TIMING-Apoptosis-Dataset/CNN-LSTM/Target/Ty28_Combine_test.npy')
print("Test Data loaded!")

y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

print("Saving Sequential Data ...")

h5f = h5py.File(data_dir + 'sequential_data_28.h5', 'w')
h5f.create_dataset('X_train', data=X_train)
h5f.create_dataset('Y_train', data=y_train)
h5f.create_dataset('X_valid', data=X_test)
h5f.create_dataset('Y_valid', data=y_test)
h5f.create_dataset('X_test', data=X_test)
h5f.create_dataset('Y_test', data=y_test)
h5f.close()





