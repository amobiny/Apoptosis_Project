import numpy as np
from keras.utils import to_categorical

#Step-0 Load Training Data
print("Loading Training Data ...")
X = np.load('/home/cougarnet.uh.edu/amobiny/Desktop/TIMING-Apoptosis-Dataset/CNN-LSTM/Target/Tx28_Combine_train.npy')
y = np.load('/home/cougarnet.uh.edu/amobiny/Desktop/TIMING-Apoptosis-Dataset/CNN-LSTM/Target/Ty28_Combine_train.npy')
y = to_categorical(y, 2)
y = y.reshape(X.shape[0], 72, 2)
print("Training Data loaded!")


