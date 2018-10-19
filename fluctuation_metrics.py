import h5py
import numpy as np
import matplotlib.pyplot as plt

h5f = h5py.File('bilstm_alexnet.h5', 'r')
x = h5f['x'][:]
y_pred_cnn_lstm = h5f['y_pred'][:]
y_true = h5f['y_true'][:]
h5f.close()

h5f = h5py.File('bilstm_capsnet.h5', 'r')
y_pred_cap_lstm = h5f['y_pred'][:]
h5f.close()

h5f = h5py.File('cnn.h5', 'r')
y_pred_cnn = h5f['y_pred'][:]
h5f.close()


def get_seq_jump(x):
    length = x.shape[0]
    count = 0
    for i in range(length - 1):
        if x[i] != x[i + 1]:
            count += 1
    return count


def calc_mud(array):
    """
    calculate MUD metric
    :param array: array of shape: [#sequences, sequence_length]
    :return: scalar jumps
    """
    jumps = []
    for seq in array:
        jumps.append(get_seq_jump(seq))
    return np.array(jumps)


jumps_gt = calc_mud(y_true)
jumps_cnn = calc_mud(y_pred_cnn)
jumps_cnn_lstm = calc_mud(y_pred_cnn_lstm)
jumps_cap_lstm = calc_mud(y_pred_cap_lstm)

print(np.mean(np.abs(jumps_gt-jumps_cnn)))
print(np.mean(np.abs(jumps_gt-jumps_cnn_lstm)))
print(np.mean(np.abs(jumps_gt-jumps_cap_lstm)))


# should be computed only over 2127 test sequences which have either 0 or 1 jump
def get_death_distance(y_gt, y_pred):
    a = 0
    for i in range(y_gt.shape[0]):
        if get_seq_jump(y_gt[i]) == 1 and np.sum(y_pred[i]):
            a += np.abs(np.where(y_gt[i] == 1)[0][0] - np.where(y_pred[i] == 1)[0][0])
    print(a/2127.)


get_death_distance(y_true, y_pred_cnn)
get_death_distance(y_true, y_pred_cnn_lstm)
get_death_distance(y_true, y_pred_cap_lstm)

print()
