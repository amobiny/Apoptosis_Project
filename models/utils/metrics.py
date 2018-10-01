import numpy as np
from sklearn.metrics import confusion_matrix


def get_sequence_label(y):
    """
    maps many sequence of labels to single labels based on the pre-defined criterions
    :param y: sequences of labels of shape [#sequences, sequence_length]
    :return: sequence labels of shape [#sequences]
    """
    sequence_length = y.shape[1]
    # criterion#1: if at least one-fifth of the sequence has value 1 --> label=1
    y_out = (np.sum(y, axis=1) >= sequence_length / 5.).astype(int)
    # criterion#2: if the sequence has at least five consecutive 1s --> label=1
    counts = np.zeros((y.shape[0]))
    for i in range(y.shape[0]):
        seq = y[i]
        for j in range(sequence_length):
            if counts[i] == 5:
                break
            if seq[j] == 1:
                counts[i] += 1
            elif seq[j] == 0 and 5 >= counts[i] > 0:
                counts[i] = 0
        if counts[i] < 5:
            counts[i] = 0
        else:
            counts[i] = 1
    return y_out * counts


def compute_sequence_accuracy(y_true, y_pred):
    """
    compute the sequence-based accuracy
    :param y_true: true labels of shape [#sequences, sequence_length]
    :param y_pred: predicted labels of shape [#sequences, sequence_length]
    :return: sequence-based accuracy value
    """
    ytrue = get_sequence_label(y_true)
    ypred = get_sequence_label(y_pred)
    print(confusion_matrix(ytrue, ypred))
    accuracy = float(np.sum(ytrue == ypred)) / ytrue.shape[0]
    print('Sequence-based accuracy={}'.format(accuracy))
    return accuracy

