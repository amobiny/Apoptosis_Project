import numpy as np
import os
import h5py
import skimage.transform
from tqdm import tqdm_notebook
import sys


def one_hot(y, num_class):
    y_ohe = (np.arange(num_class) == y[:, None]).astype(np.float32)
    return y_ohe


def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :, :, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def split_valid_test(x, y, num_valid_per_class):
    idx_0 = np.where(np.argmax(y, axis=1) == 0)[0]
    idx_1 = np.where(np.argmax(y, axis=1) == 1)[0]
    x0, x1 = x[idx_0], x[idx_1]
    y0, y1 = y[idx_0], y[idx_1]
    x0, y0 = randomize(x0, y0)
    x1, y1 = randomize(x1, y1)
    x_valid = np.concatenate((x0[:num_valid_per_class], x1[:num_valid_per_class]), axis=0)
    y_valid = np.concatenate((y0[:num_valid_per_class], y1[:num_valid_per_class]), axis=0)
    x_valid, y_valid = randomize(x_valid, y_valid)
    x_test = np.concatenate((x0[num_valid_per_class:], x1[num_valid_per_class:]), axis=0)
    y_test = np.concatenate((y0[num_valid_per_class:], y1[num_valid_per_class:]), axis=0)
    x_test, y_test = randomize(x_test, y_test)
    return x_valid, y_valid, x_test, y_test


def resize(x, new_size=28):
    """
    Get and resize images or masks
    :param x: images os shape [#images, old_size, old_size, #channels]
    :param new_size: resize images to this size
    :return: resized images os shape [#images, new_size, new_size, #channels]
    """
    X = np.zeros((x.shape[0], new_size, new_size, x.shape[-1]))
    sys.stdout.flush()
    for n in tqdm_notebook(range(x.shape[0]), total=len(x)):
        img = x[n]
        X[n] = skimage.transform.resize(img, (new_size, new_size, 1), mode='constant', preserve_range=True)
    return np.minimum(np.maximum(X, 0), 1)


def merge_all_classes(x_effector, y_effector, x_target, y_target):
    x = np.concatenate((x_effector, x_target), axis=0)
    y_eff_temp = np.concatenate((y_effector, np.zeros((y_effector.shape[0], 2))), axis=1)
    y_tar_temp = np.concatenate((np.zeros((y_target.shape[0], 2)), y_target), axis=1)
    y = np.concatenate((y_eff_temp, y_tar_temp), axis=0)
    x, y = randomize(x, y)
    return x, y


def minor_augmentation(x_51, x_28, y, num_per_class):
    idx_0 = np.where(np.argmax(y, axis=1) == 0)[0]
    idx_1 = np.where(np.argmax(y, axis=1) == 1)[0]
    selected_0 = np.random.choice(idx_0, num_per_class)
    selected_1 = np.random.choice(idx_1, num_per_class)

    x_flip_51_0 = np.array([np.fliplr(x) for x in x_51[selected_0]])
    x_flip_51_1 = np.array([np.fliplr(x) for x in x_51[selected_0]])
    x_51 = np.concatenate((x_51, x_flip_51_0, x_flip_51_1), axis=0)

    x_flip_28_0 = np.array([np.fliplr(x) for x in x_28[selected_0]])
    x_flip_28_1 = np.array([np.fliplr(x) for x in x_28[selected_0]])
    x_28 = np.concatenate((x_28, x_flip_28_0, x_flip_28_1), axis=0)
    y = np.concatenate((y, y[selected_0], y[selected_1]), axis=0)
    return x_51, x_28, y
