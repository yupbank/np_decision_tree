from functools import partial

import numpy as np


def _greedy(Xf, y):
    thresholds, inverse, count = np.unique(
        Xf, return_counts=True, return_inverse=True)

    if thresholds.size <= 1:
        return -1.0, -1.0

    y_sum = np.zeros(count.size)
    np.add.at(y_sum, inverse, y)
    if y.size != count.size:
        print(y.size, count.size)
    cum_count = np.cumsum(count)

    left_sums = np.cumsum(y_sum)
    total_sum, left_sums = left_sums[-1], left_sums[:-1]
    total_count, left_counts = cum_count[-1], cum_count[:-1]

    surrogate = np.square(left_sums) / left_counts + \
                np.square(total_sum - left_sums) / (total_count - left_counts)
    if surrogate.size == 0:
        return -1.0, -1.0

    index = np.argmax(surrogate)
    threshold = thresholds[index]

    improvement = (surrogate[index] -
                   np.square(total_sum) / total_count) / total_count

    return improvement, threshold


def greedy_split(X, y):
    split_column = partial(_greedy, y=y)
    improvements = np.apply_along_axis(split_column, 0, X)
    best_index = np.argmax(improvements[0])
    best_improvement, best_threshold = improvements[:, best_index]
    constant_attr_mask = np.logical_and(
        improvements[0] == -1.0, improvements[1] == -1.0)
    return best_index, best_threshold, best_improvement, constant_attr_mask


def greedy_split_v2(X, y):
    orders = np.argsort(X, axis=0)

    total_sum = y.sum()
    total_count = X.shape[0]
    left_sums = np.cumsum(y[orders], axis=0)[:-1]
    left_count = np.arange(1, total_count)[:, np.newaxis]

    surrogate = np.square(left_sums) / left_count + \
                np.square(total_sum - left_sums) / (total_count - left_count)

    best_row, best_column = np.unravel_index(np.argmax(surrogate), surrogate.shape)
    best_threshold = X[best_row, best_column]
    best_improvement = surrogate[best_row, best_column] - np.square(np.mean(y))
    return best_column, best_threshold, best_improvement, np.zeros(X.shape[1], dtype=np.bool)
