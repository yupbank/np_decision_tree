import numpy as np


def cal_improvements(y, masks):
    total_count = y.size
    total_sum = y.sum()
    left_sums = masks.T.dot(y)
    right_sums = total_sum - left_sums
    left_counts = masks.sum(axis=0)
    right_counts = total_count - left_counts
    return np.square(left_sums) / left_counts + \
        np.square(right_sums) / right_counts \
        - np.square(np.mean(y))


def random_split(X, y):
    indexes = np.argsort(X, axis=0)
    min_max_index = indexes[[0, -1]]
    auxiliary_col, _ = np.meshgrid(np.arange(X.shape[1]), [0, 1])
    min_max_value = X[min_max_index, auxiliary_col]
    constant_attr_mask = min_max_value[0] == min_max_value[1]
    min_max_value = min_max_value[:, ~constant_attr_mask]
    thresholds = np.random.uniform(min_max_value[0], min_max_value[1])
    left_masks = X <= thresholds
    improvements = cal_improvements(y, left_masks)
    best_index = np.argmax(improvements)
    return best_index, thresholds[best_index], improvements[best_index], constant_attr_mask
