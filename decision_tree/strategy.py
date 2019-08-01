import numpy as np


def cal_variance_improvements(y, masks):
    total_count = y.size
    total_sum = y.sum()
    left_sums = masks.T.dot(y)
    right_sums = total_sum - left_sums
    left_counts = masks.sum(axis=0)
    right_counts = total_count - left_counts
    return (np.square(left_sums) / left_counts +
            np.square(right_sums) / right_counts) / total_count \
        - np.square(np.mean(y))


def cal_gini_improvement(y, masks, num_class=2):
    encoding = np.eye(num_class)
    ye = encoding[y]
    total_class_count = ye.sum(axis=0)
    total_count = total_class_count.sum()
    left_class_counts = masks.T.dot(ye)
    left_squared_sum = np.square(left_class_counts).sum(axis=1)
    right_squared_sum = np.square(
        total_class_count-left_class_counts).sum(axis=1)
    return (left_squared_sum/left_class_count + right_squared_sum/right_class_count)/total_count


def random_split(X, y, cal_improvements=cal_variance_improvements):
    """
    Randomly pick a threshold among  all features for maximum variance reduction
    """
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


def greedy_split(X, y):
    """
    Greedy split all features for maximum variance reduction
    """
    orders = np.argsort(X, axis=0)

    total_sum = y.sum()
    total_count = X.shape[0]
    left_sums = np.cumsum(y[orders], axis=0)[:-1]
    left_counts = np.arange(1, total_count)[:, np.newaxis]

    surrogate = np.square(left_sums) / left_counts + \
        np.square(total_sum - left_sums) / (total_count - left_counts)

    best_row, best_column = np.unravel_index(
        np.argmax(surrogate), surrogate.shape)
    best_data_row = orders[best_row, best_column]
    best_threshold = X[best_data_row, best_column]
    best_improvement = surrogate[best_row,
                                 best_column] / total_count - np.square(np.mean(y))
    return best_column, best_threshold, best_improvement, np.zeros(X.shape[1], dtype=np.bool)
