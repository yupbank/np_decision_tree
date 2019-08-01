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
    cutting_rank = np.random.choice(X.shape[0]-1, X.shape[1])
    left_masks = indexes <= cutting_rank
    improvements = cal_improvements(y, left_masks)
    best_index = np.argmax(improvements)
    data_row_index = indexes[cutting_rank[best_index], best_index]
    return best_index, X[data_row_index, best_index], improvements[best_index], np.zeros(X.shape[1], dtype=np.bool)


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
