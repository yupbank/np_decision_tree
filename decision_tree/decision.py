import numpy as np


def greedy_split(X, y):
    orders = np.argsort(X, axis=0)

    total_sum = y.sum()
    total_count = X.shape[0]
    left_sums = np.cumsum(y[orders], axis=0)[:-1]
    left_counts = np.arange(1, total_count)[:, np.newaxis]

    surrogate = np.square(left_sums) / left_counts + np.square(total_sum - left_sums) / (total_count - left_counts)

    best_row, best_column = np.unravel_index(np.argmax(surrogate), surrogate.shape)
    best_data_row = orders[best_row, best_column]
    best_threshold = X[best_data_row, best_column]
    best_improvement = surrogate[best_row, best_column] / y.size - np.square(np.mean(y))
    return best_column, best_threshold, best_improvement, np.zeros(X.shape[1], dtype=np.bool)