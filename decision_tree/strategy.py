import numpy as np


def surrogate_variance_improvements(left_sums, left_counts, total_sums=None, total_count=None):
    if total_sums is None:
        candidate_left_sums, total_sums = left_sums[:-1], left_sums[-1]
    else:
        candidate_left_sums = left_sums
    if total_count is None:
        candidate_left_counts, total_count = left_counts[:-1], left_counts[-1]
    else:
        candidate_left_counts = left_counts

    candidate_right_sums = total_sums - candidate_left_sums
    candidate_right_counts = total_count - candidate_left_counts

    # variance_reduce =  (
    #     np.square(candidate_left_sums)/candidate_left_counts
    #     + np.square(candidate_right_sums)/candidate_right_counts
    #     - np.square(total_sums)/total_count
    # ) / total_count
    return (
        np.square(candidate_left_sums)/candidate_left_counts
        + np.square(candidate_right_sums)/candidate_right_counts
    )


def surrogate_variance_improvements_v2(left_sums, left_counts, total_sums=None, total_count=None):
    if total_sums is None:
        candidate_left_sums, total_sums = left_sums[:-1], left_sums[-1]
    else:
        candidate_left_sums = left_sums
    if total_count is None:
        candidate_left_counts, total_count = left_counts[:-1], left_counts[-1]
    else:
        candidate_left_counts = left_counts

    candidate_right_sums = total_sums - candidate_left_sums
    candidate_right_counts = total_count - candidate_left_counts

    return (
        np.square(candidate_left_sums)/candidate_left_counts
        + np.square(candidate_right_sums)/candidate_right_counts
        - np.square(total_sums)/total_count
    ) / total_count
    # variance_reduce =  (
    #     np.square(candidate_left_sums)/candidate_left_counts
    #     + np.square(candidate_right_sums)/candidate_right_counts
    #     - np.square(total_sums)/total_count
    # ) / total_count
    return (
        np.square(candidate_left_sums)/candidate_left_counts
        + np.square(candidate_right_sums)/candidate_right_counts
    )


def surrogate_to_variance(value, total_count, y_mean):
    return value/total_count - np.square(y_mean)


def surrogate_gini_improvements(left_class_counts, left_counts, total_class_counts=None, total_count=None):
    if total_class_counts is None:
        candidate_left_class_counts, total_class_counts = left_class_counts[:-1],\
            left_class_counts[-1]
    else:
        candidate_left_class_counts = left_class_counts
    if total_count is None:
        candidate_left_counts, total_count = left_counts[:-1], left_counts[-1]
    else:
        candidate_left_counts = left_counts
    candidate_right_class_counts = total_class_counts - candidate_left_class_counts
    candidate_right_counts = total_count - candidate_left_counts
    candidate_left_sum_of_squared = np.square(
        candidate_left_class_counts).sum(axis=2)
    candidate_right_sum_of_squared = np.square(
        candidate_right_class_counts).sum(axis=2)
    #gini_improve = (candidate_left_sum_of_squared/candidate_left_counts
    #        + candidate_right_sum_of_squared/candidate_right_counts
    #        - np.square(total_class_counts).sum()/total_count
    #        )/total_count
    return (
        candidate_left_sum_of_squared/candidate_left_counts
        + candidate_right_sum_of_squared/candidate_right_counts
    )


def surrogate_gini_improvements_v2(left_class_counts, left_counts, total_class_counts=None, total_count=None):
    if total_class_counts is None:
        candidate_left_class_counts, total_class_counts = left_class_counts[:-1],\
            left_class_counts[-1]
    else:
        candidate_left_class_counts = left_class_counts
    if total_count is None:
        candidate_left_counts, total_count = left_counts[:-1], left_counts[-1]
    else:
        candidate_left_counts = left_counts
    candidate_right_class_counts = total_class_counts - candidate_left_class_counts
    candidate_right_counts = total_count - candidate_left_counts
    candidate_left_sum_of_squared = np.square(
        candidate_left_class_counts).sum(axis=2)
    candidate_right_sum_of_squared = np.square(
        candidate_right_class_counts).sum(axis=2)
    return (candidate_left_sum_of_squared/candidate_left_counts
            + candidate_right_sum_of_squared/candidate_right_counts
            - np.square(total_class_counts).sum()/total_count
            )/total_count


def surrogate_p_at_k_improvements(left_class_counts, left_counts, total_class_counts=None, total_count=None, k=1):
    if total_class_counts is None:
        candidate_left_class_counts, total_class_counts = left_class_counts[:-1],\
            left_class_counts[-1]
    else:
        candidate_left_class_counts = left_class_counts
    if total_count is None:
        candidate_left_counts, total_count = left_counts[:-1], left_counts[-1]
    else:
        candidate_left_counts = left_counts
    candidate_right_class_counts = total_class_counts - candidate_left_class_counts
    candidate_right_counts = total_count - candidate_left_counts
    candidate_left_top_class_count = np.max(
        candidate_left_class_counts, axis=2)
    candidate_right_top_class_count = np.max(
        candidate_right_class_counts, axis=2)
    return (
        candidate_left_top_class_count
        + candidate_right_top_class_count
    )


def surrogate_to_p_at_k(value, y):
    return (value - np.max(y.sum(axis=0))) / y.shape[0]


def surrogate_to_gini(value, y):
    return value/y.shape[0] - np.sum(np.square(y.sum(axis=0)/y.sum()))


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


def cal_gini_improvements(y, masks):
    total_count = y.shape[0]
    total_class_counts = y.sum(axis=0)
    left_class_counts = masks.T.dot(y)
    left_counts = masks.sum(axis=0)
    right_counts = total_count-left_counts
    left_squared_sums = np.square(left_class_counts).sum(axis=1)
    right_squared_sums = np.square(
        total_class_counts-left_class_counts).sum(axis=1)
    return (left_squared_sums/left_counts
            + right_squared_sums/right_counts
            - np.square(total_class_counts).sum()/total_count
            )/total_count


def random_split_v2(X, y, cal_improvements=cal_variance_improvements):
    thresholds = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0))
    masks = X <= thresholds
    left_sums = masks.T.dot(y)
    left_counts = masks.sum(axis=0)
    total_sum = y.sum()
    total_count = y.shape[0]
    improvements = surrogate_variance_improvements(
        left_sums, left_counts, total_sum, total_count)
    best_index = np.argmax(improvements)
    return best_index, thresholds[best_index], surrogate_to_variance(improvements[best_index], total_count, np.mean(y)), np.zeros(X.shape[1], dtype=np.bool)


def random_classify(X, y):
    thresholds = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0))
    masks = X <= thresholds
    left_sums = masks.T.dot(y)[np.newaxis, :]
    left_counts = masks.sum(axis=0)
    total_sums = y.sum(axis=0)
    total_count = y.shape[0]
    improvements = surrogate_gini_improvements(
        left_sums, left_counts, total_sums, total_count).ravel()
    best_index = np.argmax(improvements)
    return best_index, thresholds[best_index], surrogate_to_gini(improvements[best_index], y), np.zeros(X.shape[1], dtype=np.bool)


def random_classify_p_at_k(X, y, k=1):
    thresholds = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0))
    masks = X <= thresholds
    left_sums = masks.T.dot(y)[np.newaxis, :]
    left_counts = masks.sum(axis=0)
    total_sums = y.sum(axis=0)
    total_count = y.shape[0]
    improvements = surrogate_p_at_k_improvements(
        left_sums, left_counts, total_sums, total_count, k=k).ravel()
    best_index = np.argmax(improvements)
    return best_index, thresholds[best_index], surrogate_to_p_at_k(improvements[best_index], y), np.zeros(X.shape[1], dtype=np.bool)


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


def greedy_split_v2(X, y):
    orders = np.argsort(X, axis=0)
    total_sums = y.sum()
    total_count = X.shape[0]
    left_sums = np.cumsum(y[orders], axis=0)[:-1]
    left_counts = np.arange(1, total_count)[:, np.newaxis]
    return left_sums, left_counts

    improvements = surrogate_variance_improvements(
        left_sums, left_counts, total_sums, total_count)
    best_row, best_column = np.unravel_index(
        np.argmax(improvements), improvements.shape)
    best_data_row = orders[best_row, best_column]
    best_threshold = X[best_data_row, best_column]
    best_improvement = improvements[best_row, best_column]
    return best_column, best_threshold, surrogate_to_variance(best_improvement, total_count, np.mean(y)), np.zeros(X.shape[1], dtype=np.bool)


def select_by_mask(array, bool_mask, shape):
    return array.T[bool_mask.T].reshape(shape[1], shape[0]).T


def split_orders_and_cumsums_v2(orders, cumsums, col, row, y):
    y_value_aux = np.zeros_like(y)
    y_bool_aux = np.zeros_like(y, dtype=np.bool)

    left_index = orders[:row, col]

    left_shape = (row, orders.shape[1])
    right_shape = (orders.shape[0]-row, orders.shape[1])

    y_bool_aux[left_index] = 1
    y_value_aux[left_index] = y[left_index]

    left_mask = y_bool_aux[orders]
    right_mask = ~left_mask
    gross_left_cumsums = np.cumsum(y_value_aux[orders], axis=0)
    gross_right_cumsums = cumsums - gross_left_cumsums

    left_orders = select_by_mask(
        orders, left_mask, left_shape)
    left_cumsums = select_by_mask(
        gross_left_cumsums, left_mask, left_shape)

    right_orders = select_by_mask(
        orders, right_mask, right_shape)
    right_cumsums = select_by_mask(
        gross_right_cumsums, right_mask, right_shape)
    return left_orders, left_cumsums, right_orders, right_cumsums


def greedy_split_v3(orders, cumsums, y):
    counts = 1+np.arange(orders.shape[0])[:, np.newaxis]

    improvements = surrogate_variance_improvements_v2(
        cumsums, counts)

    best_order_row, best_column = np.unravel_index(
        np.argmax(improvements), improvements.shape)

    best_data_row = orders[best_order_row, best_column]

    best_improvement = improvements[best_order_row, best_column]

    return best_order_row, best_column, best_data_row, best_improvement


def greedy_classification(X, y):
    orders = np.argsort(X, axis=0)
    left_sums = np.cumsum(y[orders], axis=0)
    left_counts = np.arange(1, y.shape[0]+1)[:, np.newaxis]
    improvements = surrogate_gini_improvements(left_sums, left_counts)
    best_row, best_column = np.unravel_index(
        np.argmax(improvements), improvements.shape)
    best_data_row = orders[best_row, best_column]
    best_threshold = X[best_data_row, best_column]
    best_improvement = improvements[best_row, best_column]
    return best_column, best_threshold, surrogate_to_gini(best_improvement, y), np.zeros(X.shape[1], dtype=np.bool)


def greedy_classification_v3(orders, cumsums, y):
    counts = 1+np.arange(orders.shape[0])[:, np.newaxis]
    improvements = surrogate_gini_improvements_v2(
        cumsums, counts)
    best_order_row, best_column = np.unravel_index(
        np.argmax(improvements), improvements.shape)

    best_data_row = orders[best_order_row, best_column]

    best_improvement = improvements[best_order_row, best_column]

    return best_order_row, best_column, best_data_row, best_improvement


def greedy_classification_both(X, y, k=1):
    orders = np.argsort(X, axis=0)
    left_sums = np.cumsum(y[orders], axis=0)
    left_counts = np.arange(1, y.shape[0]+1)[:, np.newaxis]
    improvements_a = surrogate_gini_improvements(left_sums, left_counts)
    improvements_b = surrogate_p_at_k_improvements(left_sums, left_counts, k=k)
    if np.argmax(improvements_a) != np.argmax(improvements_b):
        print(np.argmax(improvements_a), np.max(improvements_a), np.mean(improvements_a),
              np.argmax(improvements_b), np.max(improvements_b), np.mean(improvements_b))
    best_row, best_column = np.unravel_index(
        np.argmax(improvements_a), improvements_a.shape)
    best_data_row = orders[best_row, best_column]
    best_threshold = X[best_data_row, best_column]
    best_improvement = improvements_a[best_row, best_column]
    return best_column, best_threshold, surrogate_to_gini(best_improvement, y), np.zeros(X.shape[1], dtype=np.bool)


def greedy_classification_p_at_k(X, y, k=1):
    orders = np.argsort(X, axis=0)
    left_sums = np.cumsum(y[orders], axis=0)
    left_counts = np.arange(1, y.shape[0]+1)[:, np.newaxis]
    improvements = surrogate_p_at_k_improvements(left_sums, left_counts, k=k)
    best_row, best_column = np.unravel_index(
        np.argmax(improvements), improvements.shape)
    best_data_row = orders[best_row, best_column]
    best_threshold = X[best_data_row, best_column]
    best_improvement = improvements[best_row, best_column]
    return best_column, best_threshold, surrogate_to_p_at_k(best_improvement, y), np.zeros(X.shape[1], dtype=np.bool)
