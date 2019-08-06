import numpy as np
from collections import namedtuple
from decision_tree.base import DecisionTree, is_leaf_v2
Task = namedtuple('Task', 'orders parent is_left depth')
# variance_reduce =  (
#     np.square(candidate_left_sums)/candidate_left_counts
#     + np.square(candidate_right_sums)/candidate_right_counts
#     - np.square(total_sums)/total_count
# ) / total_count


def best_variance_improvements(cumsums):
    assert len(cumsums.shape) >= 2

    counts = 1+np.arange(cumsums.shape[0])[:, np.newaxis]
    candidate_left_sums, total_sums = cumsums[:-1], cumsums[-1]
    candidate_left_counts, total_count = counts[:-1], counts[-1]

    candidate_right_sums = total_sums - candidate_left_sums
    candidate_right_counts = total_count - candidate_left_counts
    surrogate = (
        np.square(candidate_left_sums)/candidate_left_counts
        + np.square(candidate_right_sums)/candidate_right_counts
    )

    best_row, best_column = np.unravel_index(
        np.argmax(surrogate), surrogate.shape)

    best_surrogate = surrogate[best_row, best_column]

    variance_reduce = (
        best_surrogate/total_count -
        np.square(total_sums[best_column]/total_count)
    )
    return best_row, best_column, variance_reduce


def greedy_regression_split(orders, x, y):
    cumsums = np.cumsum(y[orders], axis=0)
    best_row, best_column, best_improvement = best_variance_improvements(
        cumsums)
    best_data_row = orders[best_row, best_column]
    threshold = x[best_data_row, best_column]
    left_rows = orders[:best_row+1, best_column]
    return threshold, best_column, best_improvement, left_rows


def cal_mask(orders, index, n_samples):
    bool_aux = np.zeros(n_samples, dtype=np.bool)
    bool_aux[index] = 1
    return bool_aux[orders]


def mask_with_fix_column_size_2d(array, bool_mask, shape):
    return array.T[bool_mask.T].reshape(shape[1], shape[0]).T


def split_orders(orders, index, n_samples, mask_select=mask_with_fix_column_size_2d):
    left_mask = cal_mask(orders, index, n_samples)
    right_mask = ~left_mask

    total_row, total_column = orders.shape
    left_shape = (index.shape[0], total_column)
    right_shape = (total_row-index.shape[0], total_column)

    left_orders = mask_select(orders, left_mask, left_shape)
    right_orders = mask_select(orders, right_mask, right_shape)

    return left_orders, right_orders


leaf_from_data = np.mean


def build_regression_tree(X, y, max_depth=2, max_feature=100, min_improvement=1e-7, min_sample_leaf=1):
    max_node = 2 ** (max_depth + 1)
    tree = DecisionTree(max_node)

    orders = np.argsort(X, axis=0)

    tasks = [Task(orders, parent=None, is_left=True, depth=0)]
    while tasks:
        task = tasks.pop()
        node_id = tree.new_node_from_task(task)
        if is_leaf_v2(task.orders, min_sample_leaf) or task.depth >= max_depth:
            tree.add_leaf(node_id, leaf_from_data(y[task.orders]))
        else:
            threshold, best_column, best_improvement, left_rows = greedy_regression_split(
                task.orders, X, y)
            if best_improvement < min_improvement:
                tree.add_leaf(node_id, leaf_from_data(y[task.orders]))
            else:
                tree.add_binary_v2(node_id, best_column,
                                   threshold)
                left_orders, right_orders = split_orders(
                    task.orders, left_rows, y.shape[0])
                tasks.append(
                    Task(right_orders, parent=node_id, is_left=False, depth=task.depth + 1))
                tasks.append(
                    Task(left_orders, parent=node_id, is_left=True, depth=task.depth + 1))

    return tree.final()

    cumsums = np.cumsum(y[orders], axis=0)
