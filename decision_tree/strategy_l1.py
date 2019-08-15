import numpy as np
from decision_tree.stream_median import StreamMedian
from decision_tree.one import DecisionTree, Task, split_orders
from np_stream_median import cummedian


def l1_difference(y):
    return np.sum(np.abs(y-np.median(y, axis=0)), axis=0)


def running_median_v2(seq):
    cummedian = StreamMedian()
    return np.array([cummedian(i) for i in seq])


def running_median(seq):
    elements = np.zeros((seq.size, seq.size))
    elements[np.arange(seq.size), seq] = seq+1
    elements.cumsum(axis=0, out=elements)
    elements[elements == 0] = np.nan
    return np.nanmedian(elements, axis=1, overwrite_input=True) - 1


def running_m(seq):
    mask = np.zeros(seq.size, dtype=np.bool)
    res = np.zeros(seq.size)
    x, y = 0, 0
    for n, i in enumerate(seq):
        mask[i] = 1
        index = np.flatnonzero(mask)
        res[n] = np.mean(index[[x, y]])
        if n % 2 == 0:
            x, y = x, y+1
        if n % 2 == 1:
            x, y = x+1, y
    return res


def best_l1_improvements(ys):
    left_median = np.apply_along_axis(running_median_v2, 0, ys)
    left_ys = ys[1:]
    small_left_median = left_median[:-1, 0]
    large_left_median = left_median[:-1, 1]
    left_improvements = np.cumsum(
        np.where(left_ys <= small_left_median,
                 small_left_median-left_ys,
                 0)
        + np.where(left_ys >= large_left_median,
                   left_ys - large_left_median,
                   0),
        axis=0
    )
    left_improvements = np.concatenate(
        [np.zeros((1, ys.shape[1])), left_improvements])
    right_median = np.apply_along_axis(
        running_median_v2, 0, ys[::-1])[::-1]
    right_ys = ys
    small_right_median = right_median[:, 0]
    large_right_median = right_median[:, 1]

    right_improvements = np.cumsum(
        (np.where(right_ys <= small_right_median,
                  large_right_median-right_ys, 0)
         + np.where(right_ys >= large_right_median,
                    right_ys - small_right_median,
                    0)
         )[::-1],
        axis=0
    )[::-1]

    improvements = right_improvements[1:] + left_improvements[:-1]
    best_row, best_column = np.unravel_index(
        np.argmin(improvements), improvements.shape)
    return best_row, best_column, improvements[best_row, best_column]


def best_l1_improvements_v2(ys):
    left_median = np.apply_along_axis(cummedian, 0, ys)
    # left_median = np.apply_along_axis(running_median_v2, 0, ys)
    left_ys = ys[1:]
    small_left_median = left_median[:-1, 0]
    large_left_median = left_median[:-1, 1]
    left_improvements = np.cumsum(
        np.where(left_ys <= small_left_median,
                 small_left_median-left_ys,
                 0)
        + np.where(left_ys >= large_left_median,
                   left_ys - large_left_median,
                   0),
        axis=0
    )
    left_improvements = np.concatenate(
        [np.zeros((1, ys.shape[1])), left_improvements])
    # right_median = np.apply_along_axis(
    #     running_median_v2, 0, ys[::-1])[::-1]
    right_median = np.apply_along_axis(
        cummedian, 0, ys[::-1])[::-1]
    right_ys = ys
    small_right_median = right_median[:, 0]
    large_right_median = right_median[:, 1]

    right_improvements = np.cumsum(
        (np.where(right_ys <= small_right_median,
                  large_right_median-right_ys, 0)
         + np.where(right_ys >= large_right_median,
                    right_ys - small_right_median,
                    0)
         )[::-1],
        axis=0
    )[::-1]

    improvements = right_improvements[1:] + left_improvements[:-1]
    best_row, best_column = np.unravel_index(
        np.argmin(improvements), improvements.shape)
    return best_row, best_column, improvements[best_row, best_column]


def greedy_regression_split(orders, X, y):
    best_row, best_column, best_improvement = best_l1_improvements(
        y[orders])
    best_data_row = orders[best_row, best_column]
    threshold = X[best_data_row, best_column]
    left_rows = orders[:best_row+1, best_column]
    return threshold, best_column, best_improvement, left_rows


def greedy_regression_split_v2(orders, X, y):
    best_row, best_column, best_improvement = best_l1_improvements_v2(
        y[orders])
    best_data_row = orders[best_row, best_column]
    threshold = X[best_data_row, best_column]
    left_rows = orders[:best_row+1, best_column]
    return threshold, best_column, best_improvement, left_rows


def build_regression_tree(X, y, max_depth=2, max_feature=100, min_improvement=1e-7, min_sample_leaf=1, leaf_from_data=np.median):
    max_node = 2 ** (max_depth + 1)
    tree = DecisionTree(max_node)

    orders = np.argsort(X, axis=0)

    tasks = [Task(orders, parent=None, is_left=True, depth=0)]
    while tasks:
        task = tasks.pop()
        node_id = tree.new_node_from_task(task)
        if tree.is_leaf(task.orders, min_sample_leaf) or task.depth >= max_depth:
            tree.add_leaf(node_id, leaf_from_data(y[task.orders]))
        else:
            threshold, best_column, best_improvement, left_rows = greedy_regression_split(
                task.orders, X, y)
            if best_improvement < min_improvement:
                tree.add_leaf(node_id, leaf_from_data(y[task.orders]))
            else:
                tree.add_binary(node_id, best_column,
                                threshold)
                left_orders, right_orders = split_orders(
                    task.orders, left_rows, y.shape[0])
                tasks.append(
                    Task(right_orders, parent=node_id, is_left=False, depth=task.depth + 1))
                tasks.append(
                    Task(left_orders, parent=node_id, is_left=True, depth=task.depth + 1))

    return tree.final()


def build_regression_tree_v2(X, y, max_depth=2, max_feature=100, min_improvement=1e-7, min_sample_leaf=1, leaf_from_data=np.median):
    max_node = 2 ** (max_depth + 1)
    tree = DecisionTree(max_node)

    orders = np.argsort(X, axis=0)

    tasks = [Task(orders, parent=None, is_left=True, depth=0)]
    while tasks:
        task = tasks.pop()
        node_id = tree.new_node_from_task(task)
        if tree.is_leaf(task.orders, min_sample_leaf) or task.depth >= max_depth:
            tree.add_leaf(node_id, leaf_from_data(y[task.orders]))
        else:
            threshold, best_column, best_improvement, left_rows = greedy_regression_split_v2(
                task.orders, X, y)
            if best_improvement < min_improvement:
                tree.add_leaf(node_id, leaf_from_data(y[task.orders]))
            else:
                tree.add_binary(node_id, best_column,
                                threshold)
                left_orders, right_orders = split_orders(
                    task.orders, left_rows, y.shape[0])
                tasks.append(
                    Task(right_orders, parent=node_id, is_left=False, depth=task.depth + 1))
                tasks.append(
                    Task(left_orders, parent=node_id, is_left=True, depth=task.depth + 1))

    return tree.final()
