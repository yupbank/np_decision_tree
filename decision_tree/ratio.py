import numpy as np
from collections import namedtuple, deque

Task = namedtuple('Task', 'orders parent is_left depth')
Tree = namedtuple(
    'DecisionTree', 'children_left children_right feature threshold value')


class DecisionTree:
    def __init__(self, max_node):
        children_left, children_right = np.empty(
            max_node-1, dtype=np.int64), np.empty(max_node-1, dtype=np.int64)
        feature = np.empty(max_node-1, dtype=np.int64)
        value = np.empty(max_node-1)
        threshold = np.empty(max_node-1)

        children_left[:] = -1
        children_right[:] = -1
        feature[:] = -2
        threshold[:] = -2

        self.tree_ = Tree(children_left, children_right,
                          feature, threshold, value)
        self.max_node_ = -1

    def new_node_from_task(self, task):
        self.max_node_ += 1
        if task.parent is not None:
            to_save = self.tree_.children_left if task.is_left else self.tree_.children_right
            to_save[task.parent] = self.max_node_
        return self.max_node_

    def add_leaf(self, node_id, value):
        self.tree_.value[node_id] = value

    def add_binary(self, node_id, col, threshold):
        self.tree_.threshold[node_id] = threshold
        self.tree_.feature[node_id] = col

    @staticmethod
    def is_leaf(orders, min_sample_per_leaf=10):
        row, column = orders.shape
        if row <= min_sample_per_leaf:
            return True
        else:
            return False

    def final(self):
        return self


def np_gene(n): return np.reciprocal(
    np.arange(1, n, dtype=np.float32)*np.arange(n-1, 0, -1))


def best_variance_improvements(ys):
    residual = np.arange(1, ys.shape[0]+1)*(ys[-1][-1]/ys.shape[0])
    tmp = np.abs(ys-residual[:, np.newaxis])
    tmp_order = np.argmax(tmp, axis=1)
    value = tmp[np.arange(ys.shape[0]), tmp_order]
    res = np_gene(ys.shape[0])*np.square(value)[:-1]
    max_col = np.argmax(res)
    return max_col, tmp_order[max_col], res[max_col]


def best_ratio_improvments(ys):
    size, feature_size = ys.shape
    total_sum = ys[-1][-1]
    mean = total_sum/size
    ps = ys[:-1]/total_sum
    q = np.arange(1, size)/size
    qs = np.tile(q, feature_size).reshape(feature_size, -1).T
    values = np.square(ps-qs)*np.reciprocal(qs*(1-qs))

    best_row, best_column = np.unravel_index(
        np.argmax(values), values.shape)
    return best_row, best_column,  values[best_row, best_column]*(mean**2)


def greedy_regression_split(orders, x, y, v):
    cumsums = np.cumsum(y[orders], axis=0)
    best_row, best_column, best_improvement = best_variance_improvements(
        cumsums)
    best_data_row = orders[best_row, best_column]
    threshold = x[best_data_row, best_column]
    left_rows = orders[:, best_column][:best_row+1]
    return threshold, best_column, best_improvement, left_rows


def split_orders(orders, index, n_samples):
    bool_aux = np.zeros(n_samples, dtype=np.bool)
    bool_aux[index] = 1
    left_masks = bool_aux[orders]
    left_orders = orders.T[left_masks.T].reshape(-1, index.shape[0]).T
    right_orders = orders.T[~left_masks.T].reshape(
        -1, orders.shape[0]-index.shape[0]).T

    return left_orders, right_orders


def split_orders_v2(orders, index, n_samples):
    bool_aux = np.zeros(n_samples, dtype=np.bool)
    bool_aux[index] = 1
    left_masks = bool_aux[orders]
    ranks = np.argsort(left_masks.T.ravel())
    new_res = orders.T.ravel()[ranks]
    left = new_res[:index.shape[0]*n_samples].reshape(-1, n_samples).T
    right = new_res[index.shape[0]*n_samples:].reshape(-1, n_samples).T
    return left, right


leaf_from_data = np.mean


def build_regression_tree(X, y, max_depth=2, max_feature=100, min_improvement=1e-7, min_sample_leaf=1, v=1):
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
                task.orders, X, y, v)
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


def build_regression_tree_v1_5(X, y, max_depth=2, max_feature=100, min_improvement=1e-7, min_sample_leaf=1, v=1):
    max_node = 2 ** (max_depth + 1)
    tree = DecisionTree(max_node)

    orders = np.argsort(X, axis=0)

    tasks = deque([Task(orders, parent=None, is_left=True, depth=0)])
    while tasks:
        task = tasks.pop()
        node_id = tree.new_node_from_task(task)
        if tree.is_leaf(task.orders, min_sample_leaf) or task.depth >= max_depth:
            tree.add_leaf(node_id, leaf_from_data(y[task.orders]))
        else:
            threshold, best_column, best_improvement, left_rows = greedy_regression_split(
                task.orders, X, y, v)
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
