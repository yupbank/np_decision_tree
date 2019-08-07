import numpy as np
from collections import namedtuple

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
        if row <= min_sample_per_leaf*2:
            return True
        else:
            return False

    def final(self):
        return self


def best_variance_improvements(cumsums):
    n = cumsums.shape[0]
    total_sum = cumsums[-1][-1]
    parent_mean = total_sum/n
    ratio_a = np.sqrt(np.reciprocal(
        np.arange(1, n, dtype=np.float32)*np.arange(n-1, 0, -1, dtype=np.float32)))[:, np.newaxis]
    ratio_b = np.sqrt(np.arange(1, n, dtype=np.float32) /
                      np.arange(n-1, 0, -1, dtype=np.float32))[:, np.newaxis]
    improvements = np.square(ratio_a *
                             cumsums[:-1] - parent_mean * ratio_b)
    best_row, best_column = np.unravel_index(
        np.argmax(improvements), improvements.shape)
    return best_row, best_column, improvements[best_row, best_column]


def greedy_regression_split(orders, x, y):
    cumsums = np.cumsum(y[orders], axis=0)
    best_row, best_column, best_improvement = best_variance_improvements(
        cumsums)
    best_data_row = orders[best_row, best_column]
    threshold = x[best_data_row, best_column]
    left_rows = orders[:best_row+1, best_column]
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


def build_regression_tree(X, y, max_depth=2, max_feature=100, min_improvement=1e-7, min_sample_leaf=1):
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
