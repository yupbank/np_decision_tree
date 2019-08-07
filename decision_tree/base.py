from collections import namedtuple
import numpy as np


Mask = namedtuple('DataMask', 'row column')
Task = namedtuple('Task', 'mask parent is_left depth')
BestSplit = namedtuple(
    'BestSplit', 'attribute threshold constant_attrs improvement')
Tree = namedtuple(
    'DecisionTree', 'children_left children_right feature threshold value')


def is_leaf(mask, y, min_sample_per_leaf=10):
    if np.sum(mask.row) <= min_sample_per_leaf*2:
        return True
    elif np.unique(y).size <= 1:
        return True
    elif np.sum(mask.column) == 0:
        return True
    else:
        return False


def is_leaf_v2(orders, min_sample_per_leaf=10):
    row, column = orders.shape
    if row <= min_sample_per_leaf*2:
        return True
    else:
        return False


class DecisionTree:
    def __init__(self, max_node):
        children_left, children_right = np.empty(
            max_node-1, dtype=np.int64), np.empty(max_node-1, dtype=np.int64)
        feature = np.empty(max_node-1, dtype=np.int64)
        value = np.empty(max_node-1)
        threshold = np.empty(max_node-1)

        #value[:] = -2
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

    def add_binary(self, node_id, best_split):
        self.tree_.threshold[node_id] = best_split.threshold
        self.tree_.feature[node_id] = best_split.attribute

    def add_binary_v2(self, node_id, col, threshold):
        self.tree_.threshold[node_id] = threshold
        self.tree_.feature[node_id] = col

    def final(self):
        return self
