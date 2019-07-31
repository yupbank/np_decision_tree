import numpy as np
from base import Tree


class DecisionTree:
    def __init__(self, max_node):
        children_left, children_right = np.empty(
            max_node, dtype=np.int32), np.empty(max_node, dtype=np.int32)
        feature = np.empty(max_node, dtype=np.int32)
        value = np.empty(max_node)
        threshold = np.empty(max_node)

        value[:] = -1
        children_left[:] = -1
        children_right[:] = -1
        feature[:] = -1
        threshold[:] = -1

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

    def final(self):
        return self
