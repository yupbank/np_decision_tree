from functools import partial

import numpy as np
from decision_tree.tree import DecisionTree
from decision_tree.base import Mask, Task, BestSplit, is_leaf
from decision_tree.utils import timeit
from decision_tree.strategy import random_split, greedy_split, random_split_v2


def init_mask(n_row, n_col):
    return Mask(np.ones(n_row, dtype=np.bool),
                np.ones(n_col, dtype=np.bool))


def leaf_from_data(y):
    return np.mean(y)


def split_mask(mask, Xf, best_split):
    column_mask = np.copy(mask.column)
    column_mask[best_split.constant_attrs] = 0
    row_indexes = np.flatnonzero(mask.row)
    left_row_mask = np.copy(mask.row)
    right_row_mask = np.copy(mask.row)

    left_row_mask[row_indexes[Xf[:, best_split.attribute]
                              > best_split.threshold]] = 0
    right_row_mask[row_indexes[Xf[:, best_split.attribute]
                               <= best_split.threshold]] = 0
    return (
        Mask(left_row_mask,
             column_mask),
        Mask(right_row_mask,
             column_mask)
    )


def find_best_split(X, y, candidate_attributes, max_feature=100, split_method=random_split):
    if candidate_attributes.size <= max_feature:
        attributes = candidate_attributes
    else:
        attributes = np.random.permutation(candidate_attributes)[:max_feature]
    Xfs = X[:, attributes]
    best_index, best_threshold, best_improvement, constant_attr_mask = split_method(
        Xfs, y)
    new_constant_attrs = attributes[constant_attr_mask]
    return BestSplit(attributes[best_index], best_threshold, new_constant_attrs, best_improvement)


@timeit
def build_tree(X, y,
               max_depth=2,
               max_feature=100,
               min_improvement=0.01,
               min_sample_leaf=1,
               split_method=greedy_split):
    best_split_method = partial(find_best_split, split_method=split_method)
    max_node = 2 ** (max_depth + 1)
    tree = DecisionTree(max_node)
    mask = init_mask(*X.shape)
    tasks = [Task(mask, parent=None, is_left=True, depth=0)]

    while tasks:
        task = tasks.pop()
        Xf, yf = X[task.mask.row], y[task.mask.row]
        node_id = tree.new_node_from_task(task)

        if is_leaf(task.mask, yf, min_sample_leaf) or task.depth >= max_depth:
            tree.add_leaf(node_id, leaf_from_data(yf))
        else:
            best_split = best_split_method(Xf,
                                           yf,
                                           candidate_attributes=np.flatnonzero(
                                               task.mask.column),
                                           max_feature=max_feature)
            if best_split.improvement < min_improvement:
                tree.add_leaf(node_id, leaf_from_data(yf))
            else:
                tree.add_binary(node_id, best_split)

                left_mask, right_mask = split_mask(task.mask,
                                                   Xf,
                                                   best_split)
                tasks.append(
                    Task(right_mask, parent=node_id, is_left=False, depth=task.depth + 1))
                tasks.append(
                    Task(left_mask, parent=node_id, is_left=True, depth=task.depth + 1))

    return tree.final()
