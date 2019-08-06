from functools import partial

import numpy as np
from decision_tree.base import Mask, Task, BestSplit, is_leaf, DecisionTree
from decision_tree.utils import timeit
from decision_tree.strategy import random_split, greedy_split, random_split_v2, random_classify
from decision_tree.strategy import greedy_classification_v3, split_orders_and_cumsums_v2
from decision_tree.base import *


def init_mask(n_row, n_col):
    return Mask(np.ones(n_row, dtype=np.bool),
                np.ones(n_col, dtype=np.bool))


def leaf_from_data_regression(y):
    return np.mean(y)


def leaf_from_data_classification(y):
    return np.argmax(y.sum(axis=0))


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


def build_tree(X, y,
               max_depth=2,
               max_feature=100,
               min_improvement=0.01,
               min_sample_leaf=1,
               split_method=greedy_split,
               leaf_from_data=leaf_from_data_regression):
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


def build_tree_v2(X, y,
                  max_depth=2,
                  max_feature=100,
                  min_improvement=0.01,
                  min_sample_leaf=1,
                  max_classes=4,
                  leaf_from_data=leaf_from_data_regression):
    max_node = 2 ** (max_depth + 1)
    tree = DecisionTree(max_node)

    encoding = np.eye(max_classes)
    y = encoding[y]

    orders = np.argsort(X, axis=0)
    cumsums = np.cumsum(y[orders], axis=0)

    tasks = [Task((orders, cumsums), parent=None, is_left=True, depth=0)]

    while tasks:
        task = tasks.pop()
        node_id = tree.new_node_from_task(task)
        if is_leaf_v2(task.mask, min_sample_leaf) or task.depth >= max_depth:
            tree.add_leaf(node_id, leaf_from_data(y[task.mask[0]]))
        else:
            best_order_row, best_column, best_data_row, best_improvement = greedy_classification_v3(*task.mask, y)
            if best_improvement < min_improvement:
                tree.add_leaf(node_id, leaf_from_data(y[task.mask[0]]))
            else:
                tree.add_binary_v2(node_id, best_column,
                                   X[best_data_row, best_column])

                left_orders, left_cumsums, right_orders, right_cumsums = split_orders_and_cumsums_v2(*task.mask, best_column, best_order_row,  y)
                tasks.append(
                    Task((right_orders, right_cumsums), parent=node_id, is_left=False, depth=task.depth + 1))
                tasks.append(
                    Task((left_orders, left_cumsums), parent=node_id, is_left=True, depth=task.depth + 1))

    return tree.final()


def build_regression_tree(X, y,
                          max_depth=2,
                          max_feature=100,
                          min_improvement=0.01,
                          min_sample_leaf=1,
                          split_method=random_split,
                          leaf_from_data=leaf_from_data_regression):
    return build_tree(X, y, max_depth, max_feature, min_improvement, min_sample_leaf, split_method, leaf_from_data)


def build_classification_tree(X, y,
                              max_depth=2,
                              max_feature=100,
                              min_improvement=0.01,
                              min_sample_leaf=1,
                              split_method=random_classify,
                              leaf_from_data=leaf_from_data_classification,
                              max_classes=2):
    encoding = np.eye(max_classes)
    ye = encoding[y]
    return build_tree(X, ye, max_depth, max_feature, min_improvement, min_sample_leaf, split_method, leaf_from_data)
