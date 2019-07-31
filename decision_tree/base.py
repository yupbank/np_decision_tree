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
