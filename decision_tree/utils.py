import time
import numpy as np


def timeit(func):
    def _(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time() - start
        print('Func: %s, runtime: %.6f' % (func.__name__, end))
        return res
    return _


def inference(data, clf):
    feature, threshold, left, right, value = clf.tree_.feature, clf.tree_.threshold, clf.tree_.children_left, clf.tree_.children_right, clf.tree_.value
    auxilary = np.arange(data.shape[0], dtype=np.int32)
    prev_node = np.zeros(1, dtype=np.int32)
    while 1:
        condition = data[auxilary, feature[prev_node]] <= threshold[prev_node]
        potential_next_node = np.where(
            condition, left[prev_node], right[prev_node])
        potential_condition = potential_next_node != -1
        if not np.any(potential_condition):
            break
        next_node = np.where(potential_condition,
                             potential_next_node, prev_node)
        prev_node = next_node
    return value[prev_node].ravel()


def inference_leaf(data, clf):
    feature, threshold, left, right, value = clf.tree_.feature, clf.tree_.threshold, clf.tree_.children_left, clf.tree_.children_right, clf.tree_.value
    auxilary = np.arange(data.shape[0], dtype=np.int32)
    prev_node = np.zeros(1, dtype=np.int32)
    while 1:
        condition = data[auxilary, feature[prev_node]] <= threshold[prev_node]
        potential_next_node = np.where(
            condition, left[prev_node], right[prev_node])
        potential_condition = potential_next_node != -1
        if not np.any(potential_condition):
            break
        next_node = np.where(potential_condition,
                             potential_next_node, prev_node)
        prev_node = next_node
    return prev_node
