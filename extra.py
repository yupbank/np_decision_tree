from functools import partial
import numpy as np
from itertools import groupby
from collections import namedtuple
from sklearn.datasets import make_regression

import time


def timeit(func):
    def _(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time() - start
        print('Func: %s, runtime: %.6f' % (func.__name__, end))
        return res
    return _


Mask = namedtuple('DataMask', 'row column')
Task = namedtuple('Task', 'mask parent is_left depth')
BestSplit = namedtuple(
        'BestSplit', 'mask attribute threshold constant_attrs')
Tree = namedtuple('Tree', 'child_left child_right features thresholds value')


def variance(y):
    return np.var(y)


def is_leaf(mask, y, min_sample_leaf=10):
    if np.sum(mask.row) < min_sample_leaf:
        return True
    if np.unique(y[mask.row]).size <= 1:
        return True
    if np.sum(mask.column) == 0:
        return True
    return False


def score_fn(y, masks, method=variance):
    origin_score = method(y)
    sample_size, split_size = masks.shape
    sample_size = float(sample_size)
    current_weight = sample_size
    scores = np.zeros(split_size)
    left_weights = np.sum(masks, axis=0)/sample_size
    for n, split_mask in enumerate(masks.T):
        left_score = method(y[split_mask])
        right_score = method(y[~split_mask])
        left_weight = left_weights[n]
        right_weight = 1 - left_weight
        scores[n] = (origin_score
                     - left_weight * left_score
                     - right_weight * right_score
                     )
    return scores*current_weight


def random_split(X, y):
    indexes = np.argsort(X, axis=0)
    min_max_index = indexes[[0, -1]]
    auxilary_col, _ = np.meshgrid(np.arange(X.shape[1]), [0, 1])
    min_max_value = X[min_max_index, auxilary_col]
    constant_attr_mask = min_max_value[0] == min_max_value[1]
    min_max_value = min_max_value[:, ~constant_attr_mask]
    thresholds = np.random.uniform(min_max_value[0], min_max_value[1])
    left_masks = X < thresholds

    scores = score_fn(y, left_masks)

    best_index = np.argmax(scores)
    return best_index, thresholds[best_index], scores[best_index], constant_attr_mask


def _compelete_split(Xf, y):
    thresholds, inverse,  count = np.unique(
        Xf, return_counts=True, return_inverse=True)

    if thresholds.size <= 1:
        return -1.0, -1.0

    y_sum = np.zeros(count.size)
    np.add.at(y_sum, inverse, y)

    cum_count = np.cumsum(count)

    left_sums = np.cumsum(y_sum)
    total_sum, left_sums = left_sums[-1], left_sums[:-1]
    total_count, left_counts = cum_count[-1], cum_count[:-1]

    surrogate = np.square(left_sums)/left_counts + \
        np.square(total_sum-left_sums)/(total_count-left_counts)

    index = np.argmax(surrogate)
    threshold = thresholds[index]

    improvement = (surrogate[index] -
                   np.square(total_sum)/total_count)/total_count

    return improvement, threshold


def compelet_split(X, y):
    split_column = partial(_compelete_split, y=y)
    improvements = np.apply_along_axis(split_column, 0, X)
    best_index = np.argmax(improvements[0])
    best_improvement, best_threshold = improvements[:, best_index]
    constant_attr_mask = np.logical_and(
        improvements[0] == -1.0, improvements[1] == -1.0)
    return best_index, best_threshold, best_improvement, constant_attr_mask


def find_best_split_v2(X, y, candidate_attributes, max_feature=100, min_improvement=0.01):
    if candidate_attributes.size <= max_feature:
        attributes = candidate_attributes
    else:
        attributes = np.random.permutation(candidate_attributes)[:max_feature]
    Xfs = X[:, attributes]
    best_index, best_threshold, best_improvement, constant_attr_mask = compelet_split(
        Xfs, y)
    new_constant_attrs = attributes[constant_attr_mask]
    if best_improvement < min_improvement:
        return None
    else:
        return BestSplit(Xfs[:, best_index] < best_threshold, attributes[best_index], best_threshold, new_constant_attrs)


def find_best_split(X, y, candidate_attributes,  max_feature=100, min_improvement=0.01):
    if candidate_attributes.size <= max_feature:
        attributes = candidate_attributes
    else:
        attributes = np.random.permutation(candidate_attributes)[:max_feature]

    Xfs = X[:, attributes]
    best_index, threshold, score, constant_attr_mask = random_split(Xfs, y)
    new_constant_attrs = attributes[constant_attr_mask]
    if score < min_improvement:
        return None
    else:
        return BestSplit(Xfs[:, best_index] < threshold,
                         attributes[best_index],
                         threshold,
                         new_constant_attrs)


def split_mask(mask, best_split):
    column_mask = np.copy(mask.column)
    column_mask[best_split.constant_attrs] = 0
    row_indexes = np.flatnonzero(mask.row)
    left_row_mask = np.copy(mask.row)
    right_row_mask = np.copy(mask.row)
    left_row_mask[row_indexes[~best_split.mask]] = 0
    right_row_mask[row_indexes[best_split.mask]] = 0
    return (
            Mask(left_row_mask,
                 column_mask),
            Mask(right_row_mask,
                 column_mask)
            )


def leaf_from_data(y):
    return np.mean(y)


def split_to_node(best_split):
    return best_split.attribute, best_split.threshold


@timeit
def build_an_extra_tree(X, y, max_depth=2, max_feature=100, min_improvement=0.01, min_sample_leaf=1):
    child_left = np.empty(2**max_depth, dtype=np.int32)
    child_right = np.empty(2**max_depth, dtype=np.int32)
    value = np.empty(2**max_depth)
    features = np.empty(2**max_depth, dtype=np.int32)
    thresholds = np.empty(2**max_depth)
    value[:] = -1
    child_left[:] = -1
    child_right[:] = -1
    features[:] = -1
    thresholds[:] = -1

    node_id = -1
    initial_mask = Mask(np.ones(X.shape[0], dtype=np.bool),
                        np.ones(X.shape[1], dtype=np.bool))
    tasks = [Task(initial_mask, parent=node_id, is_left=None, depth=0)]

    while tasks:
        node_id += 1
        task = tasks.pop()
        Xf, yf = X[task.mask.row], y[task.mask.row]
        if task.parent >= 0:
            to_save = child_left if task.is_left else child_right
            to_save[task.parent] = node_id

        if is_leaf(task.mask, y, min_sample_leaf) or task.depth >= max_depth-1:
            value[node_id] = leaf_from_data(yf)
        else:
            best_split = find_best_split_v2(Xf,
                                            yf,
                                            candidate_attributes=np.flatnonzero(
                                             task.mask.column),
                                            max_feature=max_feature,
                                            min_improvement=min_improvement)
            if best_split is None:
                print(yf.shape, yf[best_split.mask].size)
                value[node_id] = leaf_from_data(yf)
            else:
                thresholds[node_id] = best_split.threshold
                features[node_id] = best_split.attribute

                left_mask, right_mask = split_mask(task.mask,
                                                   best_split)

                tasks.append(
                    Task(left_mask, parent=node_id, is_left=True, depth=task.depth+1))
                tasks.append(
                    Task(right_mask, parent=node_id, is_left=False, depth=task.depth+1))

    return Tree(child_left[:node_id+1],
                child_right[:node_id+1],
                features[:node_id+1],
                thresholds[:node_id+1],
                value[:node_id+1])


def inference(data, clf):
    feature, threshold, left, right, value = clf.features, clf.thresholds, clf.child_left, clf.child_right, clf.value
    auxilary = np.arange(data.shape[0], dtype=np.int32)
    prev_node = np.zeros(1, dtype=np.int32)
    while 1:
        condition = data[auxilary, feature[prev_node]] < threshold[prev_node]
        potential_next_node = np.where(
            condition, left[prev_node], right[prev_node])
        potential_condition = potential_next_node != -1
        if not np.any(potential_condition):
            break
        next_node = np.where(potential_condition,
                             potential_next_node, prev_node)
        prev_node = next_node
    return value[prev_node]


@timeit
def main():
    from sklearn.metrics import regression
    from sklearn.tree import ExtraTreeRegressor
    np.random.seed(20)
    x, y = make_regression(n_samples=1000, n_informative=10)
    t = build_an_extra_tree(x, y, max_depth=10,
                            max_feature=9, min_improvement=0.000001)
    y_hat = inference(x, t)
    #print(t)
    #print("truth", y)
    #print("we predicted", y_hat)
    print("my result", regression.mean_squared_error(y_hat, y))

    @timeit
    def fit(x, y):
        clf = ExtraTreeRegressor(
            max_depth=10, max_features=10, min_impurity_decrease=0.000001)
        clf.fit(x, y)
        return clf
    clf = fit(x, y)
    #print ("sklearn predicted", clf.predict(x))
    print("sklearn result", regression.mean_squared_error(clf.predict(x), y))


if __name__ == "__main__":
    main()
