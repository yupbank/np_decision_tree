from functools import partial

import numpy as np
from sklearn.datasets import make_regression

from decision_tree.base import BestSplit


def variance(y):
    return np.var(y)


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
    left_masks = X <= thresholds

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
    total_sum, left_sums = left_sums[-1], left_sums[1:-1]
    total_count, left_counts = cum_count[-1], cum_count[1:-1]

    surrogate = np.square(left_sums)/left_counts + \
        np.square(total_sum-left_sums)/(total_count-left_counts)
    if surrogate.size == 0:
        return -1.0, -1.0
    index = np.argmax(surrogate)
    threshold = thresholds[index+1]

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
    return BestSplit(attributes[best_index], best_threshold, new_constant_attrs, best_improvement)


def find_best_split(X, y, candidate_attributes,  max_feature=100, min_improvement=0.01):
    if candidate_attributes.size <= max_feature:
        attributes = candidate_attributes
    else:
        attributes = np.random.permutation(candidate_attributes)[:max_feature]

    Xfs = X[:, attributes]
    best_index, threshold, score, constant_attr_mask = random_split(Xfs, y)
    new_constant_attrs = attributes[constant_attr_mask]
    return BestSplit(attributes[best_index],
                     threshold,
                     new_constant_attrs,
                     score)
