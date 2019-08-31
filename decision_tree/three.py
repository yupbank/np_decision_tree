import numpy as np
from itertools import chain


def np_gene(n): return np.reciprocal(
    np.arange(1, n, dtype=np.float32)*np.arange(n-1, 0, -1))


def best_variance_improvements(ys, start, end):
    size = end-start
    cumsums = np.cumsum(ys[start:end], axis=0)
    residual = np.arange(
        1, size+1)*(cumsums[-1][-1]/cumsums.shape[0])
    tmp = np.abs(cumsums-residual[:, np.newaxis])
    tmp_order = np.argmax(tmp, axis=1)
    diff = np.square(tmp[np.arange(size), tmp_order])
    res = np_gene(size)*diff[:-1]
    max_row = np.argmax(res)
    return max_row, tmp_order[max_row], res[max_row]


def batch_diff_and_feature(ys, sizes):
    cumsums = np.cumsum(ys, axis=0)
    sizes = np.array(sizes)
    index = np.cumsum(sizes)-1
    sums = cumsums[:, 0][index]
    cumsums = cumsums - \
        np.repeat(list(chain([0], sums))[:-1], sizes)[:, np.newaxis]
    sums = cumsums[:, 0][index]
    means = sums/sizes
    left_bias = np.concatenate(
        [np.arange(1, n+1)*mean for mean, n in zip(means, sizes)])
    abs_diff = np.abs(cumsums - left_bias[:, np.newaxis])
    max_features = np.argmax(abs_diff, axis=1)
    diff = np.square(np.max(abs_diff, axis=1))
    return diff, max_features


def best_row_and_feature(diff, max_features, start, end):
    size = end-start
    weight = np_gene(size)
    impro_in_range = weight*diff[start:end-1]
    max_row = np.argmax(impro_in_range)
    improvement = impro_in_range[max_row]
    best_feature = max_features[start + max_row]
    return max_row, best_feature, improvement


def size_to_index(sizes):
    indexes = np.cumsum(sizes)
    return zip(chain([0], indexes), indexes)


def split_order_with_mask(mask, order):
    num_rank, num_feature = order.shape
    assert mask.shape[0] >= num_rank
    return order.T[(mask[order]).T].reshape(num_feature, -1).T


class DecisionTree:
    def __init__(self, max_node):
        from collections import namedtuple
        Tree = namedtuple(
            'DecisionTree', 'children_left children_right feature threshold value n_node_samples')
        children_left, children_right = np.empty(
            max_node-1, dtype=np.int64), np.empty(max_node-1, dtype=np.int64)
        feature = np.empty(max_node-1, dtype=np.int64)
        value = np.empty(max_node-1)
        threshold = np.empty(max_node-1)
        n_node_samples = np.empty(max_node-1, dtype=np.int32)

        children_left[:] = -1
        children_right[:] = -1
        n_node_samples[:] = -2
        feature[:] = -2
        threshold[:] = -2

        self.tree_ = Tree(children_left, children_right,
                          feature, threshold, value, n_node_samples)
        self.max_node_ = -1

    def add_node(self, parent=None, is_left=None):
        self.max_node_ += 1
        if parent is not None:
            to_save = self.tree_.children_left if is_left else self.tree_.children_right
            to_save[parent] = self.max_node_
        return self.max_node_

    def add_size(self, node_id, size):
        self.tree_.n_node_samples[node_id] = size

    def add_leaf(self, node_id, value):
        self.tree_.value[node_id] = value

    def add_binary(self, node_id, col, threshold):
        self.tree_.threshold[node_id] = threshold
        self.tree_.feature[node_id] = col


def build_regression_tree(X, y, max_depth=2, min_improvement=1e-7, min_sample_leaf=1):
    tree = DecisionTree(2**(max_depth+1))
    n_samples = X.shape[0]
    orders = np.argsort(X, axis=0)

    left_mask, right_mask = np.zeros_like(
        y, dtype=np.bool), np.zeros_like(y, dtype=np.bool)

    parents, sizes, left_elements = [None], [X.shape[0]], 0
    for depth in range(max_depth+1):
        ys = y[orders]
        left_sizes, right_sizes = [], []
        left_parents, right_parents = [], []
        left_mask[:], right_mask[:] = 0, 0

        if depth < max_depth:
            diff, max_features = batch_diff_and_feature(ys, sizes)

        for n, (parent, (start, end)) in enumerate(zip(parents, size_to_index(sizes))):
            size = end-start

            node_id = tree.add_node(parent, n < left_elements)
            tree.add_size(node_id, size)

            if size <= min_sample_leaf or depth >= max_depth:
                tree.add_leaf(node_id, np.mean(ys[start:end, 0]))
                continue
            else:
                max_row, best_feature, improvement = best_row_and_feature(
                    diff, max_features, start, end)
                if improvement <= min_improvement:
                    tree.add_leaf(node_id, np.mean(ys[start:end, 0]))
                    continue
                else:
                    tree.add_binary(
                        node_id, best_feature, X[orders[start+max_row, best_feature], best_feature])
                    left_parents.append(node_id)
                    right_parents.append(node_id)
                    left_sizes.append(max_row+1)
                    right_sizes.append(size-max_row-1)
                    left_mask[orders[:, best_feature]
                              [start:start+max_row+1]] = 1
                    right_mask[orders[:, best_feature]
                               [start+max_row+1:end]] = 1

        sizes = left_sizes+right_sizes
        parents = left_parents+right_parents
        left_elements = len(left_sizes)
        left_orders = split_order_with_mask(left_mask, orders)
        right_orders = split_order_with_mask(right_mask, orders)
        orders = np.concatenate([left_orders, right_orders], axis=0)

    return tree


if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.metrics import regression
    x, y = make_regression(n_samples=10000, n_features=200, random_state=2)
    t = build_regression_tree(x, y, 5)
    print(t)
