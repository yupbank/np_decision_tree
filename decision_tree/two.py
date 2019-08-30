import numpy as np


def np_gene(n): return np.reciprocal(
    np.arange(1, n, dtype=np.float32)*np.arange(n-1, 0, -1))


def best_variance_improvements(cumsums):
    residual = np.arange(
        1, cumsums.shape[0]+1)*(cumsums[-1][-1]/cumsums.shape[0])
    tmp = np.abs(cumsums-residual[:, np.newaxis])
    tmp_order = np.argmax(tmp, axis=1)
    value = tmp[np.arange(cumsums.shape[0]), tmp_order]
    res = np_gene(cumsums.shape[0])*np.square(value)[:-1]
    max_row = np.argmax(res)
    return max_row, tmp_order[max_row], res[max_row]


def size_to_index(sizes):
    from itertools import chain
    indexes = np.cumsum(sizes)
    return zip(chain([0], indexes), indexes)


def split_order_with_mask(mask, order):
    num_rank, num_feature = order.shape
    assert mask.shape[0] >= num_rank
    return order.T[(mask[order]).T].reshape(num_feature, -1).T


def build_regression_tree(X, y, max_depth=2, min_improvement=1e-7, min_sample_leaf=1):
    n_samples = X.shape[0]
    orders = np.argsort(X, axis=0)
    parents, sizes = [0], [X.shape[0]]
    left_mask, right_mask = np.zeros_like(
        y, dtype=np.bool), np.zeros_like(y, dtype=np.bool)
    ys = y[orders]
    cumsums = np.cumsum(ys, axis=0)
    max_row, max_col, improvement = best_variance_improvements(cumsums)
    data_row = orders[max_row, max_col]
    (data_row, max_col)
    left_sizes, right_sizes = [max_row], [n_samples-max_row]
    left_elements = len(left_sizes)
    left_mask[orders[:max_row+1, max_col]] = 1
    right_mask[orders[max_row+1:, max_col]] = 1
    sizes = left_sizes+right_sizes
    for depth in range(1, max_depth):
        left_orders = split_order_with_mask(left_mask, orders)
        right_orders = split_order_with_mask(right_mask, orders)
        orders = np.concatenate([left_orders, right_orders], axis=0)
        ys = y[orders]
        start_position = 2**depth

        if depth == max_depth-1:
            for n, (parent, (start, end)) in enumerate(zip(parents, size_to_index(sizes))):
                if n < left_elements:
                    node_id = start_position + parent
                else:
                    node_id = start_position + parent + 1
                (node_id, np.mean(ys[start:end, 0]), "leaf", sizes)

        cumsums = np.cumsum(ys, axis=0)
        sizes = np.array(sizes)
        sums = cumsums[:, 0][sizes-1]
        means = sums/sizes
        left_bias = np.concatenate(
            [np.arange(1, n+1)*mean for mean, n in zip(means, sizes)])

        tmp = np.abs(cumsums - left_bias[:, np.newaxis])
        max_features = np.argmax(tmp, axis=1)
        diff = np.square(tmp[np.arange(max_features.size), max_features])
        left_sizes, right_sizes = [], []
        left_parents, right_parents = [], []
        left_mask[:], right_mask[:] = 0, 0
        for n, (parent, (start, end)) in enumerate(zip(parents, size_to_index(sizes))):
            size = end-start
            if n < left_elements:
                node_id = start_position + parent
            else:
                node_id = start_position+parent+1

            if size <= min_sample_leaf:
                (parent, node_id, np.mean(ys[start:end, 0]))
                continue
            else:
                weight = np.reciprocal(
                    np.arange(1, size, dtype=np.float32)*np.arange(size-1, 0, -1))
                impro_in_range = weight*diff[start:end-1]
                max_row = np.argmax(impro_in_range)
                improvement = impro_in_range[max_row]
                if improvement <= min_improvement:
                    (parent, node_id, np.mean(ys[start:end, 0]))
                    continue
                else:
                    best_feature = max_features[start + max_row]
                    (parent, node_id, start+max_row, best_feature)

                left_parents.append(node_id)
                right_parents.append(node_id)
                left_sizes.append(max_row+1)
                right_sizes.append(size-max_row-1)
            left_mask[orders[:, best_feature][start:start+max_row+1]] = 1
            right_mask[orders[:, best_feature][start+max_row+1:end]] = 1
        sizes = left_sizes+right_sizes
        (left_parents, right_parents, "!!!")
        parents = left_parents+right_parents
        left_elements = len(left_sizes)

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    x, y = make_regression(n_samples=10000, n_features=200, random_state=2)
    build_regression_tree(x, y, 5)