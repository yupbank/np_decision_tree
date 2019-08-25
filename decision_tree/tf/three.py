import numpy as np
from itertools import chain


def np_gene(n): return np.reciprocal(
    np.arange(1, n, dtype=np.float32)*np.arange(n-1, 0, -1))


def best_variance_improvements(cumsum):
    residual = np.arange(1, cumsum.shape[0]+1)*(cumsum[-1][-1]/cumsum.shape[0])
    tmp = np.abs(cumsum-residual[:, np.newaxis])
    tmp_order = np.argmax(tmp, axis=1)
    value = tmp[np.arange(cumsum.shape[0]), tmp_order]
    res = np_gene(cumsum.shape[0])*np.square(value)[:-1]
    max_col = np.argmax(res)
    return max_col, tmp_order[max_col], res[max_col]


"""
cumsums :  
          F_1, F_2, F_3,...F_M
Left_sum_1
Left_sum_2
Left_sum_3
...
Left_sum_N
"""


def batch_variance_improvements(cumsums, sizes, orders, left_mask, right_mask):
    #print(len(sizes))
    num_instances, num_features = cumsums.shape
    sums = cumsums[:, 0][np.array(sizes)-1]
    left_means = np.concatenate(
        [np.arange(1, n+1)*(sum_/n) for sum_, n in zip(sums, sizes)])
    tmp = np.abs(cumsums - left_means[:, np.newaxis])
    max_features = np.argmax(tmp, axis=1)
    diff = np.square(tmp[np.arange(num_instances), max_features])
    left_sizes, right_sizes, cuting_points = [], [], []
    start = 0
    lleft_rows, lright_rows = [], []
    for size in sizes:
        end = start+size
        if size <= 2:
            start = end
            continue
        weight = np.reciprocal(
            np.arange(1, size, dtype=np.float32)*np.arange(size-1, 0, -1))
        impro_in_range = weight*diff[start:end-1]
        max_col = np.argmax(impro_in_range)
        cutting_rank = start+max_col
        best_feature = max_features[cutting_rank]
        improvement = impro_in_range[max_col]
        left_size = max_col+1
        right_size = size-max_col-1

        left_rows, right_rows = np.arange(
            start, start+left_size), np.arange(end-right_size, end)
        lleft_rows.extend(orders[:, best_feature][left_rows])
        lright_rows.extend(orders[:, best_feature][right_rows])
        left_sizes.append(left_size)
        right_sizes.append(right_size)
        cuting_points.append((cutting_rank, best_feature, improvement))
        start = end
    left_mask[lleft_rows] = 1
    right_mask[lright_rows] = 1
    return cuting_points, left_sizes+right_sizes, left_mask, right_mask


def mask_to_order(mask, order):
    num_rank, num_feature = order.shape
    assert mask.shape[0] >= num_rank
    return order.T[(mask[order]).T].reshape(num_feature, -1).T


def build_regression_tree(X, y, max_depth=2, max_feature=100, min_improvement=1e-7, min_sample_leaf=1):
    min_size = min_sample_leaf*2

    orders = np.argsort(X, axis=0)
    sizes = np.array([X.shape[0]])
    for i in range(max_depth):
        left_mask = np.zeros_like(y, dtype=np.bool)
        right_mask = np.zeros_like(y, dtype=np.bool)
        ys = np.cumsum(y[orders], axis=0)
        cuting_points, sizes, left_mask, right_mask = batch_variance_improvements(
            ys, sizes, orders, left_mask, right_mask)
        left_orders = mask_to_order(left_mask, orders)
        right_orders = mask_to_order(right_mask, orders)
        orders = np.concatenate([left_orders, right_orders], axis=0)
