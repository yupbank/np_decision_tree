import tensorflow as tf
import numpy as np
import collections
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

Tree = collections.namedtuple(
    'DecisionTree', 'children_left,children_right,feature,threshold,value')


def init_tree(max_depth):
    max_node = 2 ** (max_depth + 1)
    children_left, children_right = np.zeros(
        max_node - 1, dtype=np.int32), np.zeros(max_node - 1, dtype=np.int32)
    feature = np.zeros(max_node - 1, dtype=np.int32)
    value = np.zeros(max_node - 1)
    threshold = np.zeros(max_node - 1)

    children_left[:] = -1
    children_right[:] = -1
    feature[:] = -2
    threshold[:] = -2
    return Tree(
        tf.Variable(children_left),
        tf.Variable(children_right),
        tf.Variable(feature),
        tf.Variable(threshold),
        tf.Variable(value),
    )


def argmax_2d(tensor):
    tensor_shape = tf.shape(tensor)
    assert len(tensor.get_shape()) == 2
    op = tf.assert_greater(tf.shape(tensor)[0], 0)
    with tf.control_dependencies([op]):
        flat_tensor = tf.reshape(tensor, [-1])
        argmax = tf.argmax(flat_tensor, output_type=tf.int32)
        argmax_x = argmax // tensor_shape[1]
        argmax_y = argmax % tensor_shape[1]
        return argmax_x, argmax_y


def normalize(ys):
    return ys / tf.reduce_sum(ys, axis=0, keepdims=True)


def best_variance_improvements(ys, weight=None):
    def calculate():
        new_weight = weight or tf.ones_like(ys[:, 0])
        x = tf.cumsum(normalize(ys), axis=0)[:-1, :]
        new_weight = tf.cumsum(normalize(new_weight), axis=0)[:-1]
        y = x ** 2 / new_weight[:, tf.newaxis] + \
            (1 - x) ** 2 / (1 - new_weight[:, tf.newaxis])
        max_row, max_col = argmax_2d(y)
        max_value = y[max_row, max_col]
        return max_row, max_col, max_value

    return tf.cond(tf.less_equal(tf.shape(ys)[0], 1),
                   lambda: (0, 0, tf.constant(0.0, dtype=tf.float64)),
                   calculate)


def populate_mask(prev_mask, new_indices):
    return tf.tensor_scatter_nd_add(
        prev_mask, new_indices,
        tf.ones_like(new_indices[:, 0], dtype=prev_mask.dtype))


def slice_order_by_mask(orders, mask):
    return tf.transpose(tf.reshape(tf.boolean_mask(
        tf.transpose(orders),
        tf.gather(mask > 0, tf.transpose(orders))),
        (orders.shape[1], -1)))


class DecisionTree(object):
    def __init__(self, max_depth=10, min_sample_leaf=1, min_improvement=1e-7):
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        self.min_improvement = min_improvement
        self.tree_ = init_tree(max_depth)

    def new_left_node(self, node_id, parent):
        ops = self.tree_.children_left[parent].assign(node_id+1)
        with tf.control_dependencies([ops]):
            return node_id+1

    def new_right_node(self, node_id, parent):
        ops = self.tree_.children_right[parent].assign(node_id+1)
        with tf.control_dependencies([ops]):
            return node_id+1

    def add_leaf(self, node_id, value):
        return self.tree_.value[node_id].assign(value)

    def add_binary_split(self, node_id, feature, threshold):
        add_feature_ops = self.tree_.feature[node_id].assign(feature)
        add_threshold_ops = self.tree_.threshold[node_id].assign(
            threshold)
        return tf.group(add_threshold_ops, add_feature_ops)

    @staticmethod
    def split_data(orders, start, size, best_row, best_col, left_sizes, right_sizes, left_mask, right_mask):
        new_left_mask = populate_mask(
            left_mask, orders[start:start + best_row + 1, best_col:best_col + 1])
        new_right_mask = populate_mask(
            right_mask, orders[start + best_row + 1:start + size, best_col:best_col + 1])
        new_left_sizes = tf.concat([left_sizes, [best_row + 1]], axis=0)
        new_right_sizes = tf.concat(
            [right_sizes, [size - best_row - 1]], axis=0)
        return new_left_sizes, new_right_sizes, new_left_mask, new_right_mask

    def try_split(self, x, ys, depth, orders, node_id, start, size,
                  left_sizes, right_sizes, left_mask, right_mask,
                  new_parents):
        best_row, best_col, improvement = best_variance_improvements(
            ys[start:start + size])

        cond = tf.logical_or(
            tf.logical_or(
                tf.less_equal(improvement, self.min_improvement),
                tf.less_equal(size, self.min_sample_leaf)),
            tf.less_equal(self.max_depth - 1, depth)
        )

        def new_leaf():
            ops = self.add_leaf(node_id, tf.reduce_mean(
                ys[start:start + size, 0]))
            with tf.control_dependencies([ops]):
                return left_sizes, right_sizes, left_mask, right_mask, new_parents, tf.cast(node_id, tf.int64)

        def new_split():
            new_left_sizes, new_right_sizes, new_left_mask, new_right_mask = self.split_data(
                orders, start, size, best_row, best_col, left_sizes, right_sizes, left_mask, right_mask)

            ops = self.add_binary_split(
                node_id, best_col, x[orders[start + best_row, best_col], best_col])

            with tf.control_dependencies([ops]):
                return new_left_sizes, new_right_sizes, new_left_mask, new_right_mask, \
                    tf.concat([new_parents, [node_id]],
                              axis=0), tf.cast(node_id, tf.int64)

        return tf.cond(
            cond,
            new_leaf,
            new_split,
        )

    def split_level(self, x, y, depth, node_id, parents, orders, sizes, new_node_func):
        initial_counter = 0
        initial_end = 0
        initial_new_parents = tf.constant([], dtype=tf.int32)
        initial_left_sizes = tf.constant([], dtype=tf.int32)
        initial_right_sizes = tf.constant([], dtype=tf.int32)
        initial_left_mask = tf.zeros_like(y, dtype=tf.int32)
        initial_right_mask = tf.zeros_like(y, dtype=tf.int32)

        ys = tf.gather(y, orders)

        def condition(counter, *_):
            return tf.less_equal(counter, tf.shape(sizes)[0] - 1)

        def func(counter,
                 prev_end, prev_node_id, prev_new_parents,
                 prev_left_sizes, prev_right_sizes,
                 prev_left_mask, prev_right_mask):
            parent, size = parents[counter], sizes[counter]
            start, current_end = prev_end, prev_end + size
            current_node_id = new_node_func(prev_node_id, parent)
            (current_left_sizes, current_right_sizes,
             current_left_mask, current_right_mask,
             current_new_parents, current_node_id) = self.try_split(x, ys, depth, orders,
                                                                    current_node_id, start, size,
                                                                    prev_left_sizes, prev_right_sizes,
                                                                    prev_left_mask, prev_right_mask, prev_new_parents)
            return (
                counter + 1,
                current_end,
                tf.cast(current_node_id, tf.int32),
                current_new_parents,
                current_left_sizes,
                current_right_sizes,
                current_left_mask,
                current_right_mask
            )

        _counter, _end, new_node_id, new_parents, left_sizes, right_sizes, left_mask, right_mask = tf.while_loop(
            condition,
            func,
            (
                initial_counter,
                initial_end,
                node_id,
                initial_new_parents,
                initial_left_sizes,
                initial_right_sizes,
                initial_left_mask,
                initial_right_mask,
            ),
            (
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([None]),
                tf.TensorShape([None]),
                tf.TensorShape([None]),
                initial_left_mask.shape,
                initial_right_mask.shape,
            ),
            back_prop=False,
        )
        left_orders = slice_order_by_mask(orders, left_mask)
        right_orders = slice_order_by_mask(orders, right_mask)

        return (
            new_node_id,
            new_parents,
            left_sizes,
            right_sizes,
            left_orders,
            right_orders
        )

    def first_split(self, x, y):
        start, size = 0, tf.shape(x)[0]
        orders = tf.argsort(x, axis=0)
        left_mask = tf.zeros_like(y, dtype=tf.int32)
        right_mask = tf.zeros_like(y, dtype=tf.int32)
        left_sizes = tf.constant([], dtype=tf.int32)
        right_sizes = tf.constant([], dtype=tf.int32)
        ys = tf.gather(y, orders)
        best_row, best_col, improvement = best_variance_improvements(
            ys[start:start + size])
        new_left_sizes, new_right_sizes, new_left_mask, new_right_mask = self.split_data(
            orders, start, size, best_row, best_col, left_sizes, right_sizes, left_mask, right_mask)
        ops = self.add_binary_split(
            0, best_col, x[orders[start + best_row, best_col], best_col])
        with tf.control_dependencies([ops]):
            return tf.cond(improvement > self.min_improvement,
                           lambda: (new_left_sizes, new_right_sizes,
                                    slice_order_by_mask(orders, new_left_mask), slice_order_by_mask(orders, new_right_mask)),
                           lambda: (left_sizes, right_sizes,
                                    slice_order_by_mask(orders, left_mask), slice_order_by_mask(orders, right_mask))
                           )

    def fit(self, x, y):
        init_depth = tf.constant(0)
        init_node_id = tf.constant(0, dtype=tf.int32)
        init_parents = tf.constant([0])

        init_left_sizes, init_right_sizes, init_left_orders, init_right_orders = self.first_split(
            x, y)
        init_task = (
            init_depth,
            init_node_id,
            init_parents,
            init_left_orders,
            init_right_orders,
            init_left_sizes,
            init_right_sizes,
        )
        shape_invariants = (
            tf.TensorShape([]),
            tf.TensorShape([]),
            tf.TensorShape([None]),
            x.shape,
            x.shape,
            tf.TensorShape([None]),
            tf.TensorShape([None]),
        )

        def condition(depth, *_):
            return depth < self.max_depth+1

        def func(depth, node_id, parents, left_orders, right_orders, left_sizes, right_sizes):
            (node_id, left_parents,
             left_left_sizes, left_right_sizes,
             left_left_orders, left_right_orders) = self.split_level(
                x,
                y,
                depth,
                node_id,
                parents, left_orders, left_sizes, self.new_left_node)
            (node_id, right_parents,
             right_left_sizes, right_right_sizes,
             right_left_orders, right_right_orders) = self.split_level(
                x,
                y,
                depth,
                node_id,
                parents, right_orders, right_sizes, self.new_right_node)
            left_orders = tf.concat(
                [left_left_orders, right_left_orders], axis=0)
            left_sizes = tf.concat(
                [left_left_sizes, right_left_sizes], axis=0)
            right_orders = tf.concat(
                [left_right_orders, right_right_orders], axis=0)
            right_sizes = tf.concat(
                [left_right_sizes, right_right_sizes], axis=0)
            parents = tf.concat(
                [left_parents, right_parents], axis=0)

            return depth+1, node_id, parents, left_orders, right_orders, left_sizes, right_sizes

        final = tf.while_loop(condition, func,
                              loop_vars=init_task,
                              shape_invariants=shape_invariants,
                              back_prop=False)
        return final, self.tree_


if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from decision_tree.four import build_regression_tree
    dx, dy = make_regression(n_features=29)
    with tf.Session() as sess:
        x = tf.placeholder(tf.float64, [None, 29])
        y = tf.placeholder(tf.float64, [None])
        clf = DecisionTree(max_depth=4)
        new_tree = clf.fit(x, y)
        sess.run(tf.global_variables_initializer())
        _, tf_t = sess.run(new_tree, feed_dict={x: dx, y: dy})
        t = build_regression_tree(dx, dy, 4)
        import ipdb
        ipdb.set_trace()
        np.testing.assert_allclose(t.tree_.threshold, tf_t.threshold)
