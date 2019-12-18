import tensorflow as tf
from functools import partial
import collections
import numpy as np
import tensorflow.compat.v1 as tf
import sys
tf.disable_v2_behavior()

Task = collections.namedtuple(
    'Task', 'depth,node_id,parents,left_orders,right_orders,left_sizes,right_sizes')

Tree = collections.namedtuple(
    'DecisionTree', 'left_children,right_children,feature,threshold,value')


def init_tree(max_depth):
    max_node = 2 ** (max_depth + 1)
    children_left, children_right = np.zeros(
        max_node-1, dtype=np.int32), np.zeros(max_node-1, dtype=np.int32)
    feature = np.zeros(max_node-1, dtype=np.int32)
    value = np.zeros(max_node-1)
    threshold = np.zeros(max_node-1)

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
    return ys/tf.reduce_sum(ys, axis=0, keepdims=True)

 def best_variance_improvements(ys, weight=None):
    def calcluate():
        new_weight = weight or tf.ones_like(ys[:, 0])
        x = normalize(ys)[:-1, :]
        new_weight = normalize(new_weight)[:-1]
        y = x**2/new_weight[:, tf.newaxis] + \
            (1-x)**2/(1-new_weight[:, tf.newaxis])
        max_row, max_col = argmax_2d(y)
        max_value = y[max_row, max_col]
        return max_row, max_col, max_value

    return tf.cond(tf.less_equal(tf.shape(ys)[0], 1),
                   lambda: (0, 0, tf.constant(0.0, dtype=tf.float64)),
                   calcluate)

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

    def new_node(self, node_id, parent, is_left):
        if is_left:
            children = self.tree_.left_children
        else:
            children = self.tree_.right_children
        ops = children[parent].assign(node_id)
        with tf.control_dependencies([ops]):
            return node_id+1

    def add_leaf(self, ys, node_id, start, size):
        y = ys[start:start+size, 0]
        return self.tree_.value[node_id].assign(tf.reduce_mean(y))

    def add_binary_split(self, x, orders, best_row, best_col, node_id, start, size):
        add_threshold_ops = self.tree_.threshold[node_id].assign(
            x[orders[start+best_row, best_col], best_col])
        add_feature_ops = self.tree_.feature[node_id].assign(best_col)
        return tf.group(add_threshold_ops, add_feature_ops)

    def try_split(self, x, ys, depth, orders, node_id, start, size, left_sizes, right_sizes, left_mask, right_mask):
        best_row, best_col, improvement, new_left_sizes, new_right_sizes, new_left_mask, new_right_mask = self.split_data(
            ys, orders, start, size, left_sizes, right_sizes, left_mask, right_mask)
        def new_leaf():
            ops = self.add_leaf(ys, node_id, start, size)
            with tf.control_dependencies([ops]):
                return left_sizes, right_sizes, left_mask, right_mask

        def new_split():
            ops = self.add_binary_split(
                x, orders, best_row, best_col, node_id, start, size)
            with tf.control_dependencies([ops]):
                return new_left_sizes, new_right_sizes, new_left_mask, new_right_mask

        return tf.cond(
            tf.logical_or(
                tf.logical_or(
                    tf.less_equal(improvement, self.min_improvement),
                    tf.less_equal(size, self.min_sample_leaf)),
                tf.less_equal(self.max_depth, depth)),
            new_leaf, new_split)

    def split_block(self, parents, sizes, is_left, depth, ys, orders,
                    prev_counter, prev_end, prev_node_id, prev_new_parent,
                    prev_left_size, prev_right_size, prev_left_mask, prev_right_mask):
        assert_op = tf.assert_greater(
            tf.shape(sizes), prev_counter, message="size too small")
        with tf.control_dependencies([assert_op]):
            parent, size = parents[prev_counter], sizes[prev_counter]
            node_id = self.new_node(prev_node_id, parent, is_left)
            start = prev_end
            new_parent, left_size, right_size, left_mask, right_mask = tf.cond(
                tf.logical_or(size <= self.min_sample_leaf,
                              depth >= self.max_depth),
                partial(self.add_leaf, ys, node_id, start, size,
                        prev_left_mask, prev_right_mask),
                partial(self.try_split, ys, orders, node_id, start, size,
                        prev_left_mask, prev_right_mask))
            assert_op_a = tf.assert_greater_equal(size, left_size)
            assert_op_b = tf.assert_greater_equal(size, right_size)
            with tf.control_dependencies([assert_op_a, assert_op_b]):
                return (
                    prev_counter+1,
                    start+size,
                    node_id,
                    tf.concat([prev_new_parent, [new_parent]], axis=0),
                    tf.concat([prev_left_size, [left_size]], axis=0),
                    tf.concat([prev_right_size, [right_size]], axis=0),
                    left_mask,
                    right_mask
                )

    def split_level(self, x, y, depth, node_id, parents, orders, sizes, is_left=False):
        initial_counter = 0
        initial_prev_end = 0
        initial_new_parents = tf.constant([], dtype=tf.int32)
        initial_left_sizes = tf.constant([], dtype=tf.int32)
        initial_right_sizes = tf.constant([], dtype=tf.int32)
        initial_left_mask = tf.zeros_like(y, dtype=tf.int32)
        initial_right_mask = tf.zeros_like(y, dtype=tf.int32)

        ys = tf.gather(y, orders)

        def condition(counter, *_):
            return tf.less_equal(counter, tf.shape(sizes)[0]-1)

        def func(prev_counter, prev_end, prev_node_id, prev_new_parent, prev_left_sizes, prev_right_sizes, prev_left_mask, prev_right_mask):
            parent, size = parents[prev_counter], sizes[prev_counter]
            new_parents = tf.concat([prev_new_parent, [prev_node_id]], axis=0)
            start, end = prev_end, prev_end+size
            left_sizes, right_sizes, left_mask, right_mask = self.try_split(x, ys, depth, orders, prev_node_id, start, size,
                                                                            prev_left_sizes, prev_right_sizes,
                                                                            prev_left_mask, prev_right_mask)
            node_id = self.new_node(prev_node_id, parent, is_left)
            return (
                prev_counter+1,
                end,
                node_id,
                new_parents,
                left_sizes,
                right_sizes,
                left_mask,
                right_mask
            )

        _, _, node_id, new_parents, left_sizes, right_sizes, left_mask, right_mask = tf.while_loop(
            condition,
            func,
             (
                 initial_counter,
                 initial_prev_end,
                 node_id,
                initial_new_parents,
                initial_left_sizes,
                initial_right_sizes,
                initial_left_mask,
                initial_right_mask,
             ),
             (
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
            tf.concat([new_parents, new_parents], axis=0),
            left_sizes,
            right_sizes,
            left_orders,
            right_orders
        )

    @staticmethod
    def split_data(ys, orders, start, size, left_sizes, right_sizes, left_mask, right_mask):
        best_row, best_col, improvement = best_variance_improvements(
            ys[start:start+size])
        new_left_mask = populate_mask(
            left_mask, orders[start:start+best_row+1, best_col:best_col+1])
        new_right_mask = populate_mask(
            right_mask, orders[start+best_row+1:start+size, best_col:best_col+1])
        new_left_sizes = tf.concat([left_sizes, [best_row+1]], axis=0)
        new_right_sizes = tf.concat(
            [right_sizes, [size-best_row-1]], axis=0)
        return best_row, best_col, improvement, new_left_sizes, new_right_sizes, new_left_mask, new_right_mask

    def first_split(self, X, y):
        start, size = 0, tf.shape(X)[0]
        orders = tf.argsort(X, axis=0)
        left_mask = tf.zeros_like(y, dtype=tf.int32)
        right_mask = tf.zeros_like(y, dtype=tf.int32)
        left_sizes = tf.constant([], dtype=tf.int32)
        right_sizes = tf.constant([], dtype=tf.int32)
        ys = tf.gather(y, orders)
        best_row, best_col, improvement, new_left_sizes, new_right_sizes, new_left_mask, new_right_mask = self.split_data(
            ys, orders, start, size, left_sizes, right_sizes, left_mask, right_mask)

        return tf.cond(improvement > self.min_improvement,
                       lambda: (new_left_sizes, new_right_sizes,
                                slice_order_by_mask(orders, new_left_mask), slice_order_by_mask(orders, new_right_mask)),
                       lambda: (left_sizes, right_sizes,
                                slice_order_by_mask(orders, left_mask), slice_order_by_mask(orders, right_mask))
                       )

    def depth_split(self,
                    x,
                    y,
                    depth,
                    node_id,
                    parents,
                    left_orders,
                    right_orders,
                    left_sizes,
                    right_sizes):
        node_id, left_parents, left_left_sizes, left_right_sizes, left_left_orders, left_right_orders = self.split_level(
            x,
            y,
            depth,
            node_id,
            parents, left_orders, left_sizes, True)
        node_id, right_parents, right_left_sizes, right_right_sizes, right_left_orders, right_right_orders = self.split_level(
            x,
            y,
            depth,
            node_id,
            parents, right_orders, right_sizes, False)
        left_orders = tf.concat(
            [left_left_orders, right_left_orders], axis=0)
        left_sizes = tf.concat(
            [left_left_sizes, right_left_sizes], axis=0)
        right_orders = tf.concat(
            [left_right_orders, right_right_orders], axis=0)
        right_sizes = tf.concat(
            [left_right_sizes, right_right_sizes], axis=0)
        parents = tf.concat([left_parents, right_parents], axis=0)
        return Task(depth+1, node_id, parents, left_orders, right_orders,  left_sizes, right_sizes)

    def fit(self, X, y):
        init_depth = tf.constant(0)
        node_id = tf.constant(0, dtype=tf.int32)
        parents = tf.constant([0])
        init_left_orders = tf.argsort(X, axis=0)
        init_right_orders = tf.zeros_like(X, dtype=tf.int32)
        init_left_sizes = tf.shape(X)[:1]
        init_right_sizes = tf.constant([], dtype=tf.int32)
        init_task = Task(
            init_depth,
            node_id,
            parents,
            init_left_orders,
            init_right_orders,
            init_left_sizes,
            init_right_sizes,
        )
        shape_invariants = Task(
            tf.TensorShape([]),
            tf.TensorShape([]),
            tf.TensorShape([None]),
            X.shape,
            X.shape,
            tf.TensorShape([None]),
            tf.TensorShape([None]),
        )

        final = tf.while_loop(self.depth_stop_condition, partial(self.depth_split, X, y),
                              loop_vars=init_task,
                              shape_invariants=shape_invariants, back_prop=False)

        return final, self.tree_

    def depth_stop_condition(self, depth, *_):
        return depth <= self.max_depth


if __name__ == "__main__":
    from sklearn.datasets import make_regression
    dx, dy = make_regression(n_features=29)
    with tf.Session() as sess:
        x = tf.placeholder(tf.float64, [None, 29])
        y = tf.placeholder(tf.float64, [None])
        clf = DecisionTree()
        new_tree = clf.fit(x, y)
        sess.run(tf.global_variables_initializer())
        print(sess.run(new_tree, feed_dict={x: dx, y: dy}))
