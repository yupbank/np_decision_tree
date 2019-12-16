import tensorflow as tf
import numpy as np
import collections
from tf_utils import tf_best_variance_improvements
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def init_tree(max_node):
    Tree = collections.namedtuple(
        'DecisionTree', 'left_childern right_children feature threshold value')
    children_left, children_right = np.zeros(
        max_node-1, dtype=np.int64), np.zeros(max_node-1, dtype=np.int64)
    feature = np.zeros(max_node-1, dtype=np.int64)
    value = np.zeros(max_node-1)
    threshold = np.zeros(max_node-1)

    children_left[:] = -1
    children_right[:] = -1
    feature[:] = -2
    threshold[:] = -2

    return Tree(tf.get_variable('children_left', initializer=children_left, use_resource=True), tf.get_variable('children_right', initializer=children_right, use_resource=True),
                tf.get_variable('feature', initializer=feature, use_resource=True), tf.get_variable('threshold', initializer=threshold, use_resource=True), tf.get_variable('value', initializer=value, use_resource=True))


def tf_tree(X, y, max_depth=2, min_improvement=1e-7, min_sample_leaf=1):
    tree = init_tree(max_depth)
    left_sizes, right_sizes = np.array([100]), np.array([0])
    left_orders, right_orders = tf.argsort(
        X, axis=0), np.empty((100, 19), dtype=np.int32)
    left_mask, right_mask = tf.get_variable('left_mask', initializer=tf.zeros(
        100, dtype=tf.bool), use_resource=True), tf.get_variable('right_mask', initializer=tf.zeros(100, dtype=tf.bool), use_resource=True)
    init_depth = tf.constant(0)
    node_id = tf.constant(0, dtype=tf.int64)
    parents = np.array([0])
    loop_vars = (init_depth,
                 node_id,
                 parents,
                 left_orders,
                 right_orders,
                 left_sizes,
                 right_sizes,
                 )

    shape_invariants = (
        tf.TensorShape([]),
        tf.TensorShape([]),
        tf.TensorShape([None]),
        tf.TensorShape([None, 19]),
        tf.TensorShape([None, 19]),
        tf.TensorShape([None]),
        tf.TensorShape([None]),
    )

    def split_level(depth, node_id, parents, orders, sizes, is_left=False):
        @tf.function
        def add_node(node_id, parent, is_left):
            if is_left:
                children = tree.left_childern
            else:
                children = tree.right_children
            ops = children[parent].assign(node_id)
            with tf.control_dependencies([ops]):
                return node_id+1

        def add_leaf(node_id, ys):
            ops = tree.value[node_id].assign(ys)
            with tf.control_dependencies([ops]):
                return node_id, tf.constant(0, dtype=tf.int64), tf.constant(0, dtype=tf.int64)

        def add_binary_split(node_id, start, size, best_row, best_col):
            lmask_ops = tf.scatter_update(
                left_mask, orders[start:start+best_row+1, best_col], True)
            rmask_ops = tf.scatter_update(
                right_mask, orders[start+best_row+1:start+size, best_col], True)
            add_threshold_ops = tree.threshold[node_id].assign(
                tf.cast(best_row, tf.float64))
            add_feature_ops = tree.feature[node_id].assign(best_col)

            with tf.control_dependencies([lmask_ops, rmask_ops, add_threshold_ops, add_feature_ops]):
                return node_id, best_row+1, size - best_row+1

        def try_split(ys, node_id, start, size):
            best_row, best_col, improvement = tf_best_variance_improvements(
                ys[start:start+size])
            return tf.cond(improvement <= min_improvement,
                           lambda: add_leaf(
                               node_id, ys[start:start+size, 0]),
                           lambda: add_binary_split(node_id, start, size, best_row, best_col))

        def process_block(prev, elem):
            prev_end, prev_node_id, prev_new_parent, prev_left_size, prev_right_size = prev
            parent, size = elem
            node_id = add_node(prev_node_id, parent, is_left)
            start = prev_end
            new_parent, left_size, right_size = tf.cond(tf.logical_or(size <= min_sample_leaf, depth <= max_depth),
                                                        lambda: add_leaf(
                                                            node_id, ys[start:start+size, 0]),
                                                        lambda: try_split(ys, node_id, start, size))
            return start+size, node_id, new_parent, left_size, right_size

        ys = tf.gather(y, orders)
        initializer = (np.array(0),
                       node_id,
                       np.array(0),
                       np.array(0),
                       np.array(0))
        c, d, new_parents, left_sizes, right_sizes = tf.scan(
            process_block, (parents, sizes), initializer)
        left_orders = tf.reshape(tf.boolean_mask(
            tf.gather(left_mask, orders), orders), (-1, orders.shape[1]))
        right_orders = tf.reshape(tf.boolean_mask(
            tf.gather(right_mask, orders), orders), (-1, orders.shape[1]))
        return (
            tf.boolean_mask(new_parents, left_sizes > 0),
            tf.boolean_mask(left_sizes, left_sizes > 0),
            tf.boolean_mask(right_sizes, right_sizes > 0),
            left_orders,
            right_orders
        )

    #def body(depth, parents, left_orders, left_sizes, right_orders, right_sizes):
    def body(depth,
             node_id,
             parents,
             left_orders,
             right_orders,
             left_sizes,
             right_sizes):
        #return depth+1, parents, left_orders, left_sizes, right_orders, right_sizes
        refresh_left_mask_ops = left_mask.assign(tf.zeros_like(left_mask))
        refresh_right_mask_ops = right_mask.assign(tf.zeros_like(right_mask))
        refresh_ops = tf.group(refresh_left_mask_ops, refresh_right_mask_ops)
        #with tf.control_dependencies(refresh_ops):
        left_parents, left_left_sizes, left_right_sizes, left_left_orders, left_right_orders = split_level(
            depth,
            node_id,
            parents, left_orders, left_sizes, True)
        right_parents, right_left_sizes, right_right_sizes, right_left_orders, right_right_orders = split_level(
            depth,
            node_id,
            parents, right_orders, right_sizes, False)
        left_orders = tf.concat([left_left_orders, right_left_orders], axis=0)
        left_sizes = tf.concat([left_left_sizes, right_left_sizes], axis=0)
        right_orders = tf.concat(
            [left_right_orders, right_right_orders], axis=0)
        right_sizes = tf.concat([left_right_sizes, right_right_sizes], axis=0)
        parents = tf.concat([left_parents, right_parents], axis=0)
        return depth+1, node_id, parents, left_orders, right_orders,  left_sizes, right_sizes

    def condition(depth, *_):
        return depth <= max_depth

    tf.while_loop(condition, body, loop_vars=loop_vars,
                  shape_invariants=shape_invariants, back_prop=False)
    return tree


if __name__ == "__main__":
    x = tf.placeholder(tf.float64, [None, 19])
    y = tf.placeholder(tf.float64, [None])
    print(tf_tree(x, y))
