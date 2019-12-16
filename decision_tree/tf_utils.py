import tensorflow as tf
from functools import partial
import collections

Task = collections.namedtuple(
    'Task', 'depth,node_id,parents,left_orders,right_orders,left_sizes,right_sizes')


def argmax_2d(tensor):
    tensor_shape = tf.shape(tensor)
    assert len(tensor.get_shape()) == 2
    flat_tensor = tf.reshape(tensor, [-1])

    argmax = tf.argmax(flat_tensor, output_type=tf.int32)

    argmax_x = argmax // tensor_shape[1]
    argmax_y = argmax % tf.shape(tensor)[1]
    return argmax_x, argmax_y


def normalize(ys):
    return ys/tf.reduce_sum(ys, axis=0, keepdims=True)


def tf_best_variance_improvements(ys, weight=None):
    weight = weight or tf.ones_like(ys[:, 0])
    x = normalize(ys)[:, :-1]
    weight = normalize(weight)[:-1]

    y = x**2/weight[:, tf.newaxis] + (1-x)**2/(1-weight[:, tf.newaxis])
    max_row, max_col = argmax_2d(y)
    max_value = y[max_row, max_col]
    return max_row, max_col, max_value


def add_node(tree, node_id, parent, is_left):
    if is_left:
        children = tree.left_childern
    else:
        children = tree.right_children
    ops = children[parent].assign(node_id)
    with tf.control_dependencies([ops]):
        return node_id+1


def add_leaf(tree, node_id, ys, left_mask, right_mask):
    ops = tree.value[node_id].assign(tf.reduce_mean(ys))
    with tf.control_dependencies([ops]):
        return node_id, 0, 0, left_mask, right_mask


def add_binary_split(tree, orders, node_id, start, size, best_row, best_col, left_mask, right_mask):
    left_mask = tf.tensor_scatter_nd_add(left_mask, orders[start:start+best_row+1, best_col], tf.ones_like(
        orders[start:start+best_row+1, best_col], dtype=tf.bool))
    right_mask = tf.tensor_scatter_nd_add(
        right_mask, orders[start+best_row+1:start+size, best_col], tf.ones_like(orders[start+best_row+1:start+size, best_col], dtype=tf.bool))

    add_threshold_ops = tree.threshold[node_id].assign(
        tf.cast(best_row, tf.float64))
    add_feature_ops = tree.feature[node_id].assign(best_col)

    return node_id, best_row+1, size - best_row+1, left_mask, right_mask


def try_split(tree, ys, min_improvement, orders, node_id, start, size, left_mask, right_mask):
    best_row, best_col, improvement = tf_best_variance_improvements(
        ys[start:start+size])
    return tf.cond(improvement <= min_improvement,
                   lambda: add_leaf(
                       tree,
                       node_id, ys[start:start+size, 0], left_mask, right_mask),
                   lambda: add_binary_split(tree, orders, node_id, start, size, best_row, best_col, left_mask, right_mask))


def split_block(tree, is_left, min_sample_leaf, depth, max_depth, ys, min_improvement, orders, prev, elem):
    prev_end, prev_node_id, prev_new_parent, prev_left_size, prev_right_size, prev_left_mask, prev_right_mask = prev
    parent, size = elem
    node_id = add_node(tree, prev_node_id, parent, is_left)
    start = prev_end
    new_parent, left_size, right_size, left_mask, right_mask = tf.cond(tf.logical_or(size <= min_sample_leaf, depth <= max_depth),
                                                                       lambda: add_leaf(
        tree, node_id, ys[start:start+size, 0], prev_left_mask, prev_right_mask),
        lambda: try_split(tree, ys, min_improvement, orders, node_id, start, size, prev_left_mask, prev_right_mask))
    return start+size, node_id, new_parent, left_size, right_size, left_mask, right_mask


def split_level(tree, x, y, min_sample_leaf, max_depth, min_improvement, depth, node_id, parents, orders, sizes, is_left=False):
    ys = tf.gather(y, orders)
    initial_end = 0
    initial_node_id = node_id
    initial_new_parent = 0
    initial_left_size = 0
    initial_right_size = 0
    initial_left_mask = tf.zeros_like(y, dtype=tf.bool)
    initial_right_mask = tf.zeros_like(y, dtype=tf.bool)
    c, d, new_parents, left_sizes, right_sizes, left_mask, right_mask = tf.scan(
        partial(split_block, tree, is_left,
                min_sample_leaf, depth, max_depth, ys, min_improvement, orders),
        (parents, sizes),
        (initial_end,
         initial_node_id,
         initial_new_parent,
         initial_left_size,
         initial_right_size,
         initial_left_mask,
         initial_right_mask,
         ),
        back_prop=False
    )
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


def depth_split(tree,
                x,
                y,
                min_sample_leaf,
                max_depth,
                min_improvement,
                depth,
                node_id,
                parents,
                left_orders,
                right_orders,
                left_sizes,
                right_sizes):
    left_parents, left_left_sizes, left_right_sizes, left_left_orders, left_right_orders = split_level(
        tree,
        x,
        y,
        min_sample_leaf,
        max_depth,
        min_improvement,
        depth,
        node_id,
        parents, left_orders, left_sizes, True)
    right_parents, right_left_sizes, right_right_sizes, right_left_orders, right_right_orders = split_level(
        tree,
        x,
        y,
        min_sample_leaf,
        max_depth,
        min_improvement,
        depth,
        node_id,
        parents, right_orders, right_sizes, False)
    left_orders = tf.concat([left_left_orders, right_left_orders], axis=0)
    left_sizes = tf.concat([left_left_sizes, right_left_sizes], axis=0)
    right_orders = tf.concat(
        [left_right_orders, right_right_orders], axis=0)
    right_sizes = tf.concat([left_right_sizes, right_right_sizes], axis=0)
    parents = tf.concat([left_parents, right_parents], axis=0)
    return Task(depth+1, node_id, parents, left_orders, right_orders,  left_sizes, right_sizes)


def depth_stop_condition(max_depth, depth, *_):
    return depth <= max_depth
