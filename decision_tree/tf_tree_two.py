import tensorflow as tf
import numpy as np
import collections
from functools import partial
from decision_tree.tf_utils import depth_stop_condition, depth_split, Task
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

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
    return Tree(tf.get_variable('left_children', initializer=children_left, use_resource=True),
                tf.get_variable('right_children',
                                initializer=children_right, use_resource=True),
                tf.get_variable('feature', initializer=feature,
                                use_resource=True),
                tf.get_variable('threshold', initializer=threshold, use_resource=True),
                tf.get_variable('value', initializer=value, use_resource=True))


def tf_tree(X, y, max_depth=5, min_improvement=1e-7, min_sample_leaf=1):
    tree_variable = init_tree(max_depth)

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

    final = tf.while_loop(partial(depth_stop_condition, max_depth), partial(depth_split, tree_variable, X, y, min_sample_leaf, max_depth, min_improvement),
                          loop_vars=init_task,
                          shape_invariants=shape_invariants, back_prop=False)

    with tf.control_dependencies([final.node_id]):
        return tree_variable, final


if __name__ == "__main__":
    x = tf.placeholder(tf.float64, [None, 29])
    y = tf.placeholder(tf.float64, [None])
    from sklearn.datasets import make_regression
    dx, dy = make_regression(n_features=29)
    with tf.Session() as sess:
        new_tree = tf_tree(x, y)
        sess.run(tf.global_variables_initializer())
        print(sess.run(new_tree, feed_dict={x: dx, y: dy}))