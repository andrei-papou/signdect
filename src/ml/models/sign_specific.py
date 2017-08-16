import tensorflow as tf


w_init = tf.random_normal_initializer(mean=0.0, stddev=0.01)


def sign_specific(inputs: tf.Tensor, categories: int):
    reg = tf.contrib.layers.l2_regularizer(0.25)
    net = tf.layers.conv2d(
        inputs,
        kernel_size=(3, 3),
        filters=100,
        strides=1,
        activation=tf.nn.relu,
        kernel_regularizer=reg,
        kernel_initializer=w_init)
    # 100 x 62 x 62
    net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=2)
    # 100 x 31 x 31
    net = tf.layers.conv2d(
        net,
        kernel_size=(4, 4),
        filters=150,
        strides=1,
        activation=tf.nn.relu,
        kernel_regularizer=reg,
        kernel_initializer=w_init)
    # 150 x 28 x 28
    net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=2)
    # 150 x 14 x 14
    net = tf.layers.conv2d(
        net,
        kernel_size=(5, 5),
        filters=250,
        strides=1,
        activation=tf.nn.relu,
        kernel_regularizer=reg,
        kernel_initializer=w_init)
    # 250 x 10 x 10
    net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=2)
    # 250 x 5 x 5
    net = tf.layers.conv2d(
        net,
        kernel_size=(2, 2),
        filters=200,
        strides=1,
        activation=tf.nn.relu,
        kernel_regularizer=reg,
        kernel_initializer=w_init)
    # 200 x 4 x 4
    net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=2)
    net = tf.reshape(net, [-1, 200 * 2 * 2])
    net = tf.layers.dense(net, units=256, activation=tf.nn.relu, kernel_regularizer=reg, kernel_initializer=w_init)
    net = tf.layers.dropout(net, rate=0.5)
    return tf.layers.dense(net, units=categories, kernel_regularizer=reg, kernel_initializer=w_init)
