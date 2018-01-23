# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

N, IN, H, OUT = 64, 1000, 100, 10

x = tf.placeholder(tf.float32, shape=(N, IN))
y = tf.placeholder(tf.float32, shape=(N, OUT))

w1 = tf.Variable(initial_value=tf.random_normal(shape=(IN, H)))
w2 = tf.Variable(initial_value=tf.random_normal(shape=(H, OUT)))

h = tf.matmul(x, w1)
relu = tf.maximum(h, 0)
y_pred = tf.matmul(relu, w2)

loss = tf.reduce_sum((y_pred - y) ** 2)

grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

learning_rate = 1e-6
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_value = np.random.randn(N, IN)
    y_value = np.random.randn(N, OUT)
    for _ in range(500):
        # Execute the graph many times. Each time it executes we want to bind
        # x_value to x and y_value to y, specified with the feed_dict argument.
        # Each time we execute the graph we want to compute the values for loss,
        # new_w1, and new_w2; the values of these Tensors are returned as numpy
        # arrays.
        loss_value, _, _ = sess.run([loss, new_w1, new_w2],
                                    feed_dict={x: x_value, y: y_value})
        print(loss_value)
