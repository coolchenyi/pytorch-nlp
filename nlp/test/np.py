# -*- coding: utf-8 -*-

import numpy as np

D, IN, H, OUT = 64, 1000, 100, 10

x = np.random.randn(D, IN)
y = np.random.randn(D, OUT)

w1 = np.random.randn(IN, H)
w2 = np.random.randn(H, OUT)
learning_rate = 1e-4

for i in range(100):
    # Forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(0, h)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    print(i, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2





