# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

D, IN, H, OUT = 64, 1000, 100, 10

x = torch.randn(D, IN).type(torch.FloatTensor)
y = torch.randn(D, OUT).type(torch.FloatTensor)

w1 = torch.randn(IN, H).type(torch.FloatTensor)
w2 = torch.randn(H, OUT).type(torch.FloatTensor)

learning_rate = 1e-4
for t in range(500):
    # Forward pass: compute predicted y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
