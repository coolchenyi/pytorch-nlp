# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

D, IN, H, OUT = 64, 1000, 100, 10

x = Variable(torch.randn(D, IN).type(torch.FloatTensor), requires_grad=False)
y = Variable(torch.randn(D, OUT).type(torch.FloatTensor), requires_grad=False)

model = torch.nn.Sequential(
    torch.nn.Linear(IN, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, OUT)
)

loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = 1e-4

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for i in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(i, loss.data[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
