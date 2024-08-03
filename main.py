import torch
import torch.nn as nn

x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
  return w * x

print(f'Prediction before training f(x) = {forward(6):.3f}')

learning_rate = 0.01
n_iters = 100
loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=learning_rate)

for epoch in range(n_iters):
  y_pred = forward(x)
  l = loss(y_pred, y)
  l.backward()
  optimizer.step()
  optimizer.zero_grad()
  if epoch % 10 == 0:
    print(f'Epoch {epoch}, Loss: {l.item():.3f}')

print(f'Prediction after training f(x) = {forward(6):.3f}')