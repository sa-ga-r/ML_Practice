import torch

x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
  return w * x

def loss(y, y_pred):
  return ((y_pred - y)**2).mean()

print(f'Prediction before training f(x) = {forward(5):.3f})')

learning_rate = 0.01
n_iter = 100

for epoch in range(n_iter):
  y_pred = forward(x)
  l = loss(y, y_pred)
  l.backward()
  with torch.no_grad():
    w -= learning_rate * w.grad
  w.grad.zero_()
  if epoch % 10 == 0:
    print(f'Epoch {epoch:3d} | Loss: {l:.3f} | w: {w:.3f}')

print(f'Prediction after training f(x) = {forward(5):.3f})')