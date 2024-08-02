import numpy as np

x = np.array([1, 2, 3, 4], dtype=np.float32)
y = np.array([2, 4, 6, 8], dtype=np.float32)
w = 0.0

def forward(x):
  return w * x

def loss(y, y_prediction):
  return ((y_prediction - y)**2).mean()

def gredient(x, y, y_prediction):
  return np.dot(2*x, y_prediction - y).mean()

print(f'Prediction before training f(x)={forward(6):.3f}')

learning_rate = 0.01
i_iters = 100

for epoch in range(i_iters):
  y_pred = forward(x)
  l = loss(y, y_pred)
  wd = gredient(x, y, y_pred)
  w -= learning_rate * wd
  if epoch % 10 == 0:
    print(f'Epoch {epoch+1}, Loss {l:.3f}, w {w:.3f}')

print(f'Prediction after training f(x)={forward(6):.3f}')