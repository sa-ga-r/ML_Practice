import torch
import torch.nn as nn 
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np 

x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)
n_sample, n_feature = x.shape

input_size = n_feature
output_size = 1
model = nn.Linear(input_size, output_size)

learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 100
for epoch in range(epochs):
  y_pred = model(x)
  loss = criterion(y_pred, y)
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()
  if (epoch-1) % 10 == 0:
    print(f'epoch: {epoch}, loss: {loss.item():.4f}')

predicted = model(x).detach().numpy()
plt.plot(x_numpy, y_numpy, 'ro')
plt.plot(x_numpy, predicted, 'b')
plt.show()