import torch
import torch.nn as nn 
from sklearn import datasets
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

bc = datasets.load_breast_cancer()
x, y = bc.data, bc.target

n_sample, n_feature = x.shape
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

class LinearRegression(nn.Module):
  def __init__(self, n_input_feature):
    super(LinearRegression, self).__init__()
    self.linear = nn.Linear(n_input_feature, 1)

  def forward(self, x):
    y_predicted = torch.sigmoid(self.linear(x))
    return y_predicted

model = LinearRegression(n_feature)

learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 100
for epoch in range(num_epochs):
  y_pred = model(x_train)
  loss = criterion(y_pred, y_train)
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()
  if (epoch-1) % 10 == 0:
    print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

with torch.no_grad():
  y_pred = model(x_test)
  y_pred_cls = y_pred.round()
  acc = y_pred_cls.eq(y_test).sum() / float(y_test.shape[0])

print(f'Accuracy = {acc.item():.4f}')