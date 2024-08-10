import torch
from torch.utils.data import dataset, dataloader
import numpy as np 
import math

class WineDataset(dataset):
  def __init__(self):
    xy = np.loadtxt('./wine/wine.csv', delimiter = ",", dtype = np.float32, skiprows = 1)
    self.x = torch.from_numpy(xy[:, 1:])
    self.y = torch.from_numpy(xy[:, [0]])
    n_sample = xy.shape[0]
    def __getitem__(self, index):
      return self.x[index], self.y[index]
    def __len__(self):
      return n_sample

dataset = WineDataset()
dataloader = Dataloader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

num_epochs = 100
total_sample = len(dataset)
n_iterations = math.ceil(total_sample / 4)
print(total_sample, n_iterations)

for epoch in range(num_epochs):
  for i, (inputs, lables) in enumerate(dataloader):
    if (i+1) % 5 == 0:
      print(f'epoch {epoch-1}/{num_epochs}, steps={i-1}/{n_iterations}, input={input.shape}')