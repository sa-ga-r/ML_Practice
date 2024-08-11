import torch
import torchvision
from torch.utils.data import dataset
import numpy as np 

class WineDataset(dataset):
  def __init__(self, transform = None):
    xy = np.loadtxt('/wine.txt', delimiter=',', skiprows=1, dtype=np.float32)
    self.x = xy[:, 0:-1]
    self.y = xy[:, -1]
    self.n_samples = self.x.shape[0]
    self.transform = transform
    def __getitem__(self, index):
      sample = self.x[index], self.y[index]
    if self.tranform:
      sample = self.transform(sample)
    return sample
    def __len__(self):
      return self.n_samples

class ToTensor:
  def __call__(self, sample):
    inputs, targets = sample
    return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
  def __init__(self, factor):
    self.factor = factor
  def __call__(self, sample):
    inputs, targets = sample
    inputs *= self.factor
    return inputs, targets


dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, lables = first_data
print(type(features), type(lables))

composed = torchvision.transform.composes([ToTensor(), MulTransform(2)])
first_data = dataset[0]
features, lables = first_data
print(type(features), type(lables))