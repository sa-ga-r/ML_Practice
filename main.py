import torch
import torch.nn as nn
import numpy as np 

def CELoss(actual, predicted):
  loss = -np.sum(actual * np.log(predicted))
  return loss

loss = nn.CrossEntropyLoss()

Y = torch.tensor([0])
y = np.array([1, 0, 0])
Y_Pred_Good = torch.tensor([[2.0, 1.0, 0.1]])
y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.3, 0.6])
Y_Pred_Bad = torch.tensor([[0.5, 0.2, 0.3]])
L1 = loss(Y_Pred_Good, Y)
l1 = CELoss(y, y_pred_good)
L2 = loss(Y_Pred_Bad, Y)
l2 = CELoss(y, y_pred_bad)
print(f'Good Loss : {l1}')
print(f'Bad Loss : {l2}')
print(L1.item())
print(L2.item())