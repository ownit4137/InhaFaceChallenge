from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import math
import torch

class VirtualSoftmax(nn.Module):
  def __init__(self, in_features, out_features, train=True):
    super(VirtualSoftmax, self).__init__()
    self.train = train
    self.in_features = in_features
    self.out_features = out_features
    self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
    nn.init.xavier_uniform_(self.weight)

  def forward(self, input, label):
    if self.train is True:
      # -------- WX = magnitude * original_class_cosine --------
      WX = F.linear(input, self.weight)
      
      # -------- Wvert --------
      W_label = self.weight[label]
      W_label_norm = torch.linalg.norm(W_label, dim=1)
      X_norm = torch.linalg.norm(input, dim=1)
      
      # -------- W_vertX = magnitude * virtual_class_cosine --------
      W_vertX = torch.mul(W_label_norm, X_norm).reshape(-1, 1)
      output = torch.cat((WX, W_vertX), dim=1)
    else:
      output = F.linear(input, self.weight)
    return output