import torch
import torch.nn as nn
import torch.nn.functional as f


class MLP(nn.Module):

    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 250)
        self.fc2 = nn.Linear(250, 250)
        self.last = nn.Linear(250, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.last(x)
        return x
