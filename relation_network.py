import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationNetwork(nn.Module):
    """Graph Construction Module"""
    def __init__(self):
        super(RelationNetwork, self).__init__()

        self.fc1 = nn.Linear(64, 8)    #64
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = x.view(-1, 64)
        # flatten
        out = F.relu(self.fc1(x))
        out = self.fc2(out)  # no relu
        out = F.relu(out)     # cjh add
        # out = out.view(out.size(0), -1)  # bs*1
        # out = torch.exp(-out)     #cjh add
        return out
