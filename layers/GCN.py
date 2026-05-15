import torch.nn as nn
import torch
import torch.nn.functional as F 


class GCN_C(nn.Module):
    def __init__(self, config):
        super(GCN_C, self).__init__()

        self.x = 1
    

    def forward(self, x):
        return x
    
class GCN_DAE(nn.Module):
    def __init__(self, config):
        super(GCN_DAE, self).__init__()

        self.x = 1
    

    def forward(self, x):
        return x