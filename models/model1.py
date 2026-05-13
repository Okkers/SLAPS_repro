import torch 
import torch.nn as nn 


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.param1 = configs.param1

    def forward(self, x):
        return 