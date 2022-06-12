import torch
import torch.nn as nn

class SPLoss(nn.Module):

    def __init__(self, S=7, B=2, C=20):
        super(SPLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")


    def forward(self, predictions, target):
        loss = predictions - target 

        return loss