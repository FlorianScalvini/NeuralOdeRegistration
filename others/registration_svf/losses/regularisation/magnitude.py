import torch
import torch.nn as nn

class MagnitudeLoss(nn.Module):
    def __init__(self, penalty='l1'):
        super().__init__()
        loss_fn = {'l1': nn.L1Loss, 'l2': nn.MSELoss}
        if penalty not in loss_fn:
            raise ValueError(f"Unknown penalty type: {penalty}")
        self.loss = loss_fn[penalty]()

    def forward(self, x):
        return self.loss(x, torch.zeros_like(x))

