import math
import torch
import torch.nn as nn


class LoRA(nn.Module):
    def __init__(self, d, r=16, alpha=32):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(r, d))
        self.B = nn.Parameter(torch.empty(d, r))  # down-proj

        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))  # or .randn_ * 0.02

        self.scale = alpha / r

    def forward(self, x):
        # return x @ (self.A @ self.B) * self.scale
        return (x @ self.B) @ self.A * self.scale
