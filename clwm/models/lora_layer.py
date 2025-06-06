import math
import torch
import torch.nn as nn


class LoRA(nn.Module):
    """Simple Low-Rank Adaptation (LoRA) layer."""

    def __init__(self, d: int, r: int = 16, alpha: int = 32) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.zeros(r, d))
        self.B = nn.Parameter(torch.empty(d, r))

        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

        self.scale = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.B) @ self.A * self.scale
