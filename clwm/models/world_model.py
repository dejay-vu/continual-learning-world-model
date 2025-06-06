import math
import random
from torch import nn
import torch.nn.functional as F
from collections import deque
from .flash_attention import FlashAttentionBlock
from ..utils.common import MAX_ACTIONS, VOCAB_SIZE, REWARD_BINS, unimix_generic


class ReplayBuffer:
    def __init__(self, cap):
        self.b = deque(maxlen=cap)

    def add(self, seq, reward: float):
        self.b.append((seq, reward))

    def sample(self, k):
        return random.sample(self.b, min(k, len(self.b)))


class WorldModel(nn.Module):
    def __init__(self, d=256, layers=6, heads=8):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, d)
        self.blocks = nn.ModuleList(
            [
                FlashAttentionBlock(d_model=d, n_head=heads)
                for _ in range(layers)
            ]
        )
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB_SIZE, bias=False)
        self.reward_head = nn.Linear(d, len(REWARD_BINS), bias=False)
        nn.init.zeros_(self.reward_head.weight)

    def add_task(self):
        for b in self.blocks:
            b.add()

    def forward(self, seq, return_ent=False, return_reward=False):
        x = self.tok(seq)
        ent_sum = 0.0
        for b in self.blocks:
            x, g = b(x)  # g is [batch, n_adapter]
            if return_ent:
                kl = (g * (g + 1e-8).log()).sum(-1) + math.log(g.size(-1))
                ent_sum += kl

        h = self.ln(x)

        out_logits = self.head(h)  # [B, L, V]
        out = (out_logits,)

        if return_ent:
            out += (ent_sum.mean(),)
        if return_reward:
            out += (h,)

        return out if len(out) > 1 else out[0]


class ActorNetwork(nn.Module):
    def __init__(self, d_lat):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_lat, 512), nn.ReLU(), nn.Linear(512, MAX_ACTIONS)
        )

    def forward(self, z, p_unimix: float = 0.01):
        """
        Args
        ----
        z : torch.Tensor
            Latent state from the world model, shape (B, d_lat).
        p_unimix : float
            Fraction of probability mass to spread uniformly (default 1 %).

        Returns
        -------
        probs : torch.Tensor
            Smoothed action distribution, shape (B, MAX_ACTIONS), ∑=1.
        """
        logits = self.net(z)  # (B, MAX_ACTIONS)
        probs = unimix_generic(logits, p_unimix)  # 1 % uniform mix
        return probs


class CriticNetwork(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 512), nn.SiLU(), nn.Linear(512, len(REWARD_BINS), bias=False)
        )
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, z):
        return self.net(z)  # (B, |REWARD_BINS|) logits
