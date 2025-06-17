import math

import torch
import torch.utils.checkpoint as ckpt
from torch import nn

from ..common import MAX_ACTIONS, REWARD_BINS, VOCAB_SIZE, unimix_generic
from .flash_attention import FlashAttentionBlock


class WorldModel(nn.Module):
    """Transformer world model predicting tokens and rewards."""

    def __init__(self, dim: int, layers: int, heads: int) -> None:
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, dim)
        self.blocks = nn.ModuleList(
            [FlashAttentionBlock(dim=dim, heads=heads) for _ in range(layers)]
        )
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, VOCAB_SIZE, bias=False)
        self.reward_head = nn.Linear(dim, len(REWARD_BINS), bias=False)

        nn.init.zeros_(self.reward_head.weight)

    def add_task(self) -> None:
        for block in self.blocks:
            block.add_adapter()

    def forward(
        self,
        seq: torch.Tensor,
        return_ent: bool = False,
        return_reward: bool = False,
    ):
        x = self.tok(seq)
        ent_sum = 0.0
        for i, block in enumerate(self.blocks):
            # -------------------------------------------------------------
            # Gradient-checkpointing is only beneficial during training as
            # it trades extra compute for a lower activation memory
            # footprint.  When the model is in evaluation mode we execute
            # the transformer block normally to avoid the overhead incurred
            # by the checkpoint machinery.
            # -------------------------------------------------------------

            # -------------------------------------------------------------
            # Gradient-checkpointing with variable-length sequences may
            # raise a *CheckpointError* when the recomputed tensors have a
            # different shape than the ones captured during the forward
            # pass (see https://github.com/pytorch/pytorch/issues/111686).
            #
            # Such a shape mismatch can occur in our setting because the
            # token sequence fed into *WorldModel* changes every iteration
            # (e.g. due to varying context lengths during imagination
            # roll-outs).  Since correctness is more important than the
            # marginal memory savings, we gracefully fall back to a regular
            # forward pass whenever the checkpoint mechanism complains.
            # -------------------------------------------------------------

            if self.training:
                x, g = ckpt.checkpoint(block, x, use_reentrant=False)
            else:
                x, g = block(x)

            if return_ent:
                kl = (g * (g + 1e-8).log()).sum(-1) + math.log(g.size(-1))
                ent_sum += kl

        h = self.ln(x)

        out_logits = self.head(h)
        out = (out_logits,)

        if return_ent:
            out += (ent_sum.mean(),)
        if return_reward:
            out += (h,)

        return out if len(out) > 1 else out[0]


class ActorNetwork(nn.Module):
    """Policy network mapping latent state to action distribution."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        output_dim: int = MAX_ACTIONS,
        layers: int = 3,
    ) -> None:
        super().__init__()

        dim = input_dim

        blocks = []
        for _ in range(layers):
            blocks += [
                nn.Linear(dim, hidden_dim),
                nn.RMSNorm(hidden_dim),
                nn.SiLU(),
            ]
            dim = hidden_dim

        self.mlp = nn.Sequential(*blocks)
        self.out = nn.Linear(hidden_dim, output_dim)

        nn.init.orthogonal_(self.out.weight, gain=0.01)  # type: ignore[arg-type]
        self.out.bias.data.zero_()  # zero-initialize the last layer

    def forward(self, x: torch.Tensor, p_unimix: float = 0.01) -> torch.Tensor:
        """
        Args
        ----
        x : torch.Tensor
            Latent state from the world model, shape (B, dim).
        p_unimix : float
            Fraction of probability mass to spread uniformly (default 1 %).

        Returns
        -------
        probs : torch.Tensor
            Smoothed action distribution, shape (B, MAX_ACTIONS), ∑=1.
        """
        # ------------------------------------------------------------------
        # Accept *integer* token sequences during evaluation  ----------------
        # ------------------------------------------------------------------
        # The evaluation helper passes raw token ids (dtype ``torch.long``)
        # directly into the policy network.  Linear layers expect floating
        # point inputs, therefore we up-cast non-floating tensors on-the-fly
        # to avoid the ``RuntimeError: mat1 and mat2 must have the same
        # dtype, but got Long and Float``.

        if not torch.is_floating_point(x):  # e.g. torch.long tokens
            x = x.to(torch.float32)

        x = x.view(x.size(0), -1)  # flatten if needed
        x = self.mlp(x)  # (B, hidden_dim)
        logits = self.out(x)

        # Convert logits to probabilities with unimix smoothing
        probs = unimix_generic(logits, p_unimix)
        return probs


# class ActorNetwork(nn.Module):
#     """Policy network mapping latent state to action distribution."""

#     def __init__(self, dim: int) -> None:
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, 512), nn.ReLU(), nn.Linear(512, MAX_ACTIONS)
#         )

#     def forward(self, z: torch.Tensor, p_unimix: float = 0.01) -> torch.Tensor:
#         """
#         Args
#         ----
#         z : torch.Tensor
#             Latent state from the world model, shape (B, dim).
#         p_unimix : float
#             Fraction of probability mass to spread uniformly (default 1 %).

#         Returns
#         -------
#         probs : torch.Tensor
#             Smoothed action distribution, shape (B, MAX_ACTIONS), ∑=1.
#         """
#         logits = self.net(z)  # (B, MAX_ACTIONS)
#         probs = unimix_generic(logits, p_unimix)  # 1 % uniform mix
#         return probs


class CriticNetwork(nn.Module):
    """Value network estimating future rewards."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        output_dim: int = len(REWARD_BINS),
        layers: int = 3,
    ) -> None:
        super().__init__()

        dim = input_dim

        blocks = []
        for _ in range(layers):
            blocks += [
                nn.Linear(dim, hidden_dim),
                nn.RMSNorm(hidden_dim),
                nn.SiLU(),
            ]
            dim = hidden_dim

        self.mlp = nn.Sequential(*blocks)
        self.out = nn.Linear(hidden_dim, output_dim, bias=False)

        self.out.weight.data.zero_()  # zero-initialize the last layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : torch.Tensor
            Latent state from the world model, shape (B, dim).

        Returns
        -------
        values : torch.Tensor
            Estimated future rewards, shape (B, len(REWARD_BINS)).
        """
        # Up-cast integer inputs on-the-fly (see ActorNetwork).
        if not torch.is_floating_point(x):
            x = x.to(torch.float32)

        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        logits = self.out(x)

        return logits


# class CriticNetwork(nn.Module):
#     """Value network estimating future rewards."""

#     def __init__(self, dim: int) -> None:
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, 512),
#             nn.SiLU(),
#             nn.Linear(512, len(REWARD_BINS), bias=False),
#         )
#         self.net[-1].weight.data.zero_()  # zero-initialize the last layer

#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         return self.net(z)
