import math

import torch
import torch.utils.checkpoint as ckpt
from torch import nn

from ..common import (
    ACTION_ID_START,
    H16,
    MAX_ACTIONS,
    REWARD_BINS,
    VOCAB_SIZE,
    W16,
    expect_symlog,
    unimix_generic,
)
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

        nn.init.uniform_(self.reward_head.weight, -3e-3, 3e-3)  # small ≠0

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

    # ------------------------------------------------------------------
    # Imagination rollout helper ---------------------------------------
    # ------------------------------------------------------------------

    def imagine(
        self,
        input_tokens: torch.Tensor,
        *,
        actor: "ActorNetwork",
        critic: "CriticNetwork",
        context_length: int,
        horizon: int,
        gamma: float = 0.99,
        lam_return: float = 0.95,
        return_policy_stats: bool = True,
    ) -> (
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ):
        """Roll out the world model for *horizon* steps.

        Parameters
        ----------
        input_tokens : torch.Tensor
            Context sequence of shape (B, T) containing *frame* and *action*
            tokens.  The **last** frame (``N_PATCH`` tokens) represents the
            *current* observation.
        actor : ActorNetwork
            Policy network mapping latents to action distributions.
        critic : CriticNetwork
            Value network estimating future (symlog-scaled) rewards.
        context_length : int
            Number of most recent *time-steps* (⩽ T // (N_PATCH+1)) to feed
            back into the transformer at every rollout step.
        horizon : int
            How many *imagined* steps to unroll.
        gamma : float, optional
            Discount factor used for λ-return computation (default 0.99).
        lam_return : float, optional
            Mixing parameter λ of the Generalised Advantage Estimation
            (default 0.95).

        Returns
        -------
        latents : torch.Tensor
            Stack of latent vectors with shape (B, horizon + 1, dim).  The
            first slice corresponds to the *current* latent state derived
            from the input context.
        imagined_rewards : torch.Tensor
            Expected symlog-scaled rewards (B, horizon).
        values : torch.Tensor
            Value estimates produced by *critic* for every imagined step,
            shape (B, horizon).
        returns : torch.Tensor
            λ-returns combining *imagined_rewards* and *values* following the
            discount/bootstrapping scheme described in Dreamer-style agents
            (B, horizon).
        """

        device = input_tokens.device

        # Number of VQ-VAE patch tokens that constitute a single frame.
        N_PATCH = H16 * W16  # 25 for 84×84 input images with 16-px patches

        # ------------------------------------------------------------------
        # Initial latent from the *current* observation ----------------------
        # ------------------------------------------------------------------
        last_tokens = input_tokens[:, -N_PATCH:]
        initial_latent = self.tok(last_tokens).mean(1)  # (B, dim)

        latents: list[torch.Tensor] = [initial_latent]
        imagined_rewards: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        actions_list: list[torch.Tensor] = []
        log_probs_list: list[torch.Tensor] = []
        entropies_list: list[torch.Tensor] = []

        # ------------------------------------------------------------------
        # Main rollout loop --------------------------------------------------
        # ------------------------------------------------------------------
        for _ in range(horizon):
            # 1. Sample an *action* from the *actor* given the current latent.
            action_probs = actor(latents[-1].detach())  # (B, |A|)
            action_dist = torch.distributions.Categorical(action_probs)
            actions = action_dist.sample()  # (B,)
            if return_policy_stats:
                log_probs_list.append(action_dist.log_prob(actions))
                entropies_list.append(action_dist.entropy())
                # storing log probs/entropies computed before detach as they
                # are needed for gradient calculation upstream
            actions_list.append(actions)
            # 2. Append the action token to the *truncated* context.
            context_tokens = input_tokens[:, -context_length * (N_PATCH + 1) :]
            action_token = ACTION_ID_START + actions  # shift into action id
            roll = torch.cat([context_tokens, action_token.unsqueeze(1)], 1)

            # 3. Predict the *next* frame token with the transformer.
            with torch.no_grad():
                next_frame_token = self(roll)[:, -1].argmax(-1, keepdim=True)

            # 4. Convert predicted tokens to latent vector.
            next_latent = self.tok(next_frame_token).squeeze(1)  # (B, dim)
            latents.append(next_latent)

            # 5. Predicted reward and value (symlog-space expectations).
            with torch.no_grad():
                reward_pred = expect_symlog(self.reward_head(next_latent))
            imagined_rewards.append(reward_pred)

            value_pred = expect_symlog(critic(next_latent.detach()))
            values.append(value_pred.detach())

        # ------------------------------------------------------------------
        # Stack lists into tensors ------------------------------------------
        # ------------------------------------------------------------------
        latents_tensor = torch.stack(latents, dim=1)  # (B, horizon+1, dim)
        imagined_rewards_tensor = torch.stack(
            imagined_rewards, dim=1
        )  # (B, H)
        values_tensor = torch.stack(values, dim=1)  # (B, H)

        # ------------------------------------------------------------------
        # λ-return computation ----------------------------------------------
        # ------------------------------------------------------------------
        bootstrap = expect_symlog(critic(latents[-1].detach()))  # (B,)
        running_return = bootstrap
        returns = torch.zeros_like(imagined_rewards_tensor, device=device)

        for t in reversed(range(horizon)):
            running_return = imagined_rewards_tensor[:, t] + gamma * (
                (1 - lam_return) * values_tensor[:, t]
                + lam_return * running_return
            )
            returns[:, t] = running_return

        if return_policy_stats:
            actions_tensor = torch.stack(actions_list, dim=1)  # (B, H)
            log_probs_tensor = torch.stack(log_probs_list, dim=1)
            entropies_tensor = torch.stack(entropies_list, dim=1)

            return (
                latents_tensor,
                imagined_rewards_tensor,
                values_tensor,
                returns,
                actions_tensor,
                log_probs_tensor,
                entropies_tensor,
            )

        return latents_tensor, imagined_rewards_tensor, values_tensor, returns


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
