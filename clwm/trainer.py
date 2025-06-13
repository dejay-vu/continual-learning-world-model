"""High-level training driver for **Continual-Learning World Models**.

This class was previously buried inside ``clwm/utils`` – moving it into a
dedicated *top-level* module makes the public API more discoverable and keeps
the directory tree free from generic *utils* packages.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, List

import random
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

import yaml

from .env.atari_envs import make_atari_vectorized_envs
from .models.vqvae_utils import frames_to_indices, vqvae

from .config import Config
from .models.world_model import (
    WorldModel,
    ActorNetwork,
    CriticNetwork,
    Replay,
)
from .models.vqvae import H16, W16, K

from .common import (
    TORCH_DEVICE,
    ACTION_ID_START,
    encode_two_hot,
    expect_symlog,
    split_cross_entropy,
    symexp,
    fisher_diagonal,
    set_global_seed,
)

from .concurrency import AsyncExecutor, StreamManager


class Trainer:
    """High-level orchestrator for continual-learning training runs."""

    # ------------------------------------------------------------------
    # Construction -----------------------------------------------------
    # ------------------------------------------------------------------

    def __init__(self, **cfg):  # noqa: D401 – flexible signature
        """Create a new *Trainer* from a *dict*-like config structure."""

        # Store *full* config for later access (evaluation, logging, …)
        self.raw_cfg = cfg

        self.model_cfg: dict[str, Any] = cfg.get("model", {})
        self.training_cfg: dict[str, Any] = cfg.get("training", {})
        self.cli_args: dict[str, Any] = cfg.get("cli", {})

        # Reproducibility ---------------------------------------------
        set_global_seed(self.cli_args.get("seed", 1))

        # Instantiate networks ----------------------------------------
        self.wm, self.actor, self.critic = self._build_networks()

        # Book-keeping needed for EWC ----------------------------------
        self._running_weights = None
        self._running_fisher = None

    # ------------------------------------------------------------------
    # Public API -------------------------------------------------------
    # ------------------------------------------------------------------

    def train(
        self,
        tasks: Iterable[str] | None = None,
        *,
        evaluation_games: Iterable[str] | None = None,
        eval_interval: int | None = None,
    ) -> List[float]:
        """Train on *tasks* or derive them from *cli* flags."""

        if tasks is None:
            tasks, evaluation_games = self._derive_tasks()

        losses: list[float] = []

        # Build one shared replay buffer -------------------------------
        replay = Replay(cap=self.training_cfg.get("replay_size", 30_000))

        for game in tasks:
            loss = self._train_single_task(game, replay)
            losses.append(float(loss))

            # Periodic evaluation -----------------------------------
            if eval_interval is not None and len(losses) % eval_interval == 0:
                self.evaluate_policy(evaluation_games or tasks)

        return losses

    # ------------------------------------------------------------------
    # Evaluation helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def evaluate_policy(self, games: Iterable[str]):
        """Run *zero-shot* evaluation on *games*."""

        seqs = self.build_evaluation_sequences(games)
        return self.evaluate_on_sequences(seqs)

    def build_evaluation_sequences(self, games: Iterable[str]):
        """Generate rollouts using the *current* policy network."""

        cfg = self.training_cfg
        ctx = cfg["ctx"]

        seqs: list[torch.Tensor] = []

        for game in games:
            envs = make_atari_vectorized_envs(
                game,
                num_envs=cfg.get("eval_envs", 16),
                render_mode=None,
            )

            obs, _ = envs.reset()
            done = np.zeros(len(envs), dtype=bool)

            # Collect rollout ---------------------------------------
            while not done.all():
                # Encode the VQ-VAE indices for the *current* frames.
                img_tokens = frames_to_indices(obs, vqvae, device=TORCH_DEVICE)

                # Build (T, L) token sequence: image + padding (actions)
                seq = torch.full(
                    (len(envs), ctx),
                    ACTION_ID_START,  # dummy action tokens
                    dtype=torch.long,
                    device=TORCH_DEVICE,
                )
                seq[:, : img_tokens.size(1)] = img_tokens  # type: ignore[misc]

                with torch.no_grad():
                    logits = self.actor(seq)[:, -1]
                    act = torch.argmax(logits, -1) - ACTION_ID_START

                obs, _, new_done, _, _ = envs.step(act.cpu().numpy())
                done |= new_done
                seqs.append(seq.cpu())

        if len(seqs) == 0:
            return torch.empty(0, ctx, dtype=torch.long)
        return torch.cat(seqs)

    def evaluate_on_sequences(self, seqs: torch.Tensor):
        """Return CE ‑ (extrinsic) reward tuple for *seqs*."""

        if seqs.numel() == 0:
            return 0.0, 0.0

        ce_img, ce_act = split_cross_entropy(
            self.wm(seqs[:, :-1]), seqs[:, 1:]
        )
        ce = float((ce_img + ce_act).item())

        # Predict cumulative return using critic network --------------
        with torch.no_grad():
            ret_symlog = self.critic(seqs).mean()
            ret = float(symexp(ret_symlog).cpu())

        return ce, ret

    # ------------------------------------------------------------------
    # Internal training helpers ---------------------------------------
    # ------------------------------------------------------------------

    def _train_single_task(self, game: str, replay: Replay) -> float:
        """Run optimisation for *one* Atari game and return the final CE."""

        cfg = self.training_cfg

        loss = self._train_single_task_loop(
            game=game,
            wm=self.wm,
            actor=self.actor,
            critic=self.critic,
            replay=replay,
            epochs=cfg["epochs"],
            ctx=cfg["ctx"],
            imag_h=cfg.get("imag_h", 15),
            gamma=cfg.get("gamma", 0.99),
            lam_return=cfg.get("lam_return", 0.95),
            lam=cfg.get("lam", 0.1),
            running_weights=self._running_weights,
            running_fisher=self._running_fisher,
            online_steps=cfg.get("online_steps", 256),
            num_envs=cfg.get("collector_envs", 16),
            min_prefill=cfg.get("min_prefill", 128),
            batch_size=cfg.get("batch_size", 256),
            sample_ratio=cfg.get("sample_ratio", 0.2),
        )

        # Update EWC statistics after each game ------------------------
        self._update_ewc(replay)

        return float(loss)

    # The following helpers simply wrap the lower-level implementation
    # from *wm.py* to avoid code duplication. Keeping the delegation in
    # one place makes it easier to replace the underlying logic later.

    def _train_single_task_loop(self, *args, **kwargs):  # noqa: D401 – proxy
        from .models.wm import train_on_task

        return train_on_task(*args, **kwargs)

    # ------------------------------------------------------------------
    # EWC helpers ------------------------------------------------------
    # ------------------------------------------------------------------

    def _update_ewc(self, replay: Replay) -> None:
        """Update running Fisher & weight statistics (EWC)."""

        # Approximate diagonal Fisher on a subset of replay samples
        samples = replay.sample_global(256)
        if len(samples) == 0:
            return

        batch = torch.stack([s for s, _ in samples]).to(TORCH_DEVICE)
        fisher = fisher_diagonal(self.wm, batch)

        if self._running_fisher is None:
            self._running_weights = [p.clone().detach() for p in self.wm.parameters()]
            self._running_fisher = fisher
        else:
            for buf_w, buf_f, new_f, p in zip(
                self._running_weights, self._running_fisher, fisher, self.wm.parameters()
            ):
                delta_w = p.detach() - buf_w
                buf_f.add_(new_f)
                buf_w.add_(delta_w)

    # ------------------------------------------------------------------
    # Model construction ----------------------------------------------
    # ------------------------------------------------------------------

    def _build_networks(self):
        wm = WorldModel(**self.model_cfg).to(TORCH_DEVICE)
        actor = ActorNetwork(**self.model_cfg).to(TORCH_DEVICE)
        critic = CriticNetwork(**self.model_cfg).to(TORCH_DEVICE)
        return wm, actor, critic
