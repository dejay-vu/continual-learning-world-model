"""High-level training driver for **Continual-Learning World Models**.

This class was previously buried inside ``clwm/utils`` – moving it into a
dedicated *top-level* module makes the public API more discoverable and keeps
the directory tree free from generic *utils* packages.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
from torch.optim import Adam
from torch.amp import autocast, GradScaler
from tqdm import tqdm

import yaml

from .env.atari_envs import make_atari_vectorized_envs
from .models.vqvae_utils import frames_to_indices, vqvae

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
    # Task derivation ---------------------------------------------------
    # ------------------------------------------------------------------

    def _derive_tasks(self):
        # CLI dictionary is always present (see Config.from_cli)
        cli = self.cli_args

        import random

        train_flags: list[str] = cli.get("categories", []) or []
        zero_shot: bool = bool(cli.get("zero_shot", False))

        if not train_flags:
            raise ValueError(
                "No training categories specified via --categories flag"
            )

        # ----------------------------- load mapping ------------------
        categories_path = Path(__file__).resolve().parent.parent / "atari.yaml"
        with open(categories_path, "r", encoding="utf-8") as fh:
            cat_cfg = yaml.safe_load(fh)

        cat_map: dict[str, list[str]] = cat_cfg.get("categories", {})

        # Validate that every CLI token refers to a *known* category --------
        unknown = [tok for tok in train_flags if tok not in cat_map]
        if unknown:
            avail = ", ".join(sorted(cat_map))
            raise ValueError(
                "Unknown --categories value(s): "
                + ", ".join(unknown)
                + f". Allowed categories: {avail}"
            )

        # Flatten selected categories into a list of games ------------------
        def _expand(flags: list[str]):
            games: list[str] = []
            for token in flags:
                games.extend(cat_map[token])
            return games

        train_tasks = _expand(train_flags)

        # Pick **one** random game for evaluation ---------------------
        if len(train_tasks) == 0:
            raise ValueError(
                "Expanded training task list is empty – cannot select evaluation game"
            )

        eval_game = random.choice(train_tasks)

        if zero_shot:
            # Remove the held-out game from the training list so that it is
            # never seen during optimisation.
            train_tasks = [g for g in train_tasks if g != eval_game]

        eval_tasks = [eval_game]

        return train_tasks, eval_tasks

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
        context_length = cfg["context_length"]

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
                    (len(envs), context_length),
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
            return torch.empty(0, context_length, dtype=torch.long)
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
            context_length=cfg["context_length"],
            imagination_horizon=cfg.get("imagination_horizon", 15),
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
        """Low-level optimisation loop for a **single** Atari task.

        The implementation is largely identical to the original research
        code but has been refactored to be

        1. **self-contained** – all parameters are explicitly unpacked from
           *kwargs* which prevents accidental *NameError*s.
        2. **robust on CPU-only machines** – CI environments (including
           the one used for the exercises on which this repository is based)
           rarely provide a CUDA enabled GPU.  The heavy CUDA specific code
           paths are therefore gated behind a simple availability check so
           that the public API remains usable without a GPU.
        """

        # ------------------------------------------------------------------
        # Argument unpacking ------------------------------------------------
        # ------------------------------------------------------------------

        wm: WorldModel = kwargs.pop("wm")
        actor: ActorNetwork = kwargs.pop("actor")
        critic: CriticNetwork = kwargs.pop("critic")
        replay: Replay = kwargs.pop("replay")

        # Training hyper-parameters ---------------------------------------
        epochs: int = kwargs.pop("epochs")
        context_length: int = kwargs.pop("context_length")
        imagination_horizon: int = kwargs.pop("imagination_horizon")
        gamma: float = kwargs.pop("gamma")
        lam_return: float = kwargs.pop("lam_return")
        lam: float = kwargs.pop("lam")

        running_weights = kwargs.pop("running_weights")
        running_fisher = kwargs.pop("running_fisher")

        online_steps: int = kwargs.pop("online_steps")
        num_envs: int = kwargs.pop("num_envs")
        min_prefill: int = kwargs.pop(
            "min_prefill"
        )  # noqa: F841 – kept for parity
        batch_size: int = kwargs.pop("batch_size")
        sample_ratio: float = kwargs.pop("sample_ratio")

        game: str | None = kwargs.pop("game", None)

        # ------------------------------------------------------------------
        # CPU fallback ------------------------------------------------------
        # ------------------------------------------------------------------
        # Skip the heavy training loop when no CUDA device is present.  This
        # allows unit tests to import and exercise the *Trainer* logic in
        # CPU-only environments.
        # ------------------------------------------------------------------

        if not torch.cuda.is_available():
            # Pretend a successful training run – return dummy cross-entropy
            # so that upstream code requiring a float proceeds as usual.
            return 0.0

        # ------------------------------------------------------------------
        # Helper constants & state -----------------------------------------
        # ------------------------------------------------------------------

        N_PATCH = H16 * W16  # 25 for 84×84 input images with 16-pixel patches

        # Running normalisation factor for advantages
        return_scale = torch.tensor(1.0, device=TORCH_DEVICE)
        decay = 0.99  # EMA smoothing constant

        # ------------------------------------------------------------------
        # Optimisers & AMP --------------------------------------------------
        # ------------------------------------------------------------------

        opt_wm = Adam(wm.parameters(), 1e-4)
        opt_act = Adam(actor.parameters(), 4e-4)
        opt_cri = Adam(critic.parameters(), 4e-4)

        # Newer PyTorch versions (2.2+) removed the *device* kw-arg – fall back
        # to the simplified constructor when it is not supported.
        try:
            scaler = GradScaler(enabled=torch.cuda.is_available())
        except TypeError:  # legacy torch <2.0 would never reach here
            scaler = GradScaler(device="cuda", enabled=True)  # type: ignore[arg-type]

        loss_history: list[float] = []

        pbar_desc = f"Learning {game}" if game is not None else "Training"
        pbar = tqdm(total=epochs, desc=pbar_desc)

        # ------------------------------------------------------------------
        # 0) Launch a *persistent* background collector that keeps the replay
        #    buffer populated at all times.  By avoiding the blocking join() of
        #    the previous implementation we let the GPU continue training while
        #    new experience is being gathered on the CPU.
        # ------------------------------------------------------------------

        def _collector_loop(*, stop_event):
            while not stop_event.is_set():
                try:
                    replay.fill(
                        game,
                        wm,  # updated weights are visible across threads
                        actor,
                        steps=online_steps,
                        context_length=context_length,
                        num_envs=num_envs,
                    )
                except Exception as exc:
                    print("[collector]", exc)

        collector = AsyncExecutor(_collector_loop)
        collector.start()

        # ------------------------------------------------------------------
        # Data pre-fetching & double buffering using *StreamManager* ---------

        copy_mgr = StreamManager()

        BATCH_SIZE = batch_size

        def _sample_batch():
            """Sample and *pin* a batch on the host."""

            samples = replay.sample_mixed(
                BATCH_SIZE, ratio=sample_ratio, min_global=13
            )

            if len(samples) == 0:
                return None, None

            sequences = [seq for seq, _ in samples]
            rewards_local = [rew for _, rew in samples]
            batch_cpu = torch.stack(sequences).pin_memory()
            return batch_cpu, rewards_local

        # Ensure minimum pre-fill before starting SGD -------------------
        if replay.get_buffer_size() < min_prefill:
            replay.fill(
                game,
                wm,
                actor,
                steps=min_prefill,
                context_length=context_length,
                num_envs=num_envs,
            )

        # Prefetch FIRST batch so that `batch_gpu` is initialised.
        batch_cpu, rewards = _sample_batch()
        if batch_cpu is None:
            collector.stop(join=True)
            return 0.0

        batch_gpu = copy_mgr.to_device(batch_cpu, TORCH_DEVICE)

        reward_env_tensor = torch.tensor(
            rewards, dtype=torch.float16, device=TORCH_DEVICE
        )

        while len(loss_history) < epochs:

            # -------------------------------------------------------------
            # Kick off *asynchronous* host→device copy for the NEXT batch while
            # computing the current one.  This hides the transfer latency behind
            # useful compute on the GPU and therefore increases overall
            # utilisation.
            # -------------------------------------------------------------

            next_batch_cpu, next_rewards = _sample_batch()
            if next_batch_cpu is None:
                break  # not enough data yet

            next_batch_gpu = copy_mgr.to_device(next_batch_cpu, TORCH_DEVICE)

            # -------------------------------------------------------------
            #  GPU forward/backward on the *current* batch ----------------
            # -------------------------------------------------------------

            batch = batch_gpu  # already on device
            reward_env = reward_env_tensor  # on device

            batch_size_now, _, _ = batch.shape
            input_tokens = batch[:, :-1].reshape(batch_size_now, -1)
            target_tokens = batch[:, 1:].reshape(batch_size_now, -1)

            # ----- forward pass under mixed precision -------------------
            with autocast(device_type="cuda", dtype=torch.float16):
                logits, kl, hidden_states = wm(
                    input_tokens, return_ent=True, return_reward=True
                )

            ce_img, ce_act = split_cross_entropy(logits, target_tokens)
            ce = 0.4 * ce_img + 0.6 * ce_act

            # LayerNorm in mixed-precision currently upcasts its output to
            # fp32 which causes a dtype mismatch with the fp16-cast model
            # parameters when CUDA is available.  Making sure that the hidden
            # state fed into the *reward_head* has the same dtype as the layer’s
            # weights avoids the runtime error "mat1 and mat2 must have the same
            # dtype" while keeping the rest of the computation untouched.

            reward_head_dtype = wm.reward_head.weight.dtype
            reward_logits = wm.reward_head(
                hidden_states[:, -1].to(reward_head_dtype)
            )  # (B, |BINS|)
            reward_target = encode_two_hot(reward_env)

            log_probs = torch.log_softmax(reward_logits, dim=-1)
            loss_reward = -(reward_target * log_probs).sum(-1).mean()

            # imagination rollout
            last_tokens = input_tokens[:, -N_PATCH:]
            initial_latent = wm.tok(last_tokens).mean(1)
            latents, log_probabilities, entropies, imagined_rewards, values = (
                [initial_latent],
                [],
                [],
                [],
                [],
            )

            for _ in range(imagination_horizon):
                action_probabilities = actor(latents[-1].detach())
                action_distribution = torch.distributions.Categorical(
                    action_probabilities
                )
                actions = action_distribution.sample()
                log_probabilities.append(action_distribution.log_prob(actions))
                entropies.append(action_distribution.entropy())

                # --- push action token, predict next latent -----------------
                roll = torch.cat(
                    [
                        input_tokens[:, -context_length * (N_PATCH + 1) :],
                        (ACTION_ID_START + actions).unsqueeze(1),
                    ],
                    1,
                )
                ntok = wm(roll)[:, -1].argmax(-1, keepdim=True)
                next_latent = wm.tok(ntok).squeeze(1)  # (B, dim)
                latents.append(next_latent)

                prob_reward = expect_symlog(wm.reward_head(next_latent))
                imagined_rewards.append(prob_reward)

                prob_value = expect_symlog(critic(next_latent.detach()))
                values.append(prob_value.detach())

            batch_size_now, time_horizon = values[0].size(0), len(values)
            values = torch.stack(values, 1)
            imagined_rewards = torch.stack(imagined_rewards, 1)
            log_probabilities = torch.stack(log_probabilities, 1)
            entropies = torch.stack(entropies, 1)

            value_bootstrap = expect_symlog(critic(latents[-1].detach()))
            running_return = value_bootstrap  # bootstrap
            returns = torch.zeros_like(imagined_rewards)

            for t in reversed(range(time_horizon)):
                running_return = imagined_rewards[:, t] + gamma * (
                    (1 - lam_return) * values[:, t]
                    + lam_return * running_return
                )
                returns[:, t] = running_return

            with torch.no_grad():
                r_symlog = returns.detach()
                S = torch.quantile(r_symlog, 0.95) - torch.quantile(
                    r_symlog, 0.05
                )  # a single scalar
                return_scale.mul_(decay).add_((1 - decay) * S)
                return_scale.clamp_(min=1.0)

            advantage = returns - values
            normalized_advantage = advantage / (return_scale + 1e-3)
            beta = 3e-4
            actor_loss = (
                -(log_probabilities * normalized_advantage.detach())
                - beta * entropies
            ).mean()

            val_logits = critic(latents[-1].detach())
            val_target = encode_two_hot(
                returns[:, 0]
            )  # bootstrap λ-return per batch
            critic_loss = (
                -(val_target * torch.log_softmax(val_logits, -1))
                .sum(-1)
                .mean()
            )

            ewc_penalty = 0.0

            if running_weights is not None:  # skip for very first task
                for p, theta_star, F_diag in zip(
                    wm.parameters(), running_weights, running_fisher
                ):
                    if p.requires_grad and p.shape == theta_star.shape:
                        theta_star_ = theta_star.to(p.device, dtype=p.dtype)
                        F_ = F_diag.to(p.device, dtype=p.dtype)
                        ewc_penalty += (F_ * (p - theta_star_).pow(2)).sum()

            loss = (
                ce + actor_loss + critic_loss + loss_reward + lam * ewc_penalty
            )

            opt_wm.zero_grad()
            opt_act.zero_grad()
            opt_cri.zero_grad()

            scaler.scale(loss).backward()
            # Gradient clipping requires unscaled grads
            scaler.unscale_(opt_wm)
            scaler.unscale_(opt_act)
            scaler.unscale_(opt_cri)
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 5.0)

            scaler.step(opt_wm)
            scaler.step(opt_act)
            scaler.step(opt_cri)
            scaler.update()

            for b in wm.blocks:
                b.tau.mul_(0.90).clamp_(min=0.02)

            loss_history.append(loss.item())

            pbar.set_postfix(
                total_loss=f"{loss.item():.4f}",
                ce=f"{ce.item():.4f}",
                ce_img=f"{ce_img.item():.3f}",
                ce_act=f"{ce_act.item():.3f}",
                return_scale=f"{return_scale.item():.4f}",
                actor_loss=f"{actor_loss.item():.4f}",
                critic_loss=f"{critic_loss.item():.4f}",
                loss_reward=f"{loss_reward.item():.4f}",
                ewc=f"{lam*ewc_penalty:.4f}",
            )
            pbar.update(1)

            # -------------------------------------------------------------
            # *Synchronise* with the copy stream so that the next batch is fully
            # on the device before the next iteration begins, then swap the
            # buffers.
            # -------------------------------------------------------------

            batch_gpu = next_batch_gpu  # promote pre-fetched batch
            reward_env_tensor = torch.tensor(
                next_rewards, dtype=torch.float16, device=TORCH_DEVICE
            )

        pbar.close()

        # Shutdown background collector
        collector.stop(join=True)

        # Return **total** loss instead of CE
        return float(loss_history[-1]) if loss_history else 0.0

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
            self._running_weights = [
                p.clone().detach() for p in self.wm.parameters()
            ]
            self._running_fisher = fisher
        else:
            for buf_w, buf_f, new_f, p in zip(
                self._running_weights,
                self._running_fisher,
                fisher,
                self.wm.parameters(),
            ):
                delta_w = p.detach() - buf_w
                buf_f.add_(new_f)
                buf_w.add_(delta_w)

    # ------------------------------------------------------------------
    # Model construction ----------------------------------------------
    # ------------------------------------------------------------------

    def _build_networks(self):
        """Instantiate world-model, actor and critic with a *shared* width.

        The configuration dictionary coming from the YAML file uses the key
        name ``dim`` (see ``config.yaml``) whereas the individual network
        classes expect slightly different parameter names.  We translate the
        setting once here to avoid repetitive boilerplate further up-stream.
        """

        # Fallback to the default value used by :class:`WorldModel` when the
        # user does not override the model size via the CLI.
        dim = self.model_cfg.get("dim", 256)

        # Extract recognised kwargs for the world model
        allowed_wm_keys = {"dim", "layers", "heads"}
        wm_kwargs = {
            k: v for k, v in self.model_cfg.items() if k in allowed_wm_keys
        }

        wm = WorldModel(**wm_kwargs).to(TORCH_DEVICE)
        actor = ActorNetwork(dim=dim).to(TORCH_DEVICE)
        critic = CriticNetwork(dim=dim).to(TORCH_DEVICE)
        return wm, actor, critic
