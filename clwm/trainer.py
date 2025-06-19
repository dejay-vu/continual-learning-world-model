from collections.abc import Iterable
from pathlib import Path
from random import choice, shuffle
from typing import Any, List

import numpy as np
import torch
import wandb
import yaml
from cv2 import normalize
from torch.optim import Adam, AdamW
from tqdm import tqdm

from .common import (
    ACTION_ID_START,
    TORCH_DEVICE,
    RewardEMA,
    encode_two_hot,
    expect_symlog,
    fisher_diagonal,
    np_symexp,
    set_global_seed,
    split_cross_entropy,
)
from .concurrency import AsyncExecutor, StreamManager
from .env.atari_envs import make_atari_vectorized_envs
from .models.replay import Replay
from .models.vqvae import H16, W16
from .models.vqvae_utils import frames_to_indices, vqvae
from .models.world_model import ActorNetwork, CriticNetwork, WorldModel


class Trainer:
    """High-level orchestrator for continual-learning training runs."""

    def __init__(self, **cfg: dict[str, Any]):
        # Unpack the configuration dictionary
        self.raw_cfg = cfg
        self.model_cfg: dict[str, Any] = cfg["model"]
        self.train_cfg: dict[str, Any] = cfg["train"]
        self.cli_args: dict[str, Any] = cfg["cli"]

        # nets
        self.wm, self.actor, self.critic = self._build_networks()

        # optimizers
        self.opt_wm = Adam(self.wm.parameters(), lr=1e-4)
        self.opt_actor = Adam(self.actor.parameters(), lr=4e-4)
        self.opt_critic = AdamW(
            self.critic.parameters(), lr=3e-5, weight_decay=1e-4
        )

        # reward normaliser
        self.reward_ema = RewardEMA(TORCH_DEVICE)
        self.ema_vals = torch.tensor(
            [0.0, 1.0], device=TORCH_DEVICE, dtype=torch.float32
        )

        self._running_weights: list[torch.Tensor] | None = None
        self._running_fisher: list[torch.Tensor] | None = None

        self._global_frame: int = 0

        self.wandb_run = wandb.init(
            project="dreamformer",
            entity="dejayvu-university-of-oxford",
            config=cfg,
        )

        set_global_seed(self.cli_args.get("seed", 0))

    def train(
        self,
        tasks: Iterable[str] | None = None,
        evaluation_games: Iterable[str] | None = None,
        eval_interval: int | None = 1,
    ) -> List[float]:
        """Train on *tasks* or derive them from *cli* flags."""

        tasks, evaluation_games = self._derive_tasks()

        losses: list[float] = []

        # Build one shared replay buffer -------------------------------
        replay = Replay(cap=self.train_cfg["replay_size"])

        for task in tasks:
            loss = self._train_single_task(task, replay)
            losses.append(float(loss))

            # Periodic evaluation -----------------------------------
            if eval_interval is not None and len(losses) % eval_interval == 0:
                for evaluation_game in evaluation_games:
                    eval_metrics = self.evaluate(
                        evaluation_game,
                        context_length=self.train_cfg["context_length"],
                    )

                    # W&B -------------------------------------------------------------------
                    if wandb.run is not None and eval_metrics is not None:
                        # Protect against potential None returns
                        wandb.log(
                            {
                                "eval/cross_entropy": eval_metrics.get(
                                    "cross_entropy", float("nan")
                                ),
                                "eval/mean_score": eval_metrics.get(
                                    "mean_score", float("nan")
                                ),
                                "eval/mean_ep_len": eval_metrics.get(
                                    "mean_ep_len", float("nan")
                                ),
                                "eval/task": evaluation_game,
                            },
                            step=self._global_frame,
                        )

        # Finish W&B run when training loop exits
        if wandb.run is not None:
            wandb.finish()

        return losses

    def _derive_tasks(self):
        # CLI dictionary is always present (see Config.from_cli)
        cli = self.cli_args

        selected_categories: list[str] = cli["categories"]
        zero_shot: bool = bool(cli["zero_shot"])

        # ----------------------------- load mapping ------------------
        atari_list_path = Path(__file__).resolve().parent.parent / "atari.yaml"
        with open(atari_list_path, "r", encoding="utf-8") as f:
            atari_list = yaml.safe_load(f)

        all_categories: dict[str, list[str]] = atari_list["categories"]

        # Validate that every CLI token refers to a *known* category --------
        unknown_categories = [
            selected_category
            for selected_category in selected_categories
            if selected_category not in all_categories
        ]
        if unknown_categories:
            available = ", ".join(sorted(all_categories))
            raise ValueError(
                "Unknown --categories value(s): "
                + ", ".join(unknown_categories)
                + f". Allowed categories: {available}"
            )

        train_tasks = [
            game
            for selected_category in selected_categories
            for game in all_categories[selected_category]
        ]

        eval_task = choice(train_tasks)

        if zero_shot:
            train_tasks.remove(eval_task)

        shuffle(train_tasks)

        if len(train_tasks) == 0:
            raise ValueError(
                "Expanded training task list is empty - cannot select evaluation game"
            )

        return train_tasks, [eval_task]

    def _train_single_task(self, game: str, replay: Replay) -> float:
        """Run optimisation for *one* Atari game and return the final CE."""

        cfg = self.train_cfg

        loss = self._train_single_task_loop(
            game=game,
            replay=replay,
            epochs=cfg["epochs"],
            context_length=cfg["context_length"],
            imagination_horizon=cfg["imagination_horizon"],
            gamma=cfg["gamma"],
            lam_return=cfg["lam_return"],
            lam=cfg["lam"],
            running_weights=self._running_weights,
            running_fisher=self._running_fisher,
            online_steps=cfg["online_steps"],
            num_envs=cfg["num_envs"],
            min_prefill=cfg["min_prefill"],
            batch_size=cfg["batch_size"],
            sample_ratio=cfg["sample_ratio"],
        )

        # Update EWC statistics after each game ------------------------
        self._update_ewc(replay)

        return float(loss)

    def _train_single_task_loop(
        self,
        game: str,
        replay: Replay,
        epochs: int,
        context_length: int,
        imagination_horizon: int,
        gamma: float,
        lam_return: float,
        lam: float,
        running_weights,
        running_fisher,
        online_steps: int,
        num_envs: int,
        min_prefill: int,
        batch_size: int,
        sample_ratio: float,
    ):
        """Run the training loop for *epochs* iterations."""
        scaler = torch.GradScaler()

        loss_history: list[float] = []

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
                        self.wm,
                        self.actor,
                        steps=online_steps,
                        context_length=context_length,
                        num_envs=num_envs,
                    )
                except Exception as exc:
                    print("[collector]", exc)

        collector = AsyncExecutor(_collector_loop)
        collector.start()

        # ------------------------------------------------------------------
        # Data pre-fetching & double buffering using *StreamManager*

        copy_mgr = StreamManager()

        def _sample_batch():
            """Sample and *pin* a batch on the host."""

            samples = replay.sample_mixed(
                batch_size, ratio=sample_ratio, min_global=13
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
                self.wm,
                self.actor,
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
            rewards, dtype=torch.bfloat16, device=TORCH_DEVICE
        )

        pbar_desc = f"Learning {game}" if game is not None else "Training"
        pbar = tqdm(total=epochs, desc=pbar_desc)

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
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, kl, hidden_states = self.wm(
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

            reward_head_dtype = self.wm.reward_head.weight.dtype
            reward_logits = self.wm.reward_head(
                hidden_states[:, -1].to(reward_head_dtype)
            )  # (B, |BINS|)
            # The replay, environment wrapper and all downstream computations
            # already operate **in symlog space** (see
            # ``clwm.env.atari_envs.wrap_reward_symlog`` and
            # ``common.REWARD_BINS``).  The stored *reward_env* tensor
            # therefore **is** the symlog-scaled reward.  Applying *symlog*
            # once more would incorrectly compress the magnitude
            # ("symlog ∘ symlog") and break the coupling between the predicted
            # distribution and the target encoding.  The extra non-linear
            # call has been removed so that both head and target live in the
            # same, single-application symlog space.

            reward_target = encode_two_hot(reward_env)

            log_probs = torch.log_softmax(reward_logits, dim=-1)
            loss_reward = -(reward_target * log_probs).sum(-1).mean()

            (
                latents_tensor,
                imagined_rewards,
                values,
                returns,
                _actions,
                log_probabilities,
                entropies,
            ) = self.wm.imagine(
                input_tokens,
                actor=self.actor,
                critic=self.critic,
                context_length=context_length,
                horizon=imagination_horizon,
                gamma=gamma,
                lam_return=lam_return,
                return_policy_stats=True,
            )

            # -------------------- reward normalization --------------------
            offset, scale = self.reward_ema(reward_env, ema_vals=self.ema_vals)
            normalized_returns = (returns - offset) / (scale + 1e-3)
            normalized_values = (values - offset) / (scale + 1e-3)
            advantage = normalized_returns - normalized_values

            # -------------------- losses --------------------
            beta = 3e-4
            actor_loss = (
                -(log_probabilities * advantage.detach()) - beta * entropies
            ).mean()

            # --- Done computing actor_loss above ---

            # The critic should estimate the *λ-return* **from the current
            # latent state** (``latents[0]``).  Using the *final* imagined
            # latent (``latents[-1]``) – which already incorporates the whole
            # rollout – biases the value network and makes the learning signal
            # unnecessarily noisy.

            val_logits = self.critic(latents_tensor[:, 0].detach())
            val_target = encode_two_hot(returns[:, 0])  # λ-return at t=0
            critic_loss = (
                -(val_target * torch.log_softmax(val_logits, -1))
                .sum(-1)
                .mean()
            )

            ewc_penalty = 0.0

            if running_weights is not None:  # skip for very first task
                for p, theta_star, F_diag in zip(
                    self.wm.parameters(), running_weights, running_fisher
                ):
                    if p.requires_grad and p.shape == theta_star.shape:
                        theta_star_ = theta_star.to(p.device, dtype=p.dtype)
                        F_ = F_diag.to(p.device, dtype=p.dtype)
                        ewc_penalty += (F_ * (p - theta_star_).pow(2)).sum()

            loss = (
                ce + actor_loss + critic_loss + loss_reward + lam * ewc_penalty
            )

            self.opt_wm.zero_grad()
            self.opt_actor.zero_grad()
            self.opt_critic.zero_grad()

            scaler.scale(loss).backward()

            # Gradient clipping requires unscaled grads
            scaler.unscale_(self.opt_wm)
            scaler.unscale_(self.opt_actor)
            scaler.unscale_(self.opt_critic)
            torch.nn.utils.clip_grad_norm_(self.wm.parameters(), 5.0)

            scaler.step(self.opt_wm)
            scaler.step(self.opt_actor)
            scaler.step(self.opt_critic)
            scaler.update()

            for block in self.wm.blocks:
                block.tau.mul_(0.90).clamp_(min=0.02)

            loss_history.append(loss.item())

            pbar.set_postfix(
                total_loss=f"{loss.item():.4f}",
                ce=f"{ce.item():.4f}",
                ce_img=f"{ce_img.item():.3f}",
                ce_act=f"{ce_act.item():.3f}",
                return_scale=f"{advantage.std().item():.4f}",
                actor_loss=f"{actor_loss.item():.4f}",
                critic_loss=f"{critic_loss.item():.4f}",
                loss_reward=f"{loss_reward.item():.4f}",
                # ewc=f"{lam*ewc_penalty:.4f}",
            )

            # -------------------------------------------------------------
            # W&B logging --------------------------------------------------
            # -------------------------------------------------------------

            if wandb.run is not None:
                self._global_frame += 1
                wandb.log(
                    {
                        "train/total_loss": loss.item(),
                        "train/cross_entropy": ce.item(),
                        "train/ce_image": ce_img.item(),
                        "train/ce_action": ce_act.item(),
                        "train/advantage": advantage.std().item(),
                        "train/actor_loss": actor_loss.item(),
                        "train/critic_loss": critic_loss.item(),
                        "train/loss_reward": loss_reward.item(),
                        "train/ewc_penalty": lam * ewc_penalty,
                        "task": game,
                    },
                    step=self._global_frame,
                )

            pbar.update(1)

            # -------------------------------------------------------------
            # *Synchronise* with the copy stream so that the next batch is fully
            # on the device before the next iteration begins, then swap the
            # buffers.
            # -------------------------------------------------------------

            batch_gpu = next_batch_gpu  # promote pre-fetched batch
            reward_env_tensor = torch.tensor(
                next_rewards, dtype=torch.bfloat16, device=TORCH_DEVICE
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
        actor = ActorNetwork(dim).to(TORCH_DEVICE)
        critic = CriticNetwork(dim).to(TORCH_DEVICE)

        return wm, actor, critic

    @torch.inference_mode()
    def evaluate(
        self,
        game: str,
        context_length: int = 32,
        n_seq: int = 128,
        episodes: int = 128,
        num_envs: int = 128,
    ):
        """
        1. Builds `n_seq` context sequences of length `context_length`
        and returns the blended cross-entropy  (0.4 ⋅ CE_img + 0.6 ⋅ CE_act).

        2. Rolls out the policy for `episodes` episodes and
        returns the mean (sym-exp) score and mean episode length.

        All work is done in a single environment batch to avoid
        spinning up two separate sets of vectorised envs.
        """

        def _rollout_envs(envs, make_seq=True, make_score=True):
            obs, _ = envs.reset(seed=123)
            seqs, eps = [], [[] for _ in range(num_envs)]

            # running episode stats
            ep_scores = np.zeros(num_envs, np.float32)
            ep_lengths = np.zeros(num_envs, np.int32)
            finished = []

            while (make_seq and len(seqs) < n_seq) or (
                make_score and len(finished) < episodes
            ):
                ids_batch = frames_to_indices(obs, vqvae)
                z_batch = self.wm.tok(ids_batch.clone().detach()).mean(1)
                actions = (
                    torch.distributions.Categorical(self.actor(z_batch))
                    .sample()
                    .cpu()
                    .numpy()
                )

                # sequence collection ------------------------------------------------
                if make_seq:
                    for e in range(num_envs):
                        eps[e].append((ids_batch[e], int(actions[e])))
                        if (
                            len(eps[e]) >= context_length
                        ):  # enough frames ⇒ take last context_length
                            seq = torch.stack(
                                [
                                    torch.tensor(
                                        np.append(
                                            t.cpu().numpy(),
                                            ACTION_ID_START + a_,
                                        )
                                    )
                                    for t, a_ in eps[e][-context_length:]
                                ]
                            )
                            seqs.append(seq)
                            if len(seqs) >= n_seq:
                                make_seq = False  # stop collecting sequences

                # step envs ---------------------------------------------------------
                obs, r, term, trunc, _ = envs.step(actions)

                # score bookkeeping -------------------------------------------------
                if make_score:
                    ep_scores += np_symexp(r)
                    ep_lengths += 1
                    done = np.logical_or(term, trunc)
                    if done.any():
                        for e, d in enumerate(done):
                            if d:
                                finished.append(
                                    (
                                        float(ep_scores[e]),
                                        int(ep_lengths[e]),
                                    )
                                )
                                ep_scores[e] = 0.0
                                ep_lengths[e] = 0
                                eps[
                                    e
                                ].clear()  # fresh episode → wipe seq cache

            return seqs, finished[:episodes]

        try:
            self.wm.eval()  # ensure model is in evaluation mode
            self.actor.eval()
            self.critic.eval()  # ensure model is in evaluation mode

            # create env batch *once*, do both tasks
            envs = make_atari_vectorized_envs(game, num_envs=num_envs)

            seqs, finished = _rollout_envs(envs)
            envs.close()

            # ---------- sequence evaluation ------------------------------------------
            batch = torch.stack(seqs).to(
                TORCH_DEVICE
            )  # (n_seq, context_length, *)
            logits = self.wm(batch[:, :-1].reshape(len(seqs), -1))
            ce_img, ce_act = split_cross_entropy(
                logits, batch[:, 1:].reshape(len(seqs), -1)
            )

            blended_ce = 0.4 * ce_img + 0.6 * ce_act
            mean_score, mean_len = map(np.mean, zip(*finished))
            print(
                f"Evaluating on {game}: eval_img_ce={ce_img.item():.4f} | eval_act_ce={ce_act.item():.4f} | score {mean_score:.1f} | frames/ep {mean_len:.0f}"
            )

        finally:
            self.wm.train()  # ensure model is in training mode
            self.actor.train()
            self.critic.train()

        return {
            "cross_entropy": float(blended_ce),
            "mean_score": float(mean_score),
            "mean_ep_len": float(mean_len),
        }
