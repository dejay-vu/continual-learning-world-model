#!/usr/bin/env python
import random
from typing import List

import torch
from torch.optim import Adam
from tqdm import tqdm
import argparse
import yaml
from pathlib import Path
from clwm.models.vqvae import H16, W16, K
from clwm.models.world_model import (
    WorldModel,
    ActorNetwork,
    CriticNetwork,
    Replay,
)

# ---------------- Mixed-precision helpers -----------------------------
import contextlib
from torch.amp import autocast, GradScaler
from clwm.utils import (
    TORCH_DEVICE,
    encode_two_hot,
    expect_symlog,
    set_global_seed,
)
from clwm.utils.training_utils import split_cross_entropy, fisher_diagonal
from clwm.utils.evaluation_utils import (
    build_evaluation_sequences,
    evaluate_on_sequences,
    evaluate_policy,
)


# Global random seed will be set later (after CLI parse) to allow reproducible
# held-out game selection.
# We still default to 1 in case the user does not override via --seed.

DEFAULT_RANDOM_SEED = 1

N_PATCH = H16 * W16  # 25 tokens / frame
MAX_ACTIONS = 18  # max Atari action‑space size
ACTION_ID_START = K  # first action id (128)
PAD_TOKEN = K + MAX_ACTIONS  # mask token (rarely used)
VOCAB_SIZE = PAD_TOKEN + 1  # embedding size 147


def left_pad_sequence(
    seq: torch.Tensor, ctx: int, pad_id: int
) -> torch.Tensor:
    """Left-pad ``seq`` to length ``ctx`` using ``pad_id``."""

    length = seq.size(0)
    if length >= ctx:
        return seq[-ctx:]
    pad = torch.full(
        (ctx - length, seq.size(1)), pad_id, dtype=seq.dtype, device=seq.device
    )
    return torch.cat((pad, seq), 0)


def train_on_task(
    game: str,
    wm: WorldModel,
    actor: ActorNetwork,
    critic: CriticNetwork,
    replay: Replay,
    global_buffer,
    *,
    epochs: int = 6,
    ctx: int = 64,
    imag_h: int = 15,
    gamma: float = 0.99,
    lam_return: float = 0.95,
    # EWC parameters
    lam: float = 0.1,
    return_scale: torch.Tensor = torch.tensor(1.0, device=TORCH_DEVICE),
    decay: float = 0.99,
    running_weights=None,
    running_fisher=None,
    online_steps: int = 256,
    num_envs: int = 16,
):
    """Train world model and actor/critic on one game."""
    opt_wm = Adam(wm.parameters(), 1e-4)
    opt_act = Adam(actor.parameters(), 4e-4)
    opt_cri = Adam(critic.parameters(), 4e-4)

    scaler = GradScaler(device="cuda", enabled=(TORCH_DEVICE == "cuda"))

    loss_history = []

    pbar = tqdm(total=epochs, desc=f"Learning {game}")

    collect_thread = None  # background data-collection worker

    while len(loss_history) < epochs:

        # -------------------------------------------------------------
        # 1) Collect fresh experience from the live environment to keep the
        #    replay buffer up-to-date with the agent's continually improving
        #    policy.  This step runs on the CPU but leverages asynchronous
        #    vector environments and GPU-based VQ-VAE encoding to remove the
        #    previous frame-processing bottleneck.
        # -------------------------------------------------------------

        # -------------------------------------------------------------
        # Optionally overlap data collection with GPU training by running
        # the environment interaction in a *background* thread.  This hides
        # at least part of the emulator latency and increases the effective
        # GPU utilisation without touching the algebraic part of the model.
        # -------------------------------------------------------------

        # Wait for the previous collection cycle (if any) to finish so that
        # we always have fresh data available in the buffer.  The wait is at
        # the *beginning* of the loop which means that collection of the
        # *next* batch runs in parallel with the *current* optimisation step.
        if collect_thread is not None:
            collect_thread.join()

        import threading

        def _collector():
            replay.fill(
                game,
                wm,
                actor,
                global_buffer,
                steps=online_steps,
                ctx=ctx,
                num_envs=num_envs,
            )

        # Launch async collection for the *next* iteration.
        collect_thread = threading.Thread(target=_collector, daemon=True)
        collect_thread.start()

        BATCH_SIZE = 128

        current_samples = replay.sample(int(0.8 * BATCH_SIZE))
        global_samples = (
            random.sample(global_buffer, int(0.2 * BATCH_SIZE))
            if len(global_buffer) >= 13
            else []
        )
        sequences = [x[0] for x in current_samples] + [
            x[0] for x in global_samples
        ]
        sample_rewards = [x[1] for x in current_samples] + [
            x[1] for x in global_samples
        ]

        batch_cpu = torch.stack(sequences)
        if TORCH_DEVICE == "cuda":
            batch_cpu = batch_cpu.pin_memory()
        batch = batch_cpu.to(TORCH_DEVICE, non_blocking=True)  # (B, ctx, 26)
        reward_env = torch.tensor(
            sample_rewards, dtype=torch.float16, device=TORCH_DEVICE
        )  # (B,)

        B, _, _ = batch.shape
        inp = batch[:, :-1].reshape(B, -1)
        tgt = batch[:, 1:].reshape(B, -1)

        # ----- forward pass under mixed precision -------------------
        with (
            autocast(device_type="cuda", dtype=torch.float16)
            if TORCH_DEVICE == "cuda"
            else contextlib.nullcontext()
        ):
            logits, kl, h = wm(inp, return_ent=True, return_reward=True)

        ce_img, ce_act = split_cross_entropy(logits, tgt)
        ce = 0.4 * ce_img + 0.6 * ce_act

        # LayerNorm in mixed-precision currently upcasts its output to
        # fp32 which causes a dtype mismatch with the fp16-cast model
        # parameters when CUDA is available.  Making sure that the hidden
        # state fed into the *reward_head* has the same dtype as the layer’s
        # weights avoids the runtime error "mat1 and mat2 must have the same
        # dtype" while keeping the rest of the computation untouched.

        reward_head_dtype = wm.reward_head.weight.dtype
        reward_logits = wm.reward_head(
            h[:, -1].to(reward_head_dtype)
        )  # (B, |BINS|)
        reward_target = encode_two_hot(reward_env)

        log_probs = torch.log_softmax(reward_logits, dim=-1)
        loss_reward = -(reward_target * log_probs).sum(-1).mean()

        # imagination rollout
        last = inp[:, -N_PATCH:]
        z0 = wm.tok(last).mean(1)
        zs, logps, entropies, imagined_rewards, values = [z0], [], [], [], []

        for _ in range(imag_h):
            probs = actor(zs[-1].detach())
            dist = torch.distributions.Categorical(probs)
            a_s = dist.sample()
            logps.append(dist.log_prob(a_s))  # (B,)
            entropies.append(dist.entropy())  # (B,)  ⬅︎ NEW

            # --- push action token, predict next latent -----------------
            roll = torch.cat(
                [
                    inp[:, -ctx * (N_PATCH + 1) :],
                    (ACTION_ID_START + a_s).unsqueeze(1),
                ],
                1,
            )
            ntok = wm(roll)[:, -1].argmax(-1, keepdim=True)
            z_next = wm.tok(ntok).squeeze(1)  # (B,d)
            zs.append(z_next)

            prob_reward = expect_symlog(wm.reward_head(z_next))
            imagined_rewards.append(prob_reward)

            prob_value = expect_symlog(critic(z_next.detach()))
            values.append(prob_value.detach())

        B, T = values[0].size(0), len(values)
        values = torch.stack(values, 1)  # (B,T)
        imagined_rewards = torch.stack(imagined_rewards, 1)  # (B,T)
        logps = torch.stack(logps, 1)  # (B,T)
        entropies = torch.stack(entropies, 1)  # (B, T)

        v_boot = expect_symlog(critic(zs[-1].detach()))
        R = v_boot  # bootstrap
        returns = torch.zeros_like(imagined_rewards)

        for t in reversed(range(T)):
            R = imagined_rewards[:, t] + gamma * (
                (1 - lam_return) * values[:, t] + lam_return * R
            )
            returns[:, t] = R

        with torch.no_grad():
            r_symlog = returns.detach()
            S = torch.quantile(r_symlog, 0.95) - torch.quantile(
                r_symlog, 0.05
            )  # a single scalar
            return_scale.mul_(decay).add_((1 - decay) * S)
            return_scale.clamp_(min=1.0)

        adv = returns - values  # both are scalars now
        norm_adv = adv / (return_scale + 1e-3)
        beta = 3e-4
        actor_loss = (
            -(logps * norm_adv.detach()) - beta * entropies  # PG
        ).mean()  # −β·H(π)

        val_logits = critic(zs[-1].detach())
        val_target = encode_two_hot(
            returns[:, 0]
        )  # bootstrap λ-return per batch
        critic_loss = (
            -(val_target * torch.log_softmax(val_logits, -1)).sum(-1).mean()
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

        loss = ce + actor_loss + critic_loss + loss_reward + lam * ewc_penalty

        opt_wm.zero_grad()
        opt_act.zero_grad()
        opt_cri.zero_grad()

        if TORCH_DEVICE == "cuda":
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
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 5.0)
            opt_wm.step()
            opt_act.step()
            opt_cri.step()

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

    pbar.close()

    # Ensure the final background collection finishes before we leave the
    # function and potentially destroy the replay buffer.
    if "collect_thread" in locals() and collect_thread is not None:
        collect_thread.join()

    return ce


TASKS: list[str] = []


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--config", type=str, default="config.yaml")

    # New arguments for category-based training.
    cli.add_argument(
        "--categories",
        type=str,
        nargs="+",
        help="One or more game categories to train on (space-separated).",
    )
    cli.add_argument(
        "--zero-shot",
        action="store_true",
        help="Exclude the held-out evaluation game(s) from the training list.",
    )
    cli.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used for held-out game selection and shuffling.",
    )

    args = cli.parse_args()

    # ================================================================
    #  Load configuration and set random seeds
    # ================================================================

    cfg = yaml.safe_load(open(args.config))

    # Set global seed for reproducibility AFTER reading the seed argument so
    # that held-out game selection is deterministic w.r.t. the provided seed.
    set_global_seed(args.seed)

    # The original implementation relied on pre-generated offline datasets and
    # therefore expected a *dataset* section in the configuration file.  For
    # the new purely online variant this section is optional – we keep the
    # lookup but fall back to *None* when the key is absent so that users can
    # provide much leaner configs.

    base_dir = cfg.get("dataset", {}).get("base_dir", None)

    dim = cfg["model"]["dim"]
    wm = WorldModel(
        d=dim, layers=cfg["model"]["layers"], heads=cfg["model"]["heads"]
    ).to(TORCH_DEVICE)
    actor = ActorNetwork(dim).to(TORCH_DEVICE)
    critic = CriticNetwork(dim).to(TORCH_DEVICE)

    # Only use half precision when CUDA is available; on CPU float16 operations
    # often fall back to slow/unimplemented code paths and can cause numerical
    # issues.  The mixed-precision training blocks below already guard on the
    # device type, so here we merely ensure that the model dtypes are sane.
    # Keep model parameters in fp32 while relying on *autocast* for the
    # forward pass.  Casting the weights to fp16 leads to fp16 gradients which
    # are not supported by `torch.cuda.amp.GradScaler` and triggers the
    # runtime error "Attempting to unscale FP16 gradients" during the backward
    # pass.  Using fp32 weights together with autocast still provides the
    # desired mixed-precision speed-ups while maintaining numerical stability
    # and full support for gradient scaling.
    #
    # (If memory is a concern, consider using torch.compile() + NVME offload
    #  or bfloat16 on newer GPUs instead of down-casting trainable parameters.)
    pass  # ← the autocast context below already handles mixed precision

    running_weights = None
    running_fisher = None

    # ------------------------------------------------------------------
    # Determine training and evaluation game lists.
    # ------------------------------------------------------------------

    def _load_categories() -> dict[str, List[str]]:
        path = Path(__file__).resolve().parent / "atari.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"Could not locate atari.yaml at expected path {path}"
            )
        return yaml.safe_load(open(path, "r"))["categories"]

    held_out_games: List[str] = []
    eval_tasks: List[str] = []

    available_categories = _load_categories()

    unknown = [c for c in args.categories if c not in available_categories]
    if unknown:
        raise ValueError(
            "Unknown category names: "
            + ", ".join(unknown)
            + ". Available categories are: "
            + ", ".join(available_categories)
            + "."
        )

    for cat in args.categories:
        games = available_categories[cat]
        held_out = random.choice(games)
        held_out_games.append(held_out)

        if not args.zero_shot:
            TASKS.extend(games)
        else:
            TASKS.extend([g for g in games if g != held_out])

    eval_tasks = held_out_games

    # Randomize training order for continual learning setting.
    seen = set()
    random.shuffle(TASKS)

    losses = []

    for idx, game in enumerate(TASKS):
        # ------------------------------------------------------------------
        # For purely *online* training we start with an empty replay buffer and
        # immediately collect an initial batch of random experience so that
        # the world-model has data to learn from during the first optimisation
        # step.
        # ------------------------------------------------------------------

        replay = Replay(30000)
        global_buffer = []

        # Prefill buffer with on-policy experience.
        replay.fill(
            game,
            wm,
            actor,
            global_buffer,
            steps=cfg["training"].get("prefill_steps", 1024),
            ctx=cfg["training"]["ctx"],
            num_envs=cfg["training"].get("collector_envs", 16),
        )

        if idx > 0 and game not in seen:
            wm.add_task()

        seen.add(game)

        loss = train_on_task(
            game=game,
            wm=wm,
            actor=actor,
            critic=critic,
            replay=replay,
            global_buffer=global_buffer,
            epochs=cfg["training"]["epochs"],
            ctx=cfg["training"]["ctx"],
            imag_h=cfg["training"].get("imag_h", 15),
            gamma=cfg["training"].get("gamma", 0.99),
            lam_return=cfg["training"].get("lam_return", 0.95),
            lam=cfg["training"].get("lam", 0.1),
            running_weights=running_weights,
            running_fisher=running_fisher,
            online_steps=cfg["training"].get("online_steps", 256),
            num_envs=cfg["training"].get("collector_envs", 16),
        )

        losses.append(loss)

        for _eval_game in eval_tasks:
            seq_eval = build_evaluation_sequences(
                wm, actor, _eval_game, ctx=32, n_seq=256
            )
            ce_eval = evaluate_on_sequences(wm, seq_eval)
            print(f"Eval CE on {_eval_game}: {ce_eval:.4f}")
            score = evaluate_policy(actor, wm, _eval_game)
            print(f"Score {_eval_game}: {score}")

        gamma = 0.9
        k = min(256, replay.get_buffer_size())

        if k == 0:
            continue

        samples = replay.sample(k)
        mb = torch.stack([x[0] for x in samples]).reshape(k, -1)
        new_F = fisher_diagonal(wm, mb)

        params = list(wm.parameters())

        if running_fisher is None:
            running_fisher = new_F
            running_weights = [p.detach().cpu() for p in params]
        else:
            if len(new_F) > len(running_fisher):
                pad = len(new_F) - len(running_fisher)
                running_fisher.extend(
                    [torch.zeros_like(t) for t in new_F[-pad:]]
                )
                running_weights.extend(
                    [p.detach().cpu() for p in params[-pad:]]
                )

            for i, (F_run, F_new, p) in enumerate(
                zip(running_fisher, new_F, params)
            ):
                if F_run.shape == F_new.shape:
                    F_run.mul_(gamma).add_(F_new)
                    running_weights[i].copy_(p.detach().cpu())
                else:
                    running_fisher[i] = F_new
                    running_weights[i] = p.detach().cpu()
