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
    ReplayBuffer,
)
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
from clwm.data import (
    load_dataset_to_gpu,
    fill_replay_buffer,
    gather_offline_dataset,
    gather_datasets_parallel,
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
    replay: ReplayBuffer,
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
):
    """Train world model and actor/critic on one game."""
    opt_wm = Adam(wm.parameters(), 1e-4)
    opt_act = Adam(actor.parameters(), 4e-4)
    opt_cri = Adam(critic.parameters(), 4e-4)

    loss_history = []

    pbar = tqdm(total=epochs, desc=f"Learning {game}")
    while len(loss_history) < epochs:

        BATCH_SIZE = 64

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

        batch = torch.stack(sequences).to(TORCH_DEVICE)  # (B, ctx, 26)
        reward_env = torch.tensor(
            sample_rewards, dtype=torch.float16, device=TORCH_DEVICE
        )  # (B,)

        B, _, _ = batch.shape
        inp = batch[:, :-1].reshape(B, -1)
        tgt = batch[:, 1:].reshape(B, -1)

        # logits, kl = wm(inp, return_ent=True)  # kl is mean KL per batch
        logits, kl, h = wm(inp, return_ent=True, return_reward=True)

        ce_img, ce_act = split_cross_entropy(logits, tgt)
        ce = 0.4 * ce_img + 0.6 * ce_act

        reward_logits = wm.reward_head(h[:, -1])  # (B, |BINS|)
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

    base_dir = cfg["dataset"]["base_dir"]

    dim = cfg["model"]["dim"]
    wm = WorldModel(
        d=dim, layers=cfg["model"]["layers"], heads=cfg["model"]["heads"]
    ).to(TORCH_DEVICE)
    actor = ActorNetwork(dim).to(TORCH_DEVICE)
    critic = CriticNetwork(dim).to(TORCH_DEVICE)

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

    TASKS = list(dict.fromkeys(TASKS))  # remove duplicates, preserve order
    eval_tasks = held_out_games

    # Randomize training order for continual learning setting.
    seen = set()
    random.shuffle(TASKS)

    losses = []

    missing = [
        game
        for game in TASKS
        if not list((Path(base_dir) / game).glob("*.npz"))
    ]

    if missing:
        gather_datasets_parallel(
            missing,
            cfg["dataset"]["collect_steps"],
            base_dir,
            reso=cfg["dataset"].get("reso", 84),
            shard=cfg["dataset"].get("shard", 1000),
        )

    for idx, game in enumerate(TASKS):
        game_dir = Path(base_dir) / game

        frames_t, actions_t, rewards_t, dones_t = load_dataset_to_gpu(
            str(game_dir),
            batch_size=cfg["dataset"].get("load_bs", 4096),
        )

        replay = ReplayBuffer(30000)
        global_buffer = []
        fill_replay_buffer(
            frames_t,
            actions_t,
            rewards_t,
            dones_t,
            replay,
            ctx=cfg["training"]["ctx"],
            global_buffer=global_buffer,
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
        k = min(256, len(replay.b))

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
