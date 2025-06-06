#!/usr/bin/env python
import random
import torch
from torch.optim import Adam
import numpy as np
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
)

set_global_seed(1)

N_PATCH = H16 * W16  # 25 tokens / frame
MAX_ACTIONS = 18  # max Atari action‑space size
ACTION_ID_START = K  # first action id (128)
PAD_TOKEN = K + MAX_ACTIONS  # mask token (rarely used)
VOCAB_SIZE = PAD_TOKEN + 1  # embedding size 147


def left_pad_sequence(
    seq: torch.Tensor, ctx: int, pad_id: int
) -> torch.Tensor:
    L = seq.size(0)
    if L >= ctx:
        return seq[-ctx:]
    pad = torch.full(
        (ctx - L, seq.size(1)), pad_id, dtype=seq.dtype, device=seq.device
    )
    return torch.cat((pad, seq), 0)


def train_on_task(
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
    opt_wm = Adam(wm.parameters(), 1e-4)
    opt_act = Adam(actor.parameters(), 4e-4)
    opt_cri = Adam(critic.parameters(), 4e-4)

    loss_history = []

    pbar = tqdm(total=epochs, desc="offline")
    while len(loss_history) < epochs:

        current_samples = replay.sample(int(0.8 * 64))
        global_samples = (
            random.sample(global_buffer, int(0.2 * 64))
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

        assert loss_reward.dtype == torch.float32
        assert ce.dtype == torch.float32
        assert critic_loss.dtype == torch.float32
        assert actor_loss.dtype == torch.float32

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


# TASKS = [
#     "SpaceInvaders",
#     "Assault",
#     "DemonAttack",
#     "AirRaid",
#     "Atlantis",
#     "BeamRider",
#     "StarGunner",
#     "Galaxian",
#     "Solaris",
#     "Zaxxon",
# ]
# eval_task = "Phoenix"

# TASKS = [
#     "Adventure",
#     "AirRaid",
#     "Alien",
#     "Amidar",
#     "Assault",
#     "Asterix",
#     "Asteroids",
#     "Atlantis",
#     "Atlantis2",
#     "Backgammon",
#     "BankHeist",
#     "BasicMath",
#     "BattleZone",
#     "BeamRider",
#     "Berzerk",
#     "Blackjack",
#     "Bowling",
#     "Boxing",
#     "Carnival",
#     "Casino",
#     "Centipede",
#     "ChopperCommand",
#     "CrazyClimber",
#     "Crossbow",
#     "Darkchambers",
#     "Defender",
#     "DemonAttack",
#     "DonkeyKong",
#     "DoubleDunk",
#     "Earthworld",
#     "ElevatorAction",
#     "Enduro",
#     "Entombed",
#     "Et",
#     "FishingDerby",
#     "FlagCapture",
#     "Freeway",
#     "Frogger",
#     "Frostbite",
#     "Galaxian",
#     "Gopher",
#     "Gravitar",
#     "Hangman",
#     "HauntedHouse",
#     "Hero",
#     "HumanCannonball",
#     "IceHockey",
#     "Jamesbond",
#     "JourneyEscape",
#     "Kaboom",
#     "Kangaroo",
#     "KeystoneKapers",
#     "KingKong",
#     "Klax",
#     "Koolaid",
#     "Krull",
#     "KungFuMaster",
#     "LaserGates",
#     "LostLuggage",
#     "MarioBros",
#     "MiniatureGolf",
#     "MontezumaRevenge",
#     "MrDo",
#     "MsPacman",
#     "NameThisGame",
#     "Othello",
#     "Pacman",
#     "Phoenix",
#     "Pitfall",
#     "Pitfall2",
#     "Pong",
#     "Pooyan",
#     "PrivateEye",
#     "Qbert",
#     "Riverraid",
#     "RoadRunner",
#     "Robotank",
#     "Seaquest",
#     "SirLancelot",
#     "Skiing",
#     "Solaris",
#     "SpaceInvaders",
#     "SpaceWar",
#     "StarGunner",
#     "Superman",
#     "Surround",
#     "Tennis",
#     "Tetris",
#     "TicTacToe3D",
#     "TimePilot",
#     "Trondead",
#     "Turmoil",
#     "Tutankham",
#     "UpNDown",
#     "Venture",
#     "VideoCheckers",
#     "VideoChess",
#     "VideoCube",
#     "VideoPinball",
#     "WizardOfWor",
#     "WordZapper",
#     "YarsRevenge",
#     "Zaxxon",
# ]
# eval_task = "Breakout"


TASKS = []
eval_task = ""


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--config", type=str, default="config.yaml")
    args = cli.parse_args()

    cfg = yaml.safe_load(open(args.config))

    base_dir = cfg["dataset"]["base_dir"]

    dim = cfg["model"]["dim"]
    wm = (
        WorldModel(
            d=dim, layers=cfg["model"]["layers"], heads=cfg["model"]["heads"]
        )
        .to(TORCH_DEVICE)
        .half()
    )
    actor = ActorNetwork(dim).to(TORCH_DEVICE).half()
    critic = CriticNetwork(dim).to(TORCH_DEVICE).half()

    running_weights = None
    running_fisher = None

    TASKS.extend(cfg["tasks"]["train"])
    eval_task = cfg["tasks"]["eval"]

    losses = []

    seen = set()
    random.shuffle(TASKS)

    for idx, game in enumerate(TASKS):
        game_dir = Path(base_dir) / game
        npz_files = list(game_dir.glob("*.npz"))
        if not npz_files:
            gather_offline_dataset(
                game,
                cfg["dataset"]["collect_steps"],
                str(game_dir),
                reso=cfg["dataset"].get("reso", 84),
                shard=cfg["dataset"].get("shard", 1000),
            )

        frames_t, actions_t, rewards_t, dones_t = load_dataset_to_gpu(
            str(game_dir)
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

        seq_eval = build_evaluation_sequences(
            wm, actor, eval_task, ctx=32, n_seq=256
        )
        ce_eval = evaluate_on_sequences(wm, seq_eval)
        print(f"Eval CE on {eval_task}: {ce_eval:.4f}")
        print(f"Score {eval_task}: {evaluate_policy(actor, wm, eval_task)}")

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
