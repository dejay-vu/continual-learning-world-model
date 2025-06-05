#!/usr/bin/env python
import random
import torch
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
from clwm.vqvae import H16, W16, K
from clwm.wm import WorldModel, Actor, Critic, Buffer
from clwm.utils import (
    DEVICE,
    twohot,
    expected_symlog,
    seed_everything,
)
from clwm.vq_utils import vqvae, frames_to_ids
from clwm.train_utils import split_ce, fisher_diag
from clwm.envs import make_atari_vectorized
from clwm.eval_utils import build_eval_seq, eval_on_sequences, eval_policy

seed_everything(1)

N_PATCH = H16 * W16  # 25 tokens / frame
MAX_ACT = 18  # max Atari action‑space size
ACT_PAD = K  # first action id (128)
PAD = K + MAX_ACT  # mask token (rarely used)
VOCAB = PAD + 1  # embedding size 147


def left_pad(seq: torch.Tensor, ctx: int, pad_id: int) -> torch.Tensor:
    L = seq.size(0)
    if L >= ctx:
        return seq[-ctx:]
    pad = torch.full(
        (ctx - L, seq.size(1)), pad_id, dtype=seq.dtype, device=seq.device
    )
    return torch.cat((pad, seq), 0)


def train_task(
    env_name: str,
    wm: WorldModel,
    actor: Actor,
    critic: Critic,
    replay: Buffer,
    *,
    epochs: int = 6,
    ctx: int = 64,
    collect: int = 1024,
    warmup: int = 256,
    imag_h: int = 15,
    gamma: float = 0.99,
    lam_return: float = 0.95,
    # EWC parameters
    lam: float = 0.1,
    ret_scale: torch.Tensor = torch.tensor(1.0, device=DEVICE),
    decay: float = 0.99,
    running_W=None,
    running_F=None,
):
    envs = make_atari_vectorized(env_name)
    num_envs = envs.num_envs
    obs, _ = envs.reset()

    opt_wm = Adam(wm.parameters(), 1e-4)
    opt_act = Adam(actor.parameters(), 4e-4)
    opt_cri = Adam(critic.parameters(), 4e-4)

    eps = [[] for _ in range(num_envs)]
    losses = []
    new_added = 0

    pbar = tqdm(
        total=epochs,
        desc=env_name,
    )
    while len(losses) < epochs:
        for _ in range(collect):
            ids_batch = frames_to_ids(obs, vqvae)
            z_batch = wm.tok(torch.tensor(ids_batch, device=DEVICE)).mean(1)

            probs = actor(z_batch)
            actions = torch.distributions.Categorical(probs).sample()

            nxt, r, term, trunc, _ = envs.step(actions.cpu().numpy())

            for i in range(num_envs):
                eps[i].append((ids_batch[i], actions[i].item()))

                seq = torch.stack(
                    [
                        torch.tensor(np.append(t, ACT_PAD + a_))
                        for t, a_ in eps[i][-ctx:]
                    ]
                )
                seq = left_pad(seq, ctx, PAD)

                replay.add(seq, r[i])
                global_buf.append((seq, r[i]))

                if term[i] or trunc[i]:  # status for *this* env
                    eps[i].clear()

            new_added += 1
            obs = nxt

        if new_added < warmup:
            pbar.set_postfix(warmup=f"{new_added}/{warmup}")
            continue

        cur = replay.sample(int(0.8 * 64))
        other = (
            random.sample(global_buf, int(0.2 * 64))
            if len(global_buf) >= 13
            else []
        )
        # batch = torch.stack(cur + other).to(DEVICE)
        seqs = [x[0] for x in cur] + [x[0] for x in other]
        rewards = [x[1] for x in cur] + [x[1] for x in other]

        batch = torch.stack(seqs).to(DEVICE)  # (B, ctx, 26)
        reward_env = torch.tensor(
            rewards, dtype=torch.float32, device=DEVICE
        )  # (B,)

        B, _, _ = batch.shape
        inp = batch[:, :-1].reshape(B, -1)
        tgt = batch[:, 1:].reshape(B, -1)

        # logits, kl = wm(inp, return_ent=True)  # kl is mean KL per batch
        logits, kl, h = wm(inp, return_ent=True, return_reward=True)

        ce_img, ce_act = split_ce(logits, tgt)
        ce = 0.4 * ce_img + 0.6 * ce_act

        reward_logits = wm.reward_head(h[:, -1])  # (B, |BINS|)
        reward_target = twohot(reward_env)

        log_probs = torch.log_softmax(reward_logits, dim=-1)
        loss_reward = -(reward_target * log_probs).sum(-1).mean()

        # imagination rollout
        last = inp[:, -N_PATCH:]
        z0 = wm.tok(last).mean(1)
        zs, logps, entropies, rewards, values = [z0], [], [], [], []

        for _ in range(imag_h):
            probs = actor(zs[-1].detach())
            dist = torch.distributions.Categorical(probs)
            a_s = dist.sample()
            logps.append(dist.log_prob(a_s))  # (B,)
            entropies.append(dist.entropy())  # (B,)  ⬅︎ NEW

            # --- push action token, predict next latent -----------------
            roll = torch.cat(
                [inp[:, -ctx * (N_PATCH + 1) :], (ACT_PAD + a_s).unsqueeze(1)],
                1,
            )
            ntok = wm(roll)[:, -1].argmax(-1, keepdim=True)
            z_next = wm.tok(ntok).squeeze(1)  # (B,d)
            zs.append(z_next)

            prob_reward = expected_symlog(wm.reward_head(z_next))
            rewards.append(prob_reward)

            prob_value = expected_symlog(critic(z_next.detach()))
            values.append(prob_value.detach())

        B, T = values[0].size(0), len(values)
        values = torch.stack(values, 1)  # (B,T)
        rewards = torch.stack(rewards, 1)  # (B,T)
        logps = torch.stack(logps, 1)  # (B,T)
        entropies = torch.stack(entropies, 1)  # (B, T)

        v_boot = expected_symlog(critic(zs[-1].detach()))
        R = v_boot  # bootstrap
        returns = torch.zeros_like(rewards)

        for t in reversed(range(T)):
            R = rewards[:, t] + gamma * (
                (1 - lam_return) * values[:, t] + lam_return * R
            )
            returns[:, t] = R

        with torch.no_grad():
            r_symlog = returns.detach()
            S = torch.quantile(r_symlog, 0.95) - torch.quantile(
                r_symlog, 0.05
            )  # a single scalar
            ret_scale.mul_(decay).add_((1 - decay) * S)
            ret_scale.clamp_(min=1.0)

        adv = returns - values  # both are scalars now
        norm_adv = adv / (ret_scale + 1e-3)
        beta = 3e-4
        actor_loss = (
            -(logps * norm_adv.detach()) - beta * entropies  # PG
        ).mean()  # −β·H(π)

        val_logits = critic(zs[-1].detach())
        val_target = twohot(returns[:, 0])  # bootstrap λ-return per batch
        critic_loss = (
            -(val_target * torch.log_softmax(val_logits, -1)).sum(-1).mean()
        )

        ewc_penalty = 0.0

        if running_W is not None:  # skip for very first task
            for p, theta_star, F_diag in zip(
                wm.parameters(), running_W, running_F
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

        losses.append(loss.item())

        pbar.set_postfix(
            total_loss=f"{loss.item():.4f}",
            ce=f"{ce.item():.4f}",
            ce_img=f"{ce_img.item():.3f}",
            ce_act=f"{ce_act.item():.3f}",
            ret_scale=f"{ret_scale.item():.4f}",
            actor_loss=f"{actor_loss.item():.4f}",
            critic_loss=f"{critic_loss.item():.4f}",
            loss_reward=f"{loss_reward.item():.4f}",
            ewc=f"{lam*ewc_penalty:.4f}",
        )
        pbar.update(1)

    envs.close()
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


TASKS = [
    "Phoenix",
    "Phoenix",
    "Phoenix",
    "Phoenix",
    "Phoenix",
    "Phoenix",
    "Phoenix",
    "Phoenix",
    "Phoenix",
]
eval_task = "Phoenix"


if __name__ == "__main__":
    dim = 256
    wm = WorldModel(d=dim, layers=6, heads=8).to(DEVICE)
    actor = Actor(dim).to(DEVICE)
    critic = Critic(dim).to(DEVICE)

    replay = Buffer(30000)
    global_buf = []

    running_W = None  # list[tensor]  –  theta*
    running_F = None  # list[tensor]  –  diagonal Fisher

    losses = []

    seen = set()
    random.shuffle(TASKS)
    for idx, game in enumerate(TASKS):
        if idx > 0 and game not in seen:
            wm.add_task()

        seen.add(game)

        loss = train_task(
            env_name=game,
            wm=wm,
            actor=actor,
            critic=critic,
            replay=replay,
            epochs=20,
            ctx=128,
            collect=512,
            running_W=running_W,
            running_F=running_F,
        )

        losses.append(loss)

        seq_eval = build_eval_seq(wm, actor, eval_task, ctx=512, n_seq=256)
        ce_eval = eval_on_sequences(wm, seq_eval)
        print(f"Eval CE on {eval_task}: {ce_eval:.4f}")
        print(f"Score {eval_task}: {eval_policy(actor, wm, eval_task)}")

        gamma = 0.9
        k = min(256, len(replay.b))

        if k == 0:
            continue

        samples = replay.sample(k)
        mb = torch.stack([x[0] for x in samples]).reshape(k, -1)
        new_F = fisher_diag(wm, mb)

        params = list(wm.parameters())

        if running_F is None:
            running_F = new_F
            running_W = [p.detach().cpu() for p in params]
        else:
            # 1. pad if model gained new parameters (e.g., new LoRA)
            if len(new_F) > len(running_F):
                pad = len(new_F) - len(running_F)
                running_F.extend([torch.zeros_like(t) for t in new_F[-pad:]])
                running_W.extend([p.detach().cpu() for p in params[-pad:]])

            # 2. update where shapes match, replace where they don’t
            for i, (F_run, F_new, p) in enumerate(
                zip(running_F, new_F, params)
            ):
                if F_run.shape == F_new.shape:  # same tensor
                    F_run.mul_(gamma).add_(F_new)
                    running_W[i].copy_(p.detach().cpu())
                else:  # brand-new adapter
                    running_F[i] = F_new  # start its Fisher fresh
                    running_W[i] = p.detach().cpu()

    # seq_eval = build_eval_seq(wm, actor, eval_task, ctx=32, n_seq=256)
    # ce_eval = eval_on_sequences(wm, seq_eval)
    # print(f"Eval CE on {eval_task}: {ce_eval:.4f}")
