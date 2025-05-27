#!/usr/bin/env python

import math
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from safetensors.torch import load_file
import numpy as np
import gymnasium as gym
import ale_py
from tqdm import tqdm
from matplotlib import pyplot as plt
from vqvae import VQVAE, RES, H16, W16, D_LAT, K  # your tokenizer definition


gym.register_envs(ale_py)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "vqvae_atari.safetensors"  # path to weights


def seed_everything(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed_everything(0)

CHECKPOINT = "vqvae_atari.safetensors"  # path to weights

vqvae = VQVAE().to(DEVICE)
vqvae.load_state_dict(load_file(CHECKPOINT, device=DEVICE))
vqvae.eval()
[p.requires_grad_(False) for p in vqvae.parameters()]

N_PATCH = H16 * W16  # 25 tokens / frame
MAX_ACT = 18  # max Atari action‑space size
ACT_PAD = K  # first action id (128)
PAD = K + MAX_ACT  # mask token (rarely used)
VOCAB = PAD + 1  # embedding size 147


@torch.no_grad()
def frame_to_ids(frame_u8: np.ndarray) -> np.ndarray:
    """uint8 H*W*3 → np.int64[25] codebook ids"""
    x = (
        torch.tensor(frame_u8, dtype=torch.float32, device=DEVICE).permute(
            2, 0, 1
        )
        / 255.0
    )
    x = F.interpolate(
        x[None], (RES, RES), mode="bilinear", align_corners=False
    )
    _, ids, _ = vqvae.vq(vqvae.enc(x).reshape(-1, D_LAT))  # [1,25]
    return ids.squeeze(0).cpu().numpy()


def make_atari(name: str, mode: str = "stochastic"):
    base = f"ALE/{name}-v5"
    kwargs = dict(full_action_space=True, render_mode=None)

    if mode == "noframeskip":
        return gym.make(
            base, frameskip=1, repeat_action_probability=0.0, **kwargs
        )

    if mode == "deterministic":
        return gym.make(
            base, frameskip=4, repeat_action_probability=0.0, **kwargs
        )

    return gym.make(base, **kwargs)  # stochastic default


class Buffer:
    def __init__(self, cap):
        self.b = deque(maxlen=cap)

    def add(self, x):
        self.b.append(x)

    def sample(self, k):
        return random.sample(self.b, min(k, len(self.b)))


class LoRA(nn.Module):
    def __init__(self, d, r=8, alpha=32):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(r, d))
        self.B = nn.Parameter(torch.empty(d, r))  # down-proj

        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))  # or .randn_ * 0.02

        self.scale = alpha / r

    def forward(self, x):
        # return x @ (self.A @ self.B) * self.scale
        return (x @ self.B) @ self.A * self.scale


class Block(nn.Module):
    def __init__(self, d, heads, routers):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
        )
        self.adapters = nn.ModuleList([LoRA(d) for _ in range(routers)])
        self.router = nn.Linear(d, routers)

        self.register_buffer("tau", torch.tensor(1.0))

    def add(self):
        dev = self.router.weight.device
        self.adapters.append(LoRA(self.attn.embed_dim).to(dev))
        k_old, d = self.router.out_features, self.router.in_features
        new = nn.Linear(d, k_old + 1, device=dev)
        with torch.no_grad():
            new.weight[:k_old], new.bias[:k_old] = (
                self.router.weight,
                self.router.bias,
            )
        self.router = new

    def forward(self, x):
        h, _ = self.attn(x, x, x)
        x = self.ln1(x + h)

        logits = self.router(x.mean(1)) / self.tau  # ← divide by τ
        g = torch.softmax(logits, -1)  # [B, K]

        a_sum = torch.zeros_like(x)
        for i, a in enumerate(self.adapters):
            a_sum += g[:, i, None, None] * a(x)

        return self.ln2(x + a_sum + self.ffn(x)), g  # also return g


class WorldModel(nn.Module):
    def __init__(self, d=64, layers=4, heads=4, max_pos=4096):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Parameter(torch.randn(max_pos, d))
        self.blocks = nn.ModuleList(
            [Block(d, heads, 1) for _ in range(layers)]
        )
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)

    def add_task(self):
        for b in self.blocks:
            b.add()

    def forward(self, seq, return_ent=False):
        x = self.tok(seq) + self.pos[: seq.size(1)]
        ent_sum = 0.0
        for b in self.blocks:
            x, g = b(x)  # g is [batch, n_adapter]
            if return_ent:
                kl = (g * (g + 1e-8).log()).sum(-1) + math.log(g.size(-1))
                ent_sum += kl

        if return_ent:
            return self.head(self.ln(x)), ent_sum.mean()  # logits, H
        else:
            return self.head(self.ln(x))


class Actor(nn.Module):
    def __init__(self, d_lat):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_lat, 128), nn.ReLU(), nn.Linear(128, MAX_ACT)
        )

    def forward(self, z):
        return F.softmax(self.net(z), -1)


class Critic(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, z):
        return self.v(z).squeeze(-1)


def fisher_diag(model, batch, chunk=64):
    diag = [
        torch.zeros_like(p, dtype=torch.float32, device="cpu")
        for p in model.parameters()
        if p.requires_grad
    ]

    # --- forward / backward in chunks ---------------------------------
    for i in range(0, batch.size(0), chunk):
        sub = batch[i: i + chunk].to(next(model.parameters()).device)
        loss = F.cross_entropy(
            model(sub[:, :-1]).view(-1, VOCAB),
            sub[:, 1:].reshape(-1),
            ignore_index=PAD,
            reduction="sum",
        )
        grads = torch.autograd.grad(
            loss, [p for p in model.parameters() if p.requires_grad]
        )
        for d, g in zip(diag, grads):
            d += g.detach().cpu().pow(2)

    # --- normalise and (optionally) cast to fp16 ----------------------
    N = (batch[:, 1:] != PAD).sum().item()
    for idx, d in enumerate(diag):  # <- enumerate fixes IndexError
        d.div_(N)  # keep operation in-place
        diag[idx] = d
        # or just: diag[idx] = d         # stay fp32 if RAM is fine

    return diag


def train_task(
    env_name,
    wm,
    actor,
    critic,
    replay,
    epochs=12,
    ctx=32,
    imag_h=15,
    lam=0.1,
    running_W=None,  # list[tensor]  –  θ*
    running_F=None,  # list[tensor]  –  diagonal Fisher
):
    env = make_atari(env_name, mode="noframeskip")
    obs, _ = env.reset()

    opt_wm = Adam(wm.parameters(), 4e-4)
    opt_act = Adam(actor.parameters(), 4e-4)
    opt_cri = Adam(critic.parameters(), 4e-4)

    ep = []
    losses = []

    pbar = tqdm(
        total=epochs,
        desc=env_name,
    )
    while len(losses) < epochs:
        ids = frame_to_ids(obs)  # ONE encoder call
        z = wm.tok(torch.tensor(ids, device=DEVICE)).mean(0, keepdim=True)

        a = torch.distributions.Categorical(actor(z)).sample().item()
        nxt, _, term, trunc, _ = env.step(a)
        ep.append((ids, a))

        if len(ep) >= ctx:
            seq = [
                torch.tensor(np.append(t, ACT_PAD + a_)) for t, a_ in ep[-ctx:]
            ]
            seq = torch.stack(seq)
            replay.add(seq)
            global_buf.append(seq)

        if term or trunc:  # KEEP this
            ep = []
            obs, _ = env.reset()
        else:
            obs = nxt

        if len(replay.b) < 256:  # skip gradient step, but buffer is growing
            continue

        cur = replay.sample(int(0.8 * 64))
        other = (
            random.sample(global_buf, int(0.2 * 64))
            if len(global_buf) >= 13
            else []
        )
        batch = torch.stack(cur + other).to(DEVICE)

        B, T, D = batch.shape

        inp = batch[:, :-1].reshape(B, -1)
        tgt = batch[:, 1:].reshape(B, -1)

        logits, kl = wm(inp, return_ent=True)  # kl is mean KL per batch
        ce = F.cross_entropy(
            logits.view(-1, VOCAB), tgt.reshape(-1), ignore_index=PAD
        )

        # imagination rollout
        last = inp[:, -N_PATCH:]
        z0 = wm.tok(last).mean(1)
        zs = [z0]
        v = []

        for _ in range(imag_h):
            pa = actor(zs[-1].detach())  # stop grad into Actor
            a_s = torch.distributions.Categorical(pa).sample()
            roll = torch.cat(
                [inp[:, -ctx * (N_PATCH + 1):], (ACT_PAD + a_s).unsqueeze(1)],
                1,
            )
            ntok = wm(roll)[:, -1].argmax(-1, keepdim=True)
            z_next = wm.tok(ntok).squeeze(1)  # keep grad for WM
            zs.append(z_next)
            v.append(critic(z_next.detach()))  # stop grad into Critic

        imag = torch.stack(v, 1)
        act_loss = -imag.mean()
        val_loss = F.mse_loss(
            torch.stack(v, 1), torch.zeros_like(torch.stack(v, 1))
        )

        # EWC penalty

        ewc_penalty = 0.0
        beta = 0.9
        entropy_pen = beta * kl

        if running_W is not None:  # skip for very first task
            for p, theta_star, F_diag in zip(
                wm.parameters(), running_W, running_F
            ):
                if p.requires_grad and p.shape == theta_star.shape:
                    theta_star_ = theta_star.to(p.device, dtype=p.dtype)
                    F_ = F_diag.to(p.device, dtype=p.dtype)
                    ewc_penalty += (F_ * (p - theta_star_).pow(2)).sum()

        loss = (
            ce
            + 0.1 * act_loss
            + 0.1 * val_loss
            + lam * ewc_penalty
            - entropy_pen
        )

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
            loss=f"{loss.item():.4f}",
            ce=f"{ce.item():.4f}",
            ewc=f"{lam*ewc_penalty:.4f}",
        )
        pbar.update(1)

    env.close()
    pbar.close()

    return ce


@torch.no_grad()
def eval_ce_chunked(wm, traj_buffer, max_len=4095):
    """
    Cross-entropy on an arbitrary-length token list, processed in windows
    no longer than `max_len` so it never exceeds wm.pos.size(0).
    """
    flat = torch.stack(traj_buffer).flatten().to(DEVICE)  # [T*25]

    with torch.no_grad():
        x = wm.tok(flat[:25].unsqueeze(0))  # one Phoenix frame
        g = []
        for b in wm.blocks:
            g.append(F.softmax(b.router(x.mean(1)), -1).cpu())

    print("Router probs per block:", torch.stack(g).mean(0))

    ce_sum, tok_sum = 0.0, 0

    for start in range(0, flat.size(0) - 1, max_len):
        end = min(start + max_len, flat.size(0) - 1)
        inp = flat[start:end].unsqueeze(0)  # [1, L]
        tgt = flat[start + 1: end + 1]  # [L]
        logits = wm(inp).view(-1, VOCAB)
        ce_sum += F.cross_entropy(logits, tgt, reduction="sum").item()
        tok_sum += tgt.numel()

    return ce_sum / tok_sum


TASKS = [
    "SpaceInvaders",
    "Assault",
    "DemonAttack",
    "AirRaid",
    "Atlantis",
    "BeamRider",
    "StarGunner",
    "Galaxian",
    "Solaris",
    "Zaxxon",
]

eval_task = "Phoenix"  # initial task for evaluation


# TASKS = [
#     "SpaceInvaders",
#     "SpaceInvaders",
#     "SpaceInvaders",
#     "SpaceInvaders",
#     "SpaceInvaders",
#     "SpaceInvaders",
#     "SpaceInvaders",
#     "SpaceInvaders",
#     "SpaceInvaders",
# ]


if __name__ == "__main__":
    wm = WorldModel().to(DEVICE)
    actor = Actor(64).to(DEVICE)
    critic = Critic(64).to(DEVICE)

    replay = Buffer(30000)
    global_buf = []

    running_W = None  # list[tensor]  –  θ*
    running_F = None  # list[tensor]  –  diagonal Fisher

    eval_buf = []
    # env_eval = make_atari("Phoenix", mode="noframeskip")
    env_eval = make_atari(eval_task, mode="noframeskip")
    obs, _ = env_eval.reset(seed=42)

    while len(eval_buf) < 4096:
        eval_buf.append(torch.tensor(frame_to_ids(obs)))
        obs, _, term, trunc, _ = env_eval.step(env_eval.action_space.sample())
        if term or trunc:
            obs, _ = env_eval.reset()
    env_eval.close()

    losses = []
    ce_evals = []
    seen = set()
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
            epochs=3,
            ctx=32,
            running_W=running_W,
            running_F=running_F,
        )

        losses.append(loss)
        ce_eval = eval_ce_chunked(wm, list(eval_buf))
        ce_evals.append(ce_eval)
        print(f"Eval on {eval_task}: ", ce_eval)

        gamma = 0.9
        k = min(256, len(replay.b))

        if k == 0:
            continue

        mb = torch.stack(replay.sample(k)).reshape(k, -1)
        new_F = fisher_diag(wm, mb)

        params = list(wm.parameters())

        if running_F is None:  # first snapshot
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
