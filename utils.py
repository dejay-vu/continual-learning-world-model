import random
import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TransformReward
from gymnasium.wrappers.vector import VectorizeTransformReward
import ale_py
from safetensors.torch import load_file
from vqvae import VQVAE, RES, D_LAT, K  # your tokenizer definition

MAX_ACT = 18  # max Atari action‑space size
ACT_PAD = K  # first action id (128)
PAD = K + MAX_ACT  # mask token (rarely used)
VOCAB = PAD + 1  # embedding size 147
CHECKPOINT = "vqvae_atari.safetensors"  # path to weights
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

gym.register_envs(ale_py)

_VQVAE_SINGLETON = None  # global VQVAE instance


def seed_everything(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_vqvae() -> VQVAE:
    global _VQVAE_SINGLETON

    if _VQVAE_SINGLETON is None:
        vqvae = VQVAE().to(DEVICE)
        vqvae.load_state_dict(load_file(CHECKPOINT, device=DEVICE))
        vqvae.eval()
        for p in vqvae.parameters():
            p.requires_grad_(False)
        _VQVAE_SINGLETON = vqvae

    return _VQVAE_SINGLETON


vqvae = get_vqvae()  # pre-load VQVAE instance


def split_ce(logits, tgt):
    """
    根据 token 数值范围拆分交叉熵:
        ─ 图像 token:   0  … ACT_PAD-1   (共 K   个)
        ─ 动作 token:   ACT_PAD … PAD-1  (共 MAX_ACT 个)
    返回 (ce_img, ce_act) ，均为 batch 内平均值。
    """
    flat_logits = logits.view(-1, VOCAB)  # [N, VOCAB]
    flat_tgt = tgt.reshape(-1)  # [N]

    # boolean masks
    m_img = flat_tgt < ACT_PAD
    m_act = (flat_tgt >= ACT_PAD) & (flat_tgt != PAD)

    ce_img = (
        F.cross_entropy(flat_logits[m_img], flat_tgt[m_img], reduction="mean")
        if m_img.any()
        else torch.tensor(0.0, device=logits.device)
    )

    ce_act = (
        F.cross_entropy(flat_logits[m_act], flat_tgt[m_act], reduction="mean")
        if m_act.any()
        else torch.tensor(0.0, device=logits.device)
    )

    return ce_img, ce_act


@torch.no_grad()
def frame_to_ids(frame_u8: np.ndarray, vqvae: VQVAE) -> np.ndarray:
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
    _, ids, _ = vqvae.vq(vqvae.enc(x).reshape(-1, D_LAT))

    return ids.squeeze(0).cpu().numpy()


@torch.no_grad()
def frames_to_ids(frames_u8: np.ndarray, vqvae: VQVAE) -> np.ndarray:
    """
    frames_u8 : uint8 (N,H,W,3)
    returns   : int64 (N, N_PATCH)
    """
    N = len(frames_u8)
    x = (
        torch.from_numpy(frames_u8)
        .to(DEVICE, dtype=torch.float32, non_blocking=True)  # (N,H,W,3)
        .permute(0, 3, 1, 2)
        / 255.0  # (N,3,H,W)
    )
    x = F.interpolate(x, (RES, RES), mode="bilinear", align_corners=False)
    with torch.amp.autocast("cuda"):  # cheap
        lat = vqvae.enc(x)  # (N,*,D)
        _, ids, _ = vqvae.vq(lat.flatten(0, 1))
    return ids.view(N, -1).cpu().numpy()  # (N,25)


def symlog(x):
    if isinstance(x, torch.Tensor):
        return torch.sign(x) * torch.log1p(torch.abs(x))
    else:  # float or np.ndarray
        return np.sign(x) * np.log1p(np.abs(x))


def symexp(x):
    if isinstance(x, torch.Tensor):
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
    else:
        return np.sign(x) * (np.expm1(np.abs(x)))


_I = torch.arange(-20, 21, dtype=torch.float32)
BINS = _I


def unimix(logits, p=0.01):
    probs = torch.softmax(logits, -1)
    return probs * (1 - p) + p / len(BINS)


def unimix_generic(logits, p=0.01):
    """Uniform-mix that works for any vocabulary size."""
    probs = torch.softmax(logits, -1)
    return probs * (1 - p) + p / logits.size(-1)


def twohot(y, bins=BINS):
    bins = BINS.to(y.device)  # <── match device
    y = torch.clamp(y, bins[0], bins[-1])
    k = torch.searchsorted(bins, y)
    k0 = torch.clamp(k - 1, 0, len(bins) - 2)
    k1 = k0 + 1
    w1 = (y - bins[k0]) / (bins[k1] - bins[k0])
    w0 = 1.0 - w1

    enc = torch.zeros((*y.shape, len(bins)), device=y.device)
    enc.scatter_(-1, k0.unsqueeze(-1), w0.unsqueeze(-1))
    enc.scatter_(-1, k1.unsqueeze(-1), w1.unsqueeze(-1))
    return enc


def expected_symlog(logits):
    """
    Expectation in SYMLOG space; returns scalar symlog value.
    """
    probs = unimix(logits)
    return (probs * BINS.to(logits.device)).sum(-1)


def expected_raw(logits):
    """
    Expectation converted back to raw reward/value domain.
    """
    return symexp(expected_symlog(logits))


def _wrap_reward(env):
    return TransformReward(env, lambda r: symlog(r))


def make_atari(
    name: str,
    *,
    frameskip: int = 4,  # 4 for benchmark, 1 for no-skip ablation
    sticky: bool = True,  # True → p_repeat = 0.25
    max_episode_steps: int = None,  # 5-minute limit
):
    base = f"ALE/{name}-v5"
    env = gym.make(
        base,
        max_episode_steps=max_episode_steps,
        frameskip=frameskip,
        repeat_action_probability=0.25 if sticky else 0.0,
        full_action_space=True,
        render_mode=None,
    )

    env = _wrap_reward(env)
    return env


def make_atari_vectorized(
    name: str,
    *,
    frameskip: int = 4,  # 4 for benchmark, 1 for no-skip ablation
    sticky: bool = True,  # True → p_repeat = 0.25
    max_episode_steps: int = None,  # 5-minute limit
    num_envs: int = 128,  # number of parallel environments
):
    base = f"ALE/{name}-v5"
    envs = gym.make_vec(
        base,
        num_envs=num_envs,
        vectorization_mode="async",
        wrappers=[_wrap_reward],
        max_episode_steps=max_episode_steps,
        frameskip=frameskip,
        repeat_action_probability=0.25 if sticky else 0.0,
        full_action_space=True,
        render_mode=None,
    )

    return envs


def fisher_diag(model, batch, chunk=64):
    diag = [
        torch.zeros_like(p, dtype=torch.float32, device="cpu")
        for p in model.parameters()
        if p.requires_grad
    ]

    # --- forward / backward in chunks ---------------------------------
    for i in range(0, batch.size(0), chunk):
        sub = batch[i : i + chunk].to(next(model.parameters()).device)
        loss = F.cross_entropy(
            model(sub[:, :-1]).view(-1, VOCAB),
            sub[:, 1:].reshape(-1),
            ignore_index=PAD,
            reduction="sum",
        )
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(
            loss, params, allow_unused=True  # ← key change
        )
        for d, g in zip(diag, grads):
            if g is not None:  # skip unused (e.g. r_head)
                d += g.detach().cpu().pow(2)

    # --- normalise and (optionally) cast to fp16 ----------------------
    N = (batch[:, 1:] != PAD).sum().item()
    for idx, d in enumerate(diag):  # <- enumerate fixes IndexError
        d.div_(N)  # keep operation in-place
        diag[idx] = d
        # or just: diag[idx] = d         # stay fp32 if RAM is fine

    return diag


@torch.no_grad()
def build_eval_seq(
    wm,
    actor,
    env_name: str,
    *,
    ctx: int = 32,
    n_seq: int = 128,
    num_envs: int = 128,
):
    """
    Run the current agent in a vectorised Atari environment and collect
    `n_seq` sequences of length `ctx`.  Nothing is written to the replay buffer.
    """
    envs = make_atari_vectorized(env_name, num_envs=num_envs)
    obs, _ = envs.reset(seed=123)

    seqs: list[torch.Tensor] = []
    eps = [[] for _ in range(num_envs)]  # running episode traces

    actor.eval()
    wm.eval()

    while len(seqs) < n_seq:
        # latent state for a batch of frames
        ids_batch = frames_to_ids(obs, vqvae)  # (E,25)
        z_batch = wm.tok(torch.tensor(ids_batch, device=DEVICE)).mean(
            1
        )  # (E,d)
        actions = (
            torch.distributions.Categorical(actor(z_batch))
            .sample()
            .cpu()
            .numpy()
        )  # (E,)

        # book-keep one environment at a time
        for e in range(num_envs):
            eps[e].append((ids_batch[e], int(actions[e])))

            if len(eps[e]) >= ctx:  # we’ve got a full context window
                seq = torch.stack(
                    [
                        torch.tensor(np.append(t, ACT_PAD + a_))
                        for t, a_ in eps[e][-ctx:]
                    ]
                )
                seqs.append(seq)
                if len(seqs) >= n_seq:
                    break

        # env step ---------------------------------------------------------
        obs, _, term, trunc, _ = envs.step(actions)
        done = np.logical_or(term, trunc)
        for e, d in enumerate(done):
            if d:
                eps[e].clear()  # reset per-env trace

    envs.close()
    return seqs  # list[Tensor(ctx,26)]


@torch.no_grad()
def eval_on_sequences(wm, seq_batch):
    """
    Unchanged apart from the docstring – still evaluates CE on a batch of
    sequences that were (now) gathered with `build_eval_seq`.
    """
    batch = torch.stack(seq_batch).to(DEVICE)  # (B,ctx,26)
    B = batch.size(0)
    inp = batch[:, :-1].reshape(B, -1)
    tgt = batch[:, 1:].reshape(B, -1)

    logits = wm(inp)
    ce_img, ce_act = split_ce(logits, tgt)

    print(
        f"eval_img_ce={ce_img.item():.4f}  " f"eval_act_ce={ce_act.item():.4f}"
    )
    return 0.4 * ce_img + 0.6 * ce_act


@torch.no_grad()
def eval_policy(
    actor,
    wm,
    env_name: str,
    *,
    episodes: int = 128,
    num_envs: int = 128,
):
    """
    Play `episodes` full games in a vectorised Atari environment and return
    the mean *raw* score.  The env uses symlog rewards internally, so we
    convert them back to raw with `symexp`.
    """
    envs = make_atari_vectorized(env_name, num_envs=num_envs)
    obs, _ = envs.reset(seed=0)

    ep_scores = np.zeros(num_envs, dtype=np.float32)
    ep_lengths = np.zeros(num_envs, dtype=np.int32)
    finished = []

    while len(finished) < episodes:
        # choose actions
        ids_batch = frames_to_ids(obs, vqvae)
        z_batch = wm.tok(torch.tensor(ids_batch, device=DEVICE)).mean(1)
        actions = (
            torch.distributions.Categorical(actor(z_batch))
            .sample()
            .cpu()
            .numpy()
        )

        # env step
        obs, r, term, trunc, _ = envs.step(actions)
        ep_scores += symexp(r)
        ep_lengths += 1
        done = np.logical_or(term, trunc)

        if done.any():  # auto-reset already happened
            for e, d in enumerate(done):
                if d:
                    finished.append((float(ep_scores[e]), int(ep_lengths[e])))
                    ep_scores[e] = 0.0
                    ep_lengths[e] = 0

    envs.close()

    mean_score, mean_len = map(np.mean, zip(*finished[:episodes]))
    print(f"score {mean_score:.1f}   frames/ep {mean_len:.0f}")
    return float(mean_score)
