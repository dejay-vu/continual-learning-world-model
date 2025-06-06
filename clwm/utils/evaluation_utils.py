import numpy as np
import torch
from ..env.atari_envs import make_atari_vectorized_envs
from .training_utils import split_cross_entropy
from .common import TORCH_DEVICE, ACTION_ID_START, symexp
from ..models.vqvae_utils import frames_to_indices, vqvae


@torch.no_grad()
def build_evaluation_sequences(
    wm,
    actor,
    env_name: str,
    *,
    ctx: int = 32,
    n_seq: int = 128,
    num_envs: int = 128,
):
    envs = make_atari_vectorized_envs(env_name, num_envs=num_envs)
    obs, _ = envs.reset(seed=123)

    seqs: list[torch.Tensor] = []
    eps = [[] for _ in range(num_envs)]

    actor.eval()
    wm.eval()

    while len(seqs) < n_seq:
        ids_batch = frames_to_indices(obs, vqvae)
        z_batch = wm.tok(torch.tensor(ids_batch, device=TORCH_DEVICE)).mean(1)
        actions = (
            torch.distributions.Categorical(actor(z_batch))
            .sample()
            .cpu()
            .numpy()
        )
        for e in range(num_envs):
            eps[e].append((ids_batch[e], int(actions[e])))
            if len(eps[e]) >= ctx:
                seq = torch.stack(
                    [
                        torch.tensor(np.append(t, ACTION_ID_START + a_))
                        for t, a_ in eps[e][-ctx:]
                    ]
                )
                seqs.append(seq)
                if len(seqs) >= n_seq:
                    break
        obs, _, term, trunc, _ = envs.step(actions)
        done = np.logical_or(term, trunc)
        for e, d in enumerate(done):
            if d:
                eps[e].clear()
    envs.close()
    return seqs


@torch.no_grad()
def evaluate_on_sequences(wm, seq_batch):
    batch = torch.stack(seq_batch).to(TORCH_DEVICE)
    B = batch.size(0)
    inp = batch[:, :-1].reshape(B, -1)
    tgt = batch[:, 1:].reshape(B, -1)

    logits = wm(inp)
    ce_img, ce_act = split_cross_entropy(logits, tgt)
    print(f"eval_img_ce={ce_img.item():.4f}  eval_act_ce={ce_act.item():.4f}")
    return 0.4 * ce_img + 0.6 * ce_act


@torch.no_grad()
def evaluate_policy(
    actor, wm, env_name: str, *, episodes: int = 128, num_envs: int = 128
):
    envs = make_atari_vectorized_envs(env_name, num_envs=num_envs)
    obs, _ = envs.reset(seed=0)

    ep_scores = np.zeros(num_envs, dtype=np.float32)
    ep_lengths = np.zeros(num_envs, dtype=np.int32)
    finished = []

    while len(finished) < episodes:
        ids_batch = frames_to_indices(obs, vqvae)
        z_batch = wm.tok(torch.tensor(ids_batch, device=TORCH_DEVICE)).mean(1)
        actions = (
            torch.distributions.Categorical(actor(z_batch))
            .sample()
            .cpu()
            .numpy()
        )
        obs, r, term, trunc, _ = envs.step(actions)
        ep_scores += symexp(r)
        ep_lengths += 1
        done = np.logical_or(term, trunc)
        if done.any():
            for e, d in enumerate(done):
                if d:
                    finished.append((float(ep_scores[e]), int(ep_lengths[e])))
                    ep_scores[e] = 0.0
                    ep_lengths[e] = 0
    envs.close()
    mean_score, mean_len = map(np.mean, zip(*finished[:episodes]))
    print(f"score {mean_score:.1f}   frames/ep {mean_len:.0f}")
    return float(mean_score)
