import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from .vqvae import VQVAE, RES, D_LAT
from ..utils.common import TORCH_DEVICE, VQVAE_CHECKPOINT

_VQVAE_SINGLETON: VQVAE | None = None


def get_vqvae() -> VQVAE:
    global _VQVAE_SINGLETON
    if _VQVAE_SINGLETON is None:
        vqvae = VQVAE().to(TORCH_DEVICE)
        vqvae.load_state_dict(load_file(VQVAE_CHECKPOINT, device=TORCH_DEVICE))
        vqvae.eval()
        for p in vqvae.parameters():
            p.requires_grad_(False)
        _VQVAE_SINGLETON = vqvae
    return _VQVAE_SINGLETON


vqvae = get_vqvae()


@torch.no_grad()
def frame_to_indices(frame_u8: np.ndarray, vqvae: VQVAE) -> np.ndarray:
    x = (
        torch.tensor(frame_u8, dtype=torch.float32, device=TORCH_DEVICE).permute(
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
def frames_to_indices(frames_u8: np.ndarray, vqvae: VQVAE) -> np.ndarray:
    N = len(frames_u8)
    x = (
        torch.from_numpy(frames_u8)
        .to(TORCH_DEVICE, dtype=torch.float32, non_blocking=True)
        .permute(0, 3, 1, 2)
        / 255.0
    )
    x = F.interpolate(x, (RES, RES), mode="bilinear", align_corners=False)
    with torch.amp.autocast("cuda"):
        lat = vqvae.enc(x)
        _, ids, _ = vqvae.vq(lat.flatten(0, 1))
    return ids.view(N, -1).cpu().numpy()
