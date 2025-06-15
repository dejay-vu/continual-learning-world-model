import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from ..common import D_LAT, RES, TORCH_DEVICE, VQVAE_CHECKPOINT
from .vqvae import VQVAE

_VQVAE_SINGLETON: VQVAE | None = None


def get_vqvae() -> VQVAE:
    global _VQVAE_SINGLETON
    if _VQVAE_SINGLETON is None:
        vqvae = VQVAE().to(TORCH_DEVICE)
        vqvae.load_state_dict(load_file(VQVAE_CHECKPOINT, device="cuda"))
        vqvae.eval()
        for p in vqvae.parameters():
            p.requires_grad_(False)
        _VQVAE_SINGLETON = vqvae
    return _VQVAE_SINGLETON


vqvae = get_vqvae()


@torch.no_grad()
def frame_to_indices(frame_u8: np.ndarray, vqvae: VQVAE) -> np.ndarray:
    x = (
        torch.tensor(
            frame_u8, dtype=torch.float32, device=TORCH_DEVICE
        ).permute(2, 0, 1)
        / 255.0
    )
    x = F.interpolate(
        x[None], (RES, RES), mode="bilinear", align_corners=False
    )
    _, ids, _ = vqvae.vq(vqvae.enc(x).reshape(-1, D_LAT))
    return ids.squeeze(0).cpu().numpy()


@torch.no_grad()
def frames_to_indices(
    frames_u8: np.ndarray,
    vqvae: VQVAE,
    batch_size: int = 2048,
    device: str = TORCH_DEVICE,
) -> torch.Tensor:
    """Convert a batch of RGB frames to discrete VQ-VAE codebook indices.

    Parameters
    ----------
    frames_u8 : np.ndarray
        Array of shape (T, H, W, 3) and dtype uint8.
    vqvae : VQVAE
        Pre-loaded VQ-VAE model.
    batch_size : int, default 2048
        Mini-batch size processed by the encoder (trades memory for speed).
    device : torch.device
        Device to which the pixel data is moved for processing.

    Returns
    -------
    torch.Tensor | np.ndarray
        Tensor/array of shape (T, N_PATCH) with dtype long / int64.
    """

    ids_list: list[torch.Tensor] = []

    for i in range(0, len(frames_u8), batch_size):
        batch = frames_u8[i : i + batch_size]

        # Move the *pixel* data to the *working* device (might differ from
        # final requested output device).
        x = torch.from_numpy(batch).to(device, non_blocking=True)
        x = x.to(dtype=torch.float16)
        x = x.permute(0, 3, 1, 2) / 255.0
        x = F.interpolate(x, (RES, RES), mode="bilinear", align_corners=False)

        with torch.autocast(device_type=device, dtype=x.dtype):
            lat = vqvae.enc(x)
            _, ids, _ = vqvae.vq(lat.flatten(0, 1))

        ids = ids.view(x.size(0), -1)  # (B, N_PATCH)
        ids_list.append(ids)

    ids_cat = torch.cat(ids_list, 0)

    return ids_cat.to(device, non_blocking=True)
