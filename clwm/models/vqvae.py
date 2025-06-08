import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize

# -------------- config ----------------
D_LAT = 64  # latent dimension
RES = 84  # resolution of the input images
PATCH = 16  # patch size for the VQ-VAE
H16 = W16 = RES // PATCH  # 5×5 → 25 tokens
K = 128  # number of VQ codes
EMA = 0.9  # decay for the EMA in VectorQuantize


# --------- tiny CNN encoder/decoder ---
class Encoder(nn.Module):
    """Tiny CNN encoder used by the VQ-VAE."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
        )
        self.to_lat = nn.Linear(256, D_LAT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.flatten(2).transpose(1, 2)
        return self.to_lat(h)


class Decoder(nn.Module):
    """Mirror image of :class:`Encoder` for reconstruction."""

    def __init__(self) -> None:
        super().__init__()
        self.to_feat = nn.Linear(D_LAT, 256)
        self.deconv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        bsz, _, _ = z.shape
        z = self.to_feat(z).transpose(1, 2).view(bsz, 256, H16, W16)
        x = self.deconv(z)
        return F.interpolate(
            x, (RES, RES), mode="bilinear", align_corners=False
        )


# -------------- VQ-VAE -----------------
class VQVAE(nn.Module):
    """Lightweight VQ‑VAE used for frame tokenisation."""

    def __init__(self) -> None:
        super().__init__()
        self.enc = Encoder()
        self.vq = VectorQuantize(
            dim=D_LAT,
            codebook_size=K,
            decay=EMA,
            kmeans_init=True,
            threshold_ema_dead_code=2,
        )
        self.dec = Decoder()

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lat = self.enc(x)
        bsz, n_tok, _ = lat.shape

        z_q, ids, vq_loss = self.vq(lat.view(-1, D_LAT))
        z_q = z_q.view(bsz, n_tok, D_LAT)

        recon = self.dec(z_q).clamp(0, 1)
        rec_loss = F.mse_loss(recon, x)

        loss = rec_loss + vq_loss

        return recon, loss, rec_loss, ids
