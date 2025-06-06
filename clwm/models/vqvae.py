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
    def __init__(self):
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

    def forward(self, x):
        h = self.conv(x)  # [B,256,6,6]
        h = h.flatten(2).transpose(1, 2)  # [B,36,256]
        return self.to_lat(h)  # [B,36,D]


class Decoder(nn.Module):
    def __init__(self):
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

    def forward(self, z):
        B, N, _ = z.shape
        z = self.to_feat(z).transpose(1, 2).view(B, 256, H16, W16)
        x = self.deconv(z)
        return F.interpolate(
            x, (RES, RES), mode="bilinear", align_corners=False
        )


# -------------- VQ-VAE -----------------
class VQVAE(nn.Module):
    def __init__(self):
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

    def forward(self, x):
        lat = self.enc(x)  # [B,N,D]
        B, N, _ = lat.shape

        z_q, ids, vq_loss = self.vq(lat.view(-1, D_LAT))
        z_q = z_q.view(B, N, D_LAT)

        # -------- decode & pixel loss -------------
        recon = self.dec(z_q).clamp(0, 1)
        rec_loss = F.mse_loss(recon, x)  # or BCE if you prefer

        # -------- total VQVAE loss ---------------
        loss = rec_loss + vq_loss  # vq_loss already includes commitment

        # -------- logging ------------------------
        return recon, loss, rec_loss, ids  # keep signature for trainer
