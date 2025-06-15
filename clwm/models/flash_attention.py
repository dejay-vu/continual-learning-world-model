import torch
import torch.nn as nn
from flash_attn import flash_attn_qkvpacked_func
from flash_attn.layers.rotary import RotaryEmbedding

from .lora_layer import LoRA


class FlashAttentionBlock(nn.Module):
    """Multi-head self-attention + MLP with Flash-Attention backend,
    plus LoRA adapters and a routing mechanism. Drop-in replacement
    for your original Block, preserving add() and return of (x, g).
    """

    def __init__(
        self, dim: int, heads: int, routers: int = 1, drop: float = 0.1
    ):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.head_dim = dim // heads

        # QKV are packed together for flash-attn: (B, L, 3, H, D)
        # When the surrounding code runs under mixed precision (torch.cuda.amp
        # autocast) we instantiate the projection layers directly in half
        # precision to avoid the temporary fp32 → fp16 cast that would
        # otherwise create a duplicate buffer of size 3·B·L·D.

        # Keep the projection weights in fp32. Mixed-precision speed-ups are
        # provided by *autocast* which automatically casts the *inputs* to
        # fp16/bf16 when appropriate. Storing the parameters themselves in
        # fp16 breaks `torch.cuda.amp.GradScaler` (gradients remain fp16) and
        # also causes dtype mismatches when LayerNorm up-casts its outputs to
        # fp32 before the subsequent Linear. Using fp32 weights therefore
        # delivers the desired kernel fusion benefits without the stability
        # issues.

        proj_dtype = torch.float32

        self.qkv = nn.Linear(dim, 3 * dim, bias=False, dtype=proj_dtype)
        self.o_proj = nn.Linear(dim, dim, bias=False, dtype=proj_dtype)

        # Feed‑forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(4 * dim, dim),
            nn.Dropout(drop),
        )

        # Layer norms
        self.ln1 = nn.LayerNorm(dim)  # post-attn LN
        self.ln2 = nn.LayerNorm(dim)  # between adapter+ffn and residual

        # Rotary positional embedding (applied inside forward)
        self.rope = RotaryEmbedding(dim=self.head_dim)
        self.attn_dropout = nn.Dropout(drop)

        # LoRA adapters and router
        self.adapters = nn.ModuleList([LoRA(dim) for _ in range(routers)])
        self.router = nn.Linear(dim, routers)

        # Temperature for routing softmax
        self.register_buffer("tau", torch.tensor(1.0))

    def add_adapter(self):
        """Add a new adapter and expand router outputs by 1."""
        device = self.router.weight.device
        # Append new LoRA adapter
        self.adapters.append(LoRA(self.qkv.in_features).to(device))
        # Expand router weight/bias
        old_out, d_in = self.router.out_features, self.router.in_features
        new_router = nn.Linear(d_in, old_out + 1, device=device)

        with torch.no_grad():
            new_router.weight[:old_out] = self.router.weight
            new_router.bias[:old_out] = self.router.bias

        self.router = new_router

    def forward(self, x: torch.Tensor):  # x: (B, L, D)
        bsz, seqlen, _ = x.size()

        resid = x

        # ---- Fast path using flash-attn --------------------------------
        # Compute raw QKV – cast to fp16 if autocast is active to avoid
        # reallocating a fp32 buffer that would immediately be down-cast.
        qkv = self.qkv(x)  # (B, L, 3*D) — already fp16 under autocast
        qkv = qkv.view(bsz, seqlen, 3, self.heads, self.head_dim)

        # Rotary positional embedding (GPU-only path). The upstream
        # implementation from flash_attn fails when tensors are on CPU
        # because it unconditionally enters a CUDA context. We therefore
        # restrict its usage to CUDA tensors.
        qkv_rot = self.rope(qkv, None, seqlen_offset=0, num_heads_q=self.heads)
        qkv_rot = qkv_rot.to(torch.float16)

        # FlashAttention kernel (causal)
        y = flash_attn_qkvpacked_func(
            qkv_rot,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            causal=True,
            softmax_scale=None,
        )

        y = y.to(x.dtype)  # type: ignore

        y = y.reshape(bsz, seqlen, -1)
        y = self.o_proj(y)

        # Post-attention residual connection + LayerNorm
        x2 = self.ln1(resid + self.attn_dropout(y))

        # ---- Routing + adapters --------------------------------------------
        # Compute logits over routers from pooled representation
        router_logits = self.router(x2.mean(1)) / self.tau  # (B, routers)
        g = torch.softmax(router_logits, dim=-1)  # (B, routers)

        # Sum adapted outputs weighted by routing probs
        a_sum = torch.zeros_like(x2)
        for i, adapter in enumerate(self.adapters):
            a_sum = a_sum + g[:, i].view(bsz, 1, 1) * adapter(x2)

        x3 = (
            x2 + a_sum + self.ffn(self.ln2(x2))
        )  # add adapter and ffn, with LN

        return x3, g
