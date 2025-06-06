import torch
import torch.nn as nn
from flash_attn.layers.rotary import RotaryEmbedding
from flash_attn import flash_attn_qkvpacked_func
from .lora_layer import LoRA


class FlashAttentionBlock(nn.Module):
    """Multi‑head self‑attention + MLP with Flash‑Attention backend,
    plus LoRA adapters and a routing mechanism. Drop‑in replacement
    for your original Block, preserving add() and return of (x, g).
    """

    def __init__(
        self, d_model: int, n_head: int, routers: int = 1, drop: float = 0.1
    ):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.heads = n_head
        self.head_dim = d_model // n_head

        # QKV are packed together for flash‑attn: (B, L, 3, H, D)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Feed‑forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(drop),
        )

        # Layer norms
        self.ln1 = nn.LayerNorm(d_model)  # post-attn LN
        self.ln2 = nn.LayerNorm(d_model)  # between adapter+ffn and residual

        # Rotary positional embedding (applied inside forward)
        self.rope = RotaryEmbedding(dim=self.head_dim)
        self.attn_dropout = nn.Dropout(drop)

        # LoRA adapters and router
        self.adapters = nn.ModuleList([LoRA(d_model) for _ in range(routers)])
        self.router = nn.Linear(d_model, routers)

        # Temperature for routing softmax
        self.register_buffer("tau", torch.tensor(1.0))

    def add(self):
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

        if x.is_cuda:
            # ---- Fast path using flash-attn --------------------------------
            # Compute raw QKV on x (no pre-LN) to match original ordering
            qkv = self.qkv(x)  # (B, L, 3*D)
            qkv = qkv.view(bsz, seqlen, 3, self.heads, self.head_dim)

            # Rotary positional embedding (GPU-only path). The upstream
            # implementation from flash_attn fails when tensors are on CPU
            # because it unconditionally enters a CUDA context. We therefore
            # restrict its usage to CUDA tensors.
            qkv_rot = self.rope(
                qkv, None, seqlen_offset=0, num_heads_q=self.heads
            )
            qkv_rot = qkv_rot.to(torch.float16)

            # FlashAttention kernel (causal)
            y = flash_attn_qkvpacked_func(
                qkv_rot,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                causal=True,
                softmax_scale=None,
            )  # (B, L, H, D)

            y = y.to(x.dtype)

            y = y.reshape(bsz, seqlen, -1)
            y = self.o_proj(y)

        else:
            # ---- CPU fallback using scaled-dot-product attention ------------
            qkv = self.qkv(x)  # (B, L, 3*D)
            q, k, v = qkv.chunk(3, dim=-1)

            # Reshape to (B, H, L, D)
            q = q.view(bsz, seqlen, self.heads, self.head_dim).transpose(1, 2)
            k = k.view(bsz, seqlen, self.heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, seqlen, self.heads, self.head_dim).transpose(1, 2)

            # PyTorch 2.0+ provides an efficient (though non-flash) CPU
            # implementation of scaled-dot-product attention that is much more
            # memory-friendly than manually materialising the full attention
            # matrix. We leverage it here with causal masking enabled.
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True
            )  # (B, H, L, D)
            y = y.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
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
