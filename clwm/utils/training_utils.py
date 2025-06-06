import torch
import torch.nn.functional as F
from .common import ACTION_ID_START, VOCAB_SIZE, PAD_TOKEN


def split_cross_entropy(logits: torch.Tensor, tgt: torch.Tensor):
    """Split cross-entropy loss into image and action components."""
    flat_logits = logits.view(-1, VOCAB_SIZE)
    flat_tgt = tgt.reshape(-1)

    m_img = flat_tgt < ACTION_ID_START
    m_act = (flat_tgt >= ACTION_ID_START) & (flat_tgt != PAD_TOKEN)

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


def fisher_diagonal(model, batch: torch.Tensor, chunk: int = 64):
    diag = [
        torch.zeros_like(p, dtype=torch.float32, device="cpu")
        for p in model.parameters()
        if p.requires_grad
    ]

    for i in range(0, batch.size(0), chunk):
        sub = batch[i : i + chunk].to(next(model.parameters()).device)
        loss = F.cross_entropy(
            model(sub[:, :-1]).view(-1, VOCAB_SIZE),
            sub[:, 1:].reshape(-1),
            ignore_index=PAD_TOKEN,
            reduction="sum",
        )
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        for d, g in zip(diag, grads):
            if g is not None:
                d += g.detach().cpu().pow(2)

    N = (batch[:, 1:] != PAD_TOKEN).sum().item()
    for idx, d in enumerate(diag):
        d.div_(N)
        diag[idx] = d
    return diag
