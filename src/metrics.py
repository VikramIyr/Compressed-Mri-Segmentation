import torch


@torch.no_grad()
def dice_binary_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:

    probs = torch.sigmoid(logits)
    pred = (probs > 0.5).float()
    tgt = target.float().unsqueeze(1)  # (B,1,H,W)

    inter = (pred * tgt).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + tgt.sum(dim=(1,2,3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()
