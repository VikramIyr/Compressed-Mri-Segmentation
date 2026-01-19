import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import set_seed, build_brats_index, split_cases, Brats2DSliceDataset
from model_unet import UNet
from metrics import dice_binary_from_logits



def make_binary(mask: torch.Tensor) -> torch.Tensor:
    return (mask > 0).long()


def dice_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    
    probs = torch.sigmoid(logits)

    # Flatten per-sample
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


class CombinedBCEDiceLoss(nn.Module):


    def __init__(self, bce_weight: float = 0.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)

    def forward(self, logits: torch.Tensor, targets_float: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.bce_weight > 0:
            loss = loss + self.bce_weight * self.bce(logits, targets_float)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * dice_loss_from_logits(logits, targets_float)
        return loss


def get_loss_fn(args) -> nn.Module:
    
    if args.loss == "bce":
        return nn.BCEWithLogitsLoss()
    if args.loss == "dice":
        return CombinedBCEDiceLoss(bce_weight=0.0, dice_weight=1.0)
    if args.loss == "bce_dice":
        return CombinedBCEDiceLoss(bce_weight=args.bce_weight, dice_weight=args.dice_weight)

    raise ValueError(f"Unknown --loss {args.loss}. Choose from: bce, dice, bce_dice")



def train_one_epoch(model, loader, optim, device, loss_fn, log_every=50):
    model.train()
    total_loss = 0.0

    for step, (x, y, _) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        yb = make_binary(y).to(device, non_blocking=True)      
        y_float = yb.float().unsqueeze(1)                     

        optim.zero_grad(set_to_none=True)
        logits = model(x)                                     
        loss = loss_fn(logits, y_float)
        loss.backward()
        optim.step()

        total_loss += loss.item() * x.size(0)

        if step % log_every == 0:
            print(f"  step {step}/{len(loader)} | loss {loss.item():.4f}", flush=True)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_one_epoch(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n = 0

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        yb = make_binary(y).to(device, non_blocking=True)     
        y_float = yb.float().unsqueeze(1)                      

        logits = model(x)
        loss = loss_fn(logits, y_float)

        bsz = x.size(0)
        total_loss += loss.item() * bsz
        total_dice += dice_binary_from_logits(logits, yb) * bsz
        n += bsz

    return total_loss / n, total_dice / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=os.environ.get("BRATS_ROOT", "data"))
    p.add_argument("--modalities", type=str, default="flair")  # "flair" or "flair,t1ce"
    p.add_argument("--slice_stride", type=int, default=1)
    p.add_argument("--downsample", type=int, default=1)
    p.add_argument("--target_hw", type=int, nargs=2, default=[240, 240])

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=2)

    # NEW: loss options
    p.add_argument("--loss", type=str, default="bce", choices=["bce", "dice", "bce_dice"])
    p.add_argument("--bce_weight", type=float, default=0.5)
    p.add_argument("--dice_weight", type=float, default=0.5)

    p.add_argument("--run_dir", type=str, default="runs/baseline")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Data root:", args.data_root)
    print(f"Loss: {args.loss} (bce_w={args.bce_weight}, dice_w={args.dice_weight})")

    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    cases = build_brats_index(args.data_root, modalities=modalities)
    train_cases, val_cases = split_cases(cases, val_ratio=0.2, seed=args.seed)

    train_ds = Brats2DSliceDataset(
        train_cases,
        modalities=modalities,
        slice_stride=args.slice_stride,
        downsample=args.downsample,
        target_hw=tuple(args.target_hw),
        keep_empty=True,
        empty_ratio=0.25,
        seed=args.seed,
        cache_dir="cache",
    )

    val_ds = Brats2DSliceDataset(
        val_cases,
        modalities=modalities,
        slice_stride=args.slice_stride,
        downsample=args.downsample,
        target_hw=tuple(args.target_hw),
        keep_empty=True,
        empty_ratio=0.25,
        seed=args.seed + 1,
        cache_dir="cache",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = UNet(in_channels=len(modalities), out_channels=1, base=32).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # NEW: choose loss function
    loss_fn = get_loss_fn(args).to(device)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optim, device, loss_fn, log_every=50)
        va_loss, va_dice = eval_one_epoch(model, val_loader, device, loss_fn)

        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} | val loss {va_loss:.4f} | val dice {va_dice:.4f}")

        if va_dice > best_dice:
            best_dice = va_dice
            ckpt = {
                "model": model.state_dict(),
                "epoch": epoch,
                "best_dice": best_dice,
                "args": vars(args),
            }
            torch.save(ckpt, run_dir / "best.pt")
            print("Saved best checkpoint:", run_dir / "best.pt")

    print("Done. Best val dice:", best_dice)


if __name__ == "__main__":
    main()
