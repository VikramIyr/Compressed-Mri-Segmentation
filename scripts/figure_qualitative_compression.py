import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import nibabel as nib



def percentile_clip_zscore(img: np.ndarray, p_low=1, p_high=99, eps=1e-8) -> np.ndarray:
    lo = np.percentile(img, p_low)
    hi = np.percentile(img, p_high)
    img = np.clip(img, lo, hi)
    mean = img.mean()
    std = img.std()
    return (img - mean) / (std + eps)


def downsample_then_resize(img: np.ndarray, downsample: int, target_hw=(240, 240)) -> np.ndarray:

    x = torch.from_numpy(img).float()[None, None]  

    
    if downsample > 1:
        h, w = x.shape[-2:]
        h2 = max(1, h // downsample)
        w2 = max(1, w // downsample)
        x = F.interpolate(x, size=(h2, w2), mode="bilinear", align_corners=False)

   
    x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
    return x[0, 0].cpu().numpy()


def resize_mask_nearest(mask: np.ndarray, downsample: int, target_hw=(240, 240)) -> np.ndarray:
    """
    Apply same spatial transform as image:
      - downsample by r (nearest, to preserve labels)
      - resize to target (nearest)
    """
    x = torch.from_numpy(mask.astype(np.float32))[None, None]  

    if downsample > 1:
        h, w = x.shape[-2:]
        h2 = max(1, h // downsample)
        w2 = max(1, w // downsample)
        x = F.interpolate(x, size=(h2, w2), mode="nearest")

    x = F.interpolate(x, size=target_hw, mode="nearest")
    return (x[0, 0].cpu().numpy() > 0.5).astype(np.uint8)



def _safe_div(num: float, den: float, eps: float = 1e-8) -> float:
    return float(num) / float(den + eps)


def compute_binary_metrics(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> dict:
    """
    gt, pred: uint8/bool arrays with values {0,1}
    Returns: dict with dice, iou, precision, recall
    """
    gt = (gt > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)

    tp = int(((gt == 1) & (pred == 1)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())

    dice = _safe_div(2 * tp, (2 * tp + fp + fn), eps)
    iou = _safe_div(tp, (tp + fp + fn), eps)
    precision = _safe_div(tp, (tp + fp), eps)
    recall = _safe_div(tp, (tp + fn), eps)

    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}



def build_model_from_train_py(in_channels: int):
    try:
        from src.model_unet import UNet  
    except ModuleNotFoundError:
        from model_unet import UNet
    return UNet(in_channels=in_channels, out_channels=1, base=32)


def load_checkpoint_to_model(model, ckpt_path: str, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
        else:
            if any(k.endswith(".weight") or k.endswith(".bias") for k in ckpt.keys()):
                state = ckpt
            else:
                raise RuntimeError(
                    f"Unrecognized checkpoint format in: {ckpt_path}\nKeys: {list(ckpt.keys())[:20]}"
                )
    else:
        raise RuntimeError(f"Unrecognized checkpoint object type: {type(ckpt)}")

    model_keys = set(model.state_dict().keys())
    state_keys = set(state.keys())
    if len(model_keys.intersection(state_keys)) == 0:
        for prefix in ["model.", "net.", "module."]:
            stripped = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            if stripped and len(model_keys.intersection(stripped.keys())) > 0:
                state = stripped
                break

    model.load_state_dict(state, strict=False)
    return model


@torch.no_grad()
def predict_mask(
    model,
    img_240: np.ndarray,
    device: torch.device,
    thresh: float = 0.5,
) -> np.ndarray:
    """
    img_240: [H,W] already normalized/resized
    returns binary mask [H,W]
    """
    x = torch.from_numpy(img_240).float()[None, None].to(device)  
    logits = model(x)
    if logits.ndim == 4:
        logits = logits[:, 0]
    prob = torch.sigmoid(logits)
    pred = (prob > thresh).to(torch.uint8)
    return pred[0].cpu().numpy()


def find_case_dir(data_root: str, case_id: str) -> str:
    case_dir = os.path.join(data_root, case_id)
    if not os.path.isdir(case_dir):
        raise FileNotFoundError(f"Case folder not found: {case_dir}")
    return case_dir


def load_modality_slice(case_dir: str, modality: str, slice_idx: int) -> np.ndarray:

    cand = None
    for fn in os.listdir(case_dir):
        lower = fn.lower()
        if lower.endswith((".nii", ".nii.gz")) and f"_{modality.lower()}" in lower:
            cand = os.path.join(case_dir, fn)
            break
    if cand is None:
        raise FileNotFoundError(f"Could not find modality '{modality}' NIfTI in {case_dir}")

    vol = nib.load(cand).get_fdata().astype(np.float32)  # [H,W,D]
    if slice_idx < 0 or slice_idx >= vol.shape[2]:
        raise IndexError(f"slice_idx={slice_idx} out of range for volume with D={vol.shape[2]}")
    return vol[:, :, slice_idx]


def load_seg_slice(case_dir: str, slice_idx: int) -> np.ndarray:
    cand = None
    for fn in os.listdir(case_dir):
        lower = fn.lower()
        if lower.endswith((".nii", ".nii.gz")) and ("_seg" in lower):
            cand = os.path.join(case_dir, fn)
            break
    if cand is None:
        raise FileNotFoundError(f"Could not find segmentation (_seg) NIfTI in {case_dir}")

    seg = nib.load(cand).get_fdata().astype(np.int16)
    if slice_idx < 0 or slice_idx >= seg.shape[2]:
        raise IndexError(f"slice_idx={slice_idx} out of range for seg with D={seg.shape[2]}")
    # whole-tumor: anything >0 becomes foreground
    return (seg[:, :, slice_idx] > 0).astype(np.uint8)



def overlay_contours(ax, img, gt_mask, pred_mask, title: str):
    ax.imshow(img, cmap="gray")
    gt_color = "lime"  
    pr_color = "red"   

    ax.contour(
        gt_mask.astype(float),
        levels=[0.5],
        linewidths=1.5,
        colors=[gt_color],
    )
    ax.contour(
        pred_mask.astype(float),
        levels=[0.5],
        linewidths=1.5,
        linestyles="--",
        colors=[pr_color],
    )

    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    case_dir = find_case_dir(args.data_root, args.case_id)
    raw_img = load_modality_slice(case_dir, args.modalities[0], args.slice_idx)  
    raw_gt = load_seg_slice(case_dir, args.slice_idx) 

    panels = [
        ("Full input (k=1,r=1)", args.ckpt_baseline, 1, 1),
        ("Slice skip (k=2,r=1)", args.ckpt_stride2, 2, 1),
        ("Downsample (k=1,r=2)", args.ckpt_down2, 1, 2),
        ("Skip+Down (k=2,r=2)", args.ckpt_s2d2, 2, 2),
    ]

    
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.5))
    fig.suptitle(f"{args.case_id} | axial slice {args.slice_idx}", fontsize=10, y=0.98)
    axes = axes.ravel()

    for ax, (title, ckpt, k, r) in zip(axes, panels):
        
        img = percentile_clip_zscore(raw_img)
        img_240 = downsample_then_resize(img, downsample=r, target_hw=(240, 240))
        gt_240 = resize_mask_nearest(raw_gt, downsample=r, target_hw=(240, 240))

        
        model = build_model_from_train_py(in_channels=1)
        model = load_checkpoint_to_model(model, ckpt, device="cpu")
        model.to(device).eval()

        pred_240 = predict_mask(model, img_240, device=device, thresh=args.thresh)

        
        metrics = compute_binary_metrics(gt_240, pred_240)
        title_with_metrics = (
            f"{title}\n"
            f"Dice {metrics['dice']:.3f} | IoU {metrics['iou']:.3f}\n"
            f"P {metrics['precision']:.3f} | R {metrics['recall']:.3f}"
        )

        overlay_contours(ax, img_240, gt_240, pred_240, title_with_metrics)

    import matplotlib.lines as mlines
    gt_line = mlines.Line2D([], [], color="lime", linewidth=1.8, label="GT (contour)")
    pr_line = mlines.Line2D([], [], color="red", linewidth=1.8, linestyle="--", label="Pred (contour)")
    fig.legend(handles=[gt_line, pr_line], loc="lower center", ncol=2, frameon=False, fontsize=9)

    out_path = os.path.join(args.out_dir, args.out_name)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--case_id", type=str, required=True, help="e.g., BraTS2021_00000")
    p.add_argument("--slice_idx", type=int, required=True, help="axial slice index (0..D-1)")
    p.add_argument("--modalities", type=str, nargs="+", default=["flair"])
    p.add_argument("--thresh", type=float, default=0.5)

    p.add_argument("--ckpt_baseline", type=str, required=True)
    p.add_argument("--ckpt_stride2", type=str, required=True)
    p.add_argument("--ckpt_down2", type=str, required=True)
    p.add_argument("--ckpt_s2d2", type=str, required=True)

    p.add_argument("--out_dir", type=str, default="figures")
    p.add_argument("--out_name", type=str, default="qualitative_compression.pdf")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
