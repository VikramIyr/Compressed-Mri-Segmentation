import os
import csv
import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

import json
import time
from datetime import datetime


from datasets import build_brats_index, split_cases, Brats2DSliceDataset


# -------------------------
# Metrics
# -------------------------

def dice_score_binary(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return float((2.0 * inter + eps) / (denom + eps))


def hd95_binary_2d(pred: np.ndarray, gt: np.ndarray) -> float:

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")

    
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt
    except Exception:
        
        return hd95_binary_2d_kdtree(pred, gt)


    pred_b = np.logical_xor(pred, binary_erosion(pred))
    gt_b = np.logical_xor(gt, binary_erosion(gt))

    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 0.0
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return float("inf")

    dt_gt = distance_transform_edt(~gt_b)
    d_pred_to_gt = dt_gt[pred_b]

    dt_pred = distance_transform_edt(~pred_b)
    d_gt_to_pred = dt_pred[gt_b]

    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred], axis=0)
    return float(np.percentile(all_d, 95))


def hd95_binary_2d_kdtree(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")

    try:
        from scipy.ndimage import binary_erosion
        from scipy.spatial import cKDTree
    except Exception:
        return float("nan")

    pred_b = np.logical_xor(pred, binary_erosion(pred))
    gt_b = np.logical_xor(gt, binary_erosion(gt))

    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 0.0
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return float("inf")

    p_pts = np.column_stack(np.where(pred_b))
    g_pts = np.column_stack(np.where(gt_b))

    t_g = cKDTree(g_pts)
    t_p = cKDTree(p_pts)

    d_p_to_g, _ = t_g.query(p_pts, k=1)
    d_g_to_p, _ = t_p.query(g_pts, k=1)

    all_d = np.concatenate([d_p_to_g, d_g_to_p], axis=0)
    return float(np.percentile(all_d, 95))



def build_model_from_train_py(in_channels: int):
    """
    Tries a few common patterns depending on how your train.py was written.
    Adjust here if your model builder has a different name.
    """
    
    try:
        from train import build_model 
        return build_model(in_channels=in_channels)
    except Exception:
        pass

    
    try:
        from train import UNet2D 
        return UNet2D(in_channels=in_channels)
    except Exception:
        pass

    
    try:
        from train import UNet  # type: ignore
        return UNet(in_channels=in_channels)
    except Exception:
        pass

    raise RuntimeError(
        "Could not import a model builder from train.py.\n"
        "Open src/train.py and tell me what your model class/function is called "
        "(e.g., UNet2D, build_model, make_model), and I’ll patch eval.py accordingly."
    )


def load_checkpoint_to_model(model, ckpt_path: str, device="cpu"):

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)


    
    if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        
        for key in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
        else:
            
            if any(k.endswith(".weight") or k.endswith(".bias") for k in ckpt.keys()):
                state = ckpt
            else:
                raise RuntimeError(f"Unrecognized checkpoint format in: {ckpt_path}\nKeys: {list(ckpt.keys())[:20]}")
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

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded ckpt: {ckpt_path}")
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")

    return ckpt




def save_summary(args, mean_dice, mean_hd, mean_volerr, num_cases, num_slices, per_case_csv_path):
    os.makedirs(args.out_dir, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"k{args.slice_stride}_r{args.downsample}_{'_'.join(args.modalities)}_{args.split}"

    summary = {
        "run_id": run_id,
        "tag": tag,
        "ckpt": args.ckpt,
        "split": args.split,
        "modalities": args.modalities,
        "slice_stride": args.slice_stride,
        "downsample": args.downsample,
        "thresh": args.thresh,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_cases": int(num_cases),
        "num_slices": int(num_slices),
        "mean_dice": float(mean_dice),
        "mean_hd95_finite_mean": float(mean_hd),
        "mean_vol_err": float(mean_volerr),
        "per_case_csv": per_case_csv_path,
        "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
    }

    json_path = os.path.join(args.out_dir, f"summary_{tag}_{run_id}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON to: {json_path}")

    
    csv_path = os.path.join(args.out_dir, "summary.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(list(summary.keys()))
        w.writerow([summary[k] for k in summary.keys()])
    print(f"Appended summary row to: {csv_path}")


@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Build cases + split
    cases = build_brats_index(args.data_root, modalities=args.modalities)
    train_cases, val_cases = split_cases(cases, val_ratio=args.val_ratio, seed=args.seed)

    if args.split == "train":
        eval_cases = train_cases
    elif args.split == "val":
        eval_cases = val_cases
    else:
        raise ValueError("--split must be train or val")

    
    ds = Brats2DSliceDataset(
        eval_cases,
        modalities=args.modalities,
        slice_axis=2,
        slice_stride=args.slice_stride,
        downsample=args.downsample,
        target_hw=(240, 240),
        keep_empty=True,
        empty_ratio=1.0,   
        seed=args.seed,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    
    model = build_model_from_train_py(in_channels=len(args.modalities))
    model.to(device)
    model.eval()
    _ = load_checkpoint_to_model(model, args.ckpt)

    
    per_case_dice = defaultdict(list)
    per_case_hd95 = defaultdict(list)
    per_case_pred_vol = defaultdict(float)
    per_case_gt_vol = defaultdict(float)

    for x, y, meta in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)  
        
        out = model(x)

       
        if out.ndim == 4:
            logits = out[:, 0]
        elif out.ndim == 3:
            logits = out
        else:
            raise RuntimeError(f"Unexpected model output shape: {tuple(out.shape)}")

        prob = torch.sigmoid(logits)
        pred = (prob > args.thresh).to(torch.uint8)

        
        pred_np = pred.cpu().numpy().astype(bool)
        gt_np = (y > 0).to(torch.uint8).cpu().numpy().astype(bool)

        
        case_ids = meta["case_id"]
        if isinstance(case_ids, (list, tuple)):
            case_ids_list = list(case_ids)
        else:
            
            case_ids_list = [case_ids] * pred_np.shape[0]

        for i in range(pred_np.shape[0]):
            cid = case_ids_list[i]

            d = dice_score_binary(pred_np[i], gt_np[i])
            h = hd95_binary_2d(pred_np[i], gt_np[i])

            per_case_dice[cid].append(d)
            per_case_hd95[cid].append(h)

            
            per_case_pred_vol[cid] += float(pred_np[i].sum()) * args.slice_stride
            per_case_gt_vol[cid] += float(gt_np[i].sum()) * args.slice_stride

    
    rows = []
    dice_list = []
    hd_list = []
    volerr_list = []

    for cid in sorted(per_case_dice.keys()):
        cd = float(np.mean(per_case_dice[cid])) if len(per_case_dice[cid]) else float("nan")

        
        hvals = np.array(per_case_hd95[cid], dtype=np.float64)
        finite = np.isfinite(hvals)
        if finite.any():
            ch = float(np.mean(hvals[finite]))
        else:
            ch = float("inf")

        pv = per_case_pred_vol[cid]
        gv = per_case_gt_vol[cid]
        vol_err = float(abs(pv - gv) / (gv + 1e-7))

        rows.append((cid, cd, ch, vol_err, pv, gv, len(per_case_dice[cid])))

        dice_list.append(cd)
        hd_list.append(ch)
        volerr_list.append(vol_err)

    mean_dice = float(np.nanmean(dice_list)) if len(dice_list) else float("nan")
    mean_hd = float(np.nanmean(hd_list)) if len(hd_list) else float("nan")
    mean_volerr = float(np.nanmean(volerr_list)) if len(volerr_list) else float("nan")

    print("\n=== EVAL RESULTS ===")
    print(f"Split: {args.split} | cases: {len(rows)} | slices: {len(ds)}")
    print(f"Dice (mean over cases): {mean_dice:.4f}")
    print(f"HD95 (mean over cases, finite only): {mean_hd:.3f}")
    print(f"VolErr (mean over cases): {mean_volerr:.4f}")

    # Save CSV
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, args.out_name)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "dice_mean", "hd95_mean", "vol_err", "pred_vol_proxy", "gt_vol_proxy", "num_slices_eval"])
        w.writerows(rows)

    print(f"Saved per-case metrics to: {out_csv}")

    
    save_summary(
        args=args,
        mean_dice=mean_dice,
        mean_hd=mean_hd,
        mean_volerr=mean_volerr,
        num_cases=len(rows),
        num_slices=len(ds),
        per_case_csv_path=out_csv,
)



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (e.g., runs/baseline/best.pt)")
    p.add_argument("--data_root", type=str, default="data", help="BraTS folder containing BraTS2021_XXXXX directories")
    p.add_argument("--modalities", type=str, nargs="+", default=["flair"], help="e.g., --modalities flair t1ce")
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--slice_stride", type=int, default=1)
    p.add_argument("--downsample", type=int, default=1)
    p.add_argument("--thresh", type=float, default=0.5)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--out_dir", type=str, default="runs/eval")
    p.add_argument("--out_name", type=str, default="metrics.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
