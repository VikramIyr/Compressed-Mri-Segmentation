# scripts/make_figures.py
# Usage examples:
#   python scripts/make_figures.py --summary runs/eval/summary.csv --out_dir figures
#   python scripts/make_figures.py --summary runs/eval/summary.csv --out_dir figures --split val --modality flair
#
# What it makes (as PNGs):
#   1) dice_vs_compression.png
#   2) hd95_vs_compression.png
#   3) volerr_vs_compression.png
#   4) tradeoff_dice_vs_hd95.png
#
# Compression factor used (simple + report-friendly):
#   compression = slice_stride * (downsample ** 2)
# Interprets downsample as spatial downsampling in both H and W.

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=str, default="runs/eval/summary.csv")
    ap.add_argument("--out_dir", type=str, default="figures")
    ap.add_argument("--split", type=str, default=None, help="train or val (optional filter)")
    ap.add_argument("--modality", type=str, default=None, help="e.g. flair (optional filter)")
    ap.add_argument("--ckpt_contains", type=str, default=None, help="filter rows where ckpt path contains this substring")
    ap.add_argument("--title", type=str, default=None, help="optional title prefix for plots")
    args = ap.parse_args()

    if not os.path.exists(args.summary):
        raise FileNotFoundError(f"Summary CSV not found: {args.summary}")

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.summary)

    # --- normalize columns (in case you renamed keys slightly) ---
    # expected columns from the patch: modalities, slice_stride, downsample, split, mean_dice, mean_hd95_finite_mean, mean_vol_err
    if "mean_hd95_finite_mean" not in df.columns and "mean_hd" in df.columns:
        df["mean_hd95_finite_mean"] = df["mean_hd"]
    if "mean_vol_err" not in df.columns and "mean_volerr" in df.columns:
        df["mean_vol_err"] = df["mean_volerr"]

    # modalities column might be a string representation of a list, e.g. "['flair']"
    if "modalities" in df.columns:
        df["modalities_str"] = df["modalities"].astype(str)
    else:
        df["modalities_str"] = ""

    # optional filters
    if args.split is not None and "split" in df.columns:
        df = df[df["split"].astype(str) == str(args.split)]
    if args.modality is not None:
        # keep rows where modalities contains the modality string
        df = df[df["modalities_str"].str.contains(args.modality)]
    if args.ckpt_contains is not None and "ckpt" in df.columns:
        df = df[df["ckpt"].astype(str).str.contains(args.ckpt_contains)]

    if len(df) == 0:
        raise RuntimeError("No rows left after filtering. Check --split/--modality/--ckpt_contains.")

    # numeric columns
    for c in ["slice_stride", "downsample", "mean_dice", "mean_hd95_finite_mean", "mean_vol_err"]:
        if c in df.columns:
            df[c] = df[c].apply(safe_float)

    # compression factor
    df["compression"] = df["slice_stride"] * (df["downsample"] ** 2)

    # x-axis label: show configuration clearly
    def cfg_label(r):
        s = int(r["slice_stride"]) if np.isfinite(r["slice_stride"]) else r["slice_stride"]
        d = int(r["downsample"]) if np.isfinite(r["downsample"]) else r["downsample"]
        return f"s{s}_d{d}"

    df["cfg"] = df.apply(cfg_label, axis=1)

    # For plots: group duplicates (e.g., multiple runs with same cfg) by mean
    agg = (
        df.groupby(["compression", "cfg"], as_index=False)
          .agg(
              mean_dice=("mean_dice", "mean"),
              mean_hd95=("mean_hd95_finite_mean", "mean"),
              mean_volerr=("mean_vol_err", "mean"),
              n=("mean_dice", "count"),
          )
          .sort_values("compression")
    )

    title_prefix = (args.title + " — ") if args.title else ""

    # -------------------------
    # Plot 1: Dice vs Compression
    # -------------------------
    plt.figure()
    plt.plot(agg["compression"], agg["mean_dice"], marker="o")
    plt.xlabel("Compression factor = slice_stride × downsample²")
    plt.ylabel("Mean Dice (over cases)")
    plt.title(f"{title_prefix}Dice vs Compression")
    plt.xticks(agg["compression"], agg["cfg"], rotation=30, ha="right")
    plt.tight_layout()
    out1 = os.path.join(args.out_dir, "dice_vs_compression.png")
    plt.savefig(out1, dpi=200)
    plt.close()

    # -------------------------
    # Plot 2: HD95 vs Compression
    # -------------------------
    plt.figure()
    plt.plot(agg["compression"], agg["mean_hd95"], marker="o")
    plt.xlabel("Compression factor = slice_stride × downsample²")
    plt.ylabel("Mean HD95 (finite-only mean, per case)")
    plt.title(f"{title_prefix}HD95 vs Compression")
    plt.xticks(agg["compression"], agg["cfg"], rotation=30, ha="right")
    plt.tight_layout()
    out2 = os.path.join(args.out_dir, "hd95_vs_compression.png")
    plt.savefig(out2, dpi=200)
    plt.close()

    # -------------------------
    # Plot 3: Volume error vs Compression
    # -------------------------
    plt.figure()
    plt.plot(agg["compression"], agg["mean_volerr"], marker="o")
    plt.xlabel("Compression factor = slice_stride × downsample²")
    plt.ylabel("Mean Volume Error (proxy)")
    plt.title(f"{title_prefix}Volume Error vs Compression")
    plt.xticks(agg["compression"], agg["cfg"], rotation=30, ha="right")
    plt.tight_layout()
    out3 = os.path.join(args.out_dir, "volerr_vs_compression.png")
    plt.savefig(out3, dpi=200)
    plt.close()

    # -------------------------
    # Plot 4: Trade-off (Dice vs HD95)
    # -------------------------
    plt.figure()
    plt.scatter(agg["mean_hd95"], agg["mean_dice"])
    for _, r in agg.iterrows():
        plt.annotate(r["cfg"], (r["mean_hd95"], r["mean_dice"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Mean HD95 (finite-only mean, per case)")
    plt.ylabel("Mean Dice (over cases)")
    plt.title(f"{title_prefix}Trade-off: Dice vs HD95")
    plt.tight_layout()
    out4 = os.path.join(args.out_dir, "tradeoff_dice_vs_hd95.png")
    plt.savefig(out4, dpi=200)
    plt.close()

    print("Saved:")
    print(" ", out1)
    print(" ", out2)
    print(" ", out3)
    print(" ", out4)

    # Also dump the aggregated table (useful for LaTeX table creation)
    agg_csv = os.path.join(args.out_dir, "agg_results.csv")
    agg.to_csv(agg_csv, index=False)
    print(" ", agg_csv)


if __name__ == "__main__":
    main()
