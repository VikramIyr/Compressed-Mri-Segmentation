import os
from torch.utils.data import DataLoader

from datasets import (
    set_seed,
    build_brats_index,
    split_cases,
    Brats2DSliceDataset,
)

def main():
    set_seed(0)

   
    brats_root = os.environ.get("BRATS_ROOT", "data")

   
    modalities = ["flair"]

    print("BRATS_ROOT =", brats_root)
    cases = build_brats_index(brats_root, modalities=modalities)
    cases = cases[:10]   
    train_cases, val_cases = split_cases(cases, val_ratio=0.2, seed=0)


    print("Num cases:", len(cases))
    print("Train cases:", len(train_cases), "Val cases:", len(val_cases))

    ds = Brats2DSliceDataset(
        train_cases,
        modalities=modalities,
        slice_axis=2,
        slice_stride=1,
        downsample=1,
        target_hw=(240, 240),
        keep_empty=True,
        empty_ratio=0.25,
        seed=0,
    )

    print("Num 2D slices in train dataset:", len(ds))

    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    x, y, meta = next(iter(dl))
    print("Batch x:", x.shape, x.dtype)   
    print("Batch y:", y.shape, y.dtype)   

    
    try:
        example = {k: meta[k][0] for k in meta}
    except Exception:
        example = meta
    print("Example meta[0]:", example)

    tumor_frac = (y > 0).float().mean().item()
    print("Tumor pixel fraction in batch:", tumor_frac)

    # Quick label sanity
    uniq = sorted(list(set(y.cpu().numpy().reshape(-1).tolist())))
    print("Unique labels in batch (subset):", uniq[:20])

    if tumor_frac <= 0.0:
        print("\nWARNING: batch contains no tumor pixels. This can happen by chance.")
        print("If it persists, consider keep_empty=False or reduce empty_ratio.\n")

    print("\nSanity check PASSED ✅ (if no exceptions above).")

if __name__ == "__main__":
    main()
